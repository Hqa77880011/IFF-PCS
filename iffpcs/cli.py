import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, label
from sklearn.metrics import roc_auc_score

from .method import IFFPCS, objective


def prepare(args):
    manifest = Path(args.manifest)
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    groups = {}
    for i, row in enumerate(rows):
        group, split = row["group"], row["split"]
        if group in groups and groups[group] != split:
            raise ValueError("A source scene crosses partitions")
        groups[group] = split
        base = manifest.resolve().parent
        a, b = [np.asarray(Image.open(base / row[k]).convert("RGB").resize((args.size, args.size), Image.Resampling.BILINEAR), np.float32) / 255 for k in ["before", "after"]]
        y = np.asarray(Image.open(base / row["mask"]).convert("L").resize((args.size, args.size), Image.Resampling.NEAREST)) > 0
        na, nb = [(x - x.mean((0, 1))) / np.maximum(x.std((0, 1)), 1e-5) for x in [a, b]]
        radiometric = np.abs(nb - na).mean(-1)
        band = binary_dilation(y, iterations=2) ^ binary_erosion(y, iterations=2)
        eligible = ~y & ~band
        proxy = eligible & (radiometric > np.quantile(radiometric[eligible], args.percentile)) if eligible.any() else np.zeros_like(y)
        component, count = label(y)
        area = np.bincount(component.ravel(), minlength=count + 1)
        small = y & (area[component] <= args.small_area)
        target = out / split
        target.mkdir(exist_ok=True)
        np.savez_compressed(target / f"{i:07d}.npz", a=a.transpose(2, 0, 1), b=b.transpose(2, 0, 1),
                            y=y, proxy=proxy, small=small, dataset=str(row.get("dataset", "dataset")), group=str(group))


def load_batch(paths, device, stress="none"):
    data = [dict(np.load(path, allow_pickle=False)) for path in paths]
    a, b, y, proxy = [torch.as_tensor(np.stack([d[k] for d in data]), device=device) for k in ["a", "b", "y", "proxy"]]
    if stress == "brightness":
        b = b * 0.8
    elif stress == "noise":
        generator = torch.Generator(device=device).manual_seed(3407)
        b = (b + 0.03 * torch.randn(b.shape, generator=generator, device=device)).clamp(0, 1)
    elif stress == "translation":
        b = torch.nn.functional.pad(b[..., :, :-5], (5, 0, 0, 0))
    return a.float(), b.float(), y[:, None].float(), proxy[:, None].bool(), data


def paths(root, split):
    result = sorted((Path(root) / split).glob("*.npz"))
    if not result:
        raise ValueError(f"No {split} pairs")
    return result


def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    net = IFFPCS(args.mode, args.gate).to(args.device)
    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=args.lr)
    files, development = paths(args.data, "train"), paths(args.data, "dev")
    files = [files[i] for i in rng.permutation(len(files))[:max(1, int(len(files) * args.fraction))]]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    best = -1
    for _ in range(args.epochs):
        net.train()
        order = rng.permutation(len(files))
        for start in range(0, len(files), args.batch_size):
            batch = [files[i] for i in order[start:start + args.batch_size]]
            a, b, y, proxy, _ = load_batch(batch, args.device)
            logits, gate, distance = net(a, b)
            loss = objective(logits, gate, distance, y, proxy, args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        net.eval()
        counts = np.zeros((9, 3), float)
        with torch.no_grad():
            for path in development:
                a, b, y, _, _ = load_batch([path], args.device)
                probability = net(a, b)[0].sigmoid()
                for j, threshold in enumerate(np.linspace(0.2, 0.6, 9)):
                    pred = probability >= threshold
                    true = y.bool()
                    counts[j] += [int((pred & true).sum()), int((pred & ~true).sum()), int((~pred & true).sum())]
        f1 = 2 * counts[:, 0] / np.maximum(2 * counts[:, 0] + counts[:, 1] + counts[:, 2], 1)
        j = int(f1.argmax())
        if f1[j] > best:
            best = float(f1[j])
            torch.save({"model": net.state_dict(), "mode": args.mode, "gate": args.gate,
                        "threshold": float(np.linspace(0.2, 0.6, 9)[j])}, out / "model.pt")


def evaluate(args):
    saved = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    net = IFFPCS(saved["mode"], saved["gate"], args.permute_gate).to(args.device)
    net.load_state_dict(saved["model"])
    net.eval()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows, by_dataset = [], {}
    with torch.no_grad():
        for path in paths(args.data, "test"):
            start = perf_counter()
            a, b, y, proxy, data = load_batch([path], args.device, args.stress)
            logits, gate, distance = net(a, b)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = perf_counter() - start
            p = logits.sigmoid().cpu().numpy()[0, 0]
            truth, proxy, g, d = data[0]["y"].astype(bool), data[0]["proxy"].astype(bool), gate.cpu().numpy()[0, 0], distance.cpu().numpy()[0]
            pred = p >= saved["threshold"]
            boundary_pred, boundary_true = pred ^ binary_erosion(pred), truth ^ binary_erosion(truth)
            tp, fp, fn = int((pred & truth).sum()), int((pred & ~truth).sum()), int((~pred & truth).sum())
            changed = truth & (d[2] > np.quantile(d[2][truth], 0.5)) if truth.any() else np.zeros_like(truth)
            auc = roc_auc_score(np.r_[np.ones(proxy.sum()), np.zeros(changed.sum())], np.r_[g[proxy], g[changed]]) if proxy.any() and changed.any() else None
            precision = (boundary_pred & binary_dilation(boundary_true, iterations=2)).sum() / max(1, boundary_pred.sum())
            recall = (boundary_true & binary_dilation(boundary_pred, iterations=2)).sum() / max(1, boundary_true.sum())
            row = {"pair": path.stem, "dataset": str(data[0]["dataset"]), "tp": tp, "fp": fp, "fn": fn,
                   "bf1": float(2 * precision * recall / max(precision + recall, 1e-12)),
                   "proxy_fp": int((pred & proxy).sum()), "proxy_pixels": int(proxy.sum()),
                   "small_tp": int((pred & data[0]["small"]).sum()), "small_pixels": int(data[0]["small"].sum()),
                   "gate_auc": float(auc) if auc is not None else None,
                   "mean_gate_proxy": float(g[proxy].mean()) if proxy.any() else None,
                   "mean_gate_change": float(g[changed].mean()) if changed.any() else None,
                   "seconds": elapsed}
            for name, mask in [("appearance", proxy), ("small", data[0]["small"]), ("ordinary", ~truth & ~proxy)]:
                row[name + "_profile"] = d[:, mask].mean(1).tolist() if mask.any() else None
            rows.append(row)
            by_dataset.setdefault(row["dataset"], []).append(row)
            Image.fromarray((pred * 255).astype(np.uint8)).save(out / f"{path.stem}.png")
    metrics = {}
    for dataset, group in by_dataset.items():
        tp, fp, fn, proxy_fp, proxy_pixels, small_tp, small_pixels = [sum(r[k] for r in group) for k in ["tp", "fp", "fn", "proxy_fp", "proxy_pixels", "small_tp", "small_pixels"]]
        metrics[dataset] = {"f1": 2 * tp / max(1, 2 * tp + fp + fn), "iou": tp / max(1, tp + fp + fn),
                            "bf1": float(np.mean([r["bf1"] for r in group])),
                            "pfr": proxy_fp / proxy_pixels if proxy_pixels else None,
                            "small_change_recall": small_tp / small_pixels if small_pixels else None,
                            "fps": len(group) / sum(r["seconds"] for r in group)}
    payload = {"datasets": metrics, "trainable_parameters": sum(p.numel() for p in net.parameters() if p.requires_grad), "threshold": saved["threshold"]}
    (out / "metrics.json").write_text(json.dumps(payload, indent=2, allow_nan=False))
    (out / "pairs.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(metrics), [m["f1"] for m in metrics.values()])
    ax.set(ylabel="Change F1")
    fig.tight_layout()
    fig.savefig(out / "summary.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", default="data/prepared")
    p.add_argument("--size", type=int, default=518)
    p.add_argument("--percentile", type=float, default=0.85)
    p.add_argument("--small-area", type=int, default=64)
    p.set_defaults(function=prepare)
    for command in ["train", "evaluate"]:
        p = sub.add_parser(command)
        p.add_argument("--data", default="data/prepared")
        p.add_argument("--output", default=f"runs/{command}")
        p.add_argument("--device", default="cpu")
        if command == "train":
            p.add_argument("--mode", choices=["all", "deep", "middle", "rgb"], default="all")
            p.add_argument("--gate", action=argparse.BooleanOptionalAction, default=True)
            for flag in ["no-boundary", "no-pseudo", "no-preserve"]:
                p.add_argument("--" + flag, action="store_true")
            for key, value in [("pseudo-weight", 0.2), ("preserve-weight", 0.1), ("lr", 3e-4), ("fraction", 1.0)]:
                p.add_argument("--" + key, type=float, default=value)
            for key, value in [("epochs", 40), ("batch-size", 4), ("seed", 3407)]:
                p.add_argument("--" + key, type=int, default=value)
            p.set_defaults(function=train)
        else:
            p.add_argument("--checkpoint", default="runs/train/model.pt")
            p.add_argument("--permute-gate", action="store_true")
            p.add_argument("--stress", choices=["none", "brightness", "noise", "translation"], default="none")
            p.set_defaults(function=evaluate)
    args = parser.parse_args()
    args.function(args)
