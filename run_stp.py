"""Evaluate STP-8 on KS50 severity-5 corruption manifests."""

import argparse
from itertools import islice
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace


CORRUPTIONS = {
    "audio": ("gaussian_noise", "traffic", "crowd", "rain", "thunder", "wind"),
    "video": (
        "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
        "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
        "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
    ),
}
AUDIO_CONFIG = {
    "num_mel_bins": 128, "target_length": 1024, "freqm": 0, "timem": 0,
    "mixup": 0, "dataset": "ks50", "mode": "eval", "mean": -5.081,
    "std": 4.4849, "noise": False, "im_res": 224,
}
DATA_SEED = 3


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", choices=("stp8_ensemble", "similarity_transfer", "shared_increment"),
        default="stp8_ensemble",
    )
    parser.add_argument("--corruption-modality", required=True, choices=CORRUPTIONS)
    parser.add_argument("--corruptions", nargs="+", help="Default: all for this modality")
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument("--label-csv", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--device", default="cuda", help="For example cuda:1 or cpu")
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--max-batches", type=int, help="Limit each corruption for a smoke test")
    args = parser.parse_args()

    args.corruptions = args.corruptions or CORRUPTIONS[args.corruption_modality]
    if len(set(args.corruptions)) != len(args.corruptions) or not set(args.corruptions).issubset(
        CORRUPTIONS[args.corruption_modality]
    ):
        parser.error("Unknown or duplicate corruption for the selected modality")
    if args.num_workers < 0 or (args.max_batches is not None and args.max_batches < 1):
        parser.error("Expected nonnegative workers and a positive batch limit")

    args.label_csv = args.label_csv or args.manifest_root / "class_labels_indices_ks50.csv"
    for path in (args.checkpoint, args.label_csv):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    if args.output_json.exists():
        parser.error(f"Output already exists: {args.output_json}")
    return args


def seed_evaluation():
    import numpy as np
    import torch

    random.seed(DATA_SEED)
    os.environ["PYTHONHASHSEED"] = str(DATA_SEED)
    np.random.seed(DATA_SEED)
    torch.manual_seed(DATA_SEED)
    torch.cuda.manual_seed(DATA_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main():
    args = parse_args()
    import torch
    from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

    from dataloader import AudiosetDataset
    from models import CAVMAEFT
    from similarity_transfer import SimilarityTransfer
    from stp import STP, SharedIncrementSTP

    adapter_type = {
        "stp8_ensemble": STP,
        "similarity_transfer": SimilarityTransfer,
        "shared_increment": SharedIncrementSTP,
    }[args.method]

    device = torch.device(args.device)
    torch.set_num_threads(4)
    seed_evaluation()
    model = CAVMAEFT(label_dim=50, modality_specific_depth=11, args=SimpleNamespace(TSA=False))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    consume_prefix_in_state_dict_if_present(checkpoint, "module.")
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)

    results = []
    for corruption in args.corruptions:
        seed_evaluation()
        manifest = args.manifest_root / args.corruption_modality / corruption / "severity_5.json"
        dataset = AudiosetDataset(str(manifest), label_csv=str(args.label_csv), audio_conf=AUDIO_CONFIG.copy())
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=16, shuffle=False, num_workers=args.num_workers,
            pin_memory=device.type == "cuda", drop_last=False,
        )
        adapter = adapter_type(model)
        samples = batches = correct = single_correct = forwards = 0
        for audio, video, labels in islice(loader, args.max_batches):
            fused, single, calls = adapter(audio.to(device), video.to(device))
            targets = labels.argmax(dim=1).to(device)
            correct += (fused.topk(1, dim=1).indices.squeeze(1) == targets).sum().item()
            single_correct += (single.topk(1, dim=1).indices.squeeze(1) == targets).sum().item()
            samples += len(labels)
            batches += 1
            forwards += calls

        if samples == 0:
            raise ValueError(f"Empty evaluation manifest: {manifest}")
        row = {
            "corruption": corruption, "samples": samples, "batches": batches,
            "correct": correct, "accuracy": 100 * correct / samples,
            "single_view_accuracy": 100 * single_correct / samples,
            "forwards_per_batch": forwards // batches, "total_forwards": forwards,
        }
        results.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)

    report = {
        "method": args.method,
        "corruption_modality": args.corruption_modality,
        "severity": 5,
        "mean_accuracy": sum(row["accuracy"] for row in results) / len(results),
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
