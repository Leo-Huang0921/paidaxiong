from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


CLASS_NAMES = ["apple", "banana", "grape", "strawberry"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the RAICOM fruit YOLO detector.")
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=0, help="Batch size. 0 selects a device-aware default.")
    parser.add_argument("--model", type=str, default="", help="Optional explicit .pt/.yaml model path.")
    parser.add_argument("--name", type=str, default="fruit_yolov8_competition", help="Ultralytics run name.")
    parser.add_argument("--patience", type=int, default=35, help="Early-stopping patience.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument("--dry-run", action="store_true", help="Check paths and print train settings without training.")
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def count_files(path: Path, suffixes: set[str] | None = None) -> int:
    if suffixes is None:
        return sum(1 for item in path.iterdir() if item.is_file())
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in suffixes)


def validate_dataset(root: Path) -> None:
    data_root = root / "data" / "datasets_modified"
    required_dirs = [
        data_root / "images" / "train",
        data_root / "images" / "val",
        data_root / "labels" / "train",
        data_root / "labels" / "val",
    ]
    missing = [path for path in required_dirs if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing dataset directories: " + ", ".join(str(path) for path in missing))

    for split in ("train", "val"):
        image_dir = data_root / "images" / split
        label_dir = data_root / "labels" / split
        image_stems = {path.stem for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS}
        label_stems = {path.stem for path in label_dir.glob("*.txt")}
        if image_stems != label_stems:
            missing_labels = sorted(image_stems - label_stems)[:10]
            missing_images = sorted(label_stems - image_stems)[:10]
            raise RuntimeError(
                f"Image/label mismatch in {split}: "
                f"missing_labels={missing_labels}, missing_images={missing_images}"
            )

    train_images = count_files(data_root / "images" / "train", IMAGE_EXTS)
    val_images = count_files(data_root / "images" / "val", IMAGE_EXTS)
    if train_images == 0 or val_images == 0:
        raise RuntimeError(f"Dataset split is empty: train={train_images}, val={val_images}")


def candidate_models(root: Path, prefer_small: bool) -> list[Path | str]:
    local_n = root / "models" / "yolov8n.pt"
    local_s = root / "models" / "yolov8s.pt"
    external_n = Path("D:/ruikang/Yolov8/yolov8n.pt")
    external_s = Path("D:/ruikang/Yolov8/yolov8s.pt")
    if prefer_small:
        preferred = [local_n, local_s, external_n, external_s]
    else:
        preferred = [local_s, local_n, external_s, external_n]

    names = [
        *preferred,
        root / "models" / "best.pt",
        root / "models" / "last.pt",
    ]

    yolo_root = Path("D:/ruikang/Yolov8")
    if yolo_root.exists():
        names.extend(sorted(yolo_root.rglob("yolov8s.pt")))
        names.extend(sorted(yolo_root.rglob("yolov8n.pt")))

    existing: list[Path | str] = []
    seen: set[str] = set()
    for path in names:
        key = str(path).lower()
        if key not in seen and isinstance(path, Path) and path.exists():
            existing.append(path)
            seen.add(key)
    if not existing:
        # No local pretrained weights were found. This uses the model structure from
        # the installed ultralytics package and avoids a hidden network download.
        existing.append("yolov8n.yaml")
    return existing


def resolve_model(root: Path, explicit_model: str, prefer_small: bool) -> Path | str:
    if explicit_model:
        path = Path(explicit_model)
        if path.exists():
            return path
        return explicit_model
    return candidate_models(root, prefer_small)[0]


def device_and_batch(batch_arg: int) -> tuple[int | str, int, int, bool]:
    cuda = torch.cuda.is_available()
    device: int | str = 0 if cuda else "cpu"
    batch = batch_arg if batch_arg > 0 else (16 if cuda else 8)
    workers = 4 if cuda else 0
    amp = cuda
    return device, batch, workers, amp


def copy_final_weights(save_dir: Path, root: Path) -> Path:
    best_path = save_dir / "weights" / "best.pt"
    last_path = save_dir / "weights" / "last.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Training did not produce best.pt: {best_path}")

    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    final_best = models_dir / "fruit_best.pt"
    shutil.copy2(best_path, final_best)
    if last_path.exists():
        shutil.copy2(last_path, models_dir / "fruit_last.pt")
    return final_best


def write_metadata(root: Path, save_dir: Path, args: argparse.Namespace, model_path: Path | str, device: int | str) -> None:
    metadata = {
        "dataset": str(root / "data" / "datasets_modified"),
        "data_yaml": str(root / "configs" / "fruit.yaml"),
        "source_model": str(model_path),
        "device": str(device),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "run_dir": str(save_dir),
        "classes": CLASS_NAMES,
    }
    with (root / "models" / "fruit_best_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    root = project_root()
    data_config = root / "configs" / "fruit.yaml"
    if not data_config.exists():
        raise FileNotFoundError(f"Missing data config: {data_config}")
    validate_dataset(root)

    device, batch, workers, amp = device_and_batch(args.batch)
    model_path = resolve_model(root, args.model, prefer_small=device == "cpu")
    args.batch = batch

    settings = {
        "data": str(data_config),
        "model": str(model_path),
        "device": device,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": batch,
        "workers": workers,
        "output": str(root / "models" / "fruit_best.pt"),
    }
    print(json.dumps(settings, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    model = YOLO(str(model_path))
    results = model.train(
        data=str(data_config),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        workers=workers,
        device=device,
        project=str(root / "runs" / "train"),
        name=args.name,
        exist_ok=True,
        pretrained=str(model_path).endswith(".pt"),
        seed=args.seed,
        deterministic=True,
        patience=args.patience,
        optimizer="auto",
        cos_lr=True,
        close_mosaic=15,
        amp=amp,
        cache=False,
        plots=True,
        val=True,
        hsv_h=0.015,
        hsv_s=0.55,
        hsv_v=0.35,
        degrees=8.0,
        translate=0.08,
        scale=0.45,
        fliplr=0.5,
        mosaic=0.7,
        mixup=0.05,
        copy_paste=0.0,
        erasing=0.0,
    )

    save_dir = Path(results.save_dir)
    final_best = copy_final_weights(save_dir, root)
    best_model = YOLO(str(final_best))
    best_model.val(data=str(data_config), imgsz=args.imgsz, batch=batch, device=device, workers=workers, plots=True)
    write_metadata(root, save_dir, args, model_path, device)
    print(f"Final competition weights: {final_best}")


if __name__ == "__main__":
    main()
