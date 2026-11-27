import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
import yaml
from tqdm import tqdm

from src.extractors.text.extract_text import TextFeatureExtractor

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]

DEFAULT_SPLITS_CSV = "data/splits/subject_isolated_splits.csv"
DEFAULT_OUT_DIR = "data/processed/text"


def label_from_clip_id(clip_id: str) -> int:
    """DOLOS clip names end in lie<N> or truth<N>. 1 = deceptive, 0 = truthful."""
    return 1 if "lie" in clip_id.lower() else 0


def resolve_media_path(relative_path: str) -> Path:
    """The inventory stores Windows-style paths; normalise them for any OS."""
    return ROOT / Path(str(relative_path).replace("\\", "/"))


def export_clip(sample: dict, clip_id: str, label: int, split: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "clip_id": clip_id,
        "split": split,
        "label": torch.tensor(label, dtype=torch.long),
        **sample,  # input_ids, text_feats, text_mask, alignment_indices
    }
    path = out_dir / f"{clip_id}.pt"
    torch.save(payload, path)
    return path


def export_all(cfg: dict, splits_csv: Path, out_dir: Path, split: str = None, overwrite: bool = False) -> list:
    df = pd.read_csv(splits_csv)
    if split:
        df = df[df["split"] == split]
    logger.info("Exporting %d clips to %s", len(df), out_dir)

    extractor = TextFeatureExtractor(cfg)
    failed = []

    for row in tqdm(df.itertuples(), total=len(df), desc="Exporting text features"):
        clip_id = str(row.clip_id)
        safe_id = clip_id.replace(" ", "")
        out_path = out_dir / f"{safe_id}.pt"

        if out_path.exists() and not overwrite:
            continue  # makes the run resumable

        try:
            sample = extractor.process_clip(clip_id, resolve_media_path(row.relative_path))
            export_clip(sample, safe_id, label_from_clip_id(safe_id), row.split, out_dir)
        except Exception as exc:
            logger.error("[%s] export failed: %s: %s", clip_id, type(exc).__name__, exc)
            failed.append({"clip_id": clip_id, "error": f"{type(exc).__name__}: {exc}"})

    return failed


def main():
    parser = argparse.ArgumentParser(description="Export text features to .pt files")
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--splits-csv", default=DEFAULT_SPLITS_CSV)
    parser.add_argument("--split", choices=["train", "val", "test"], default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    out_dir = ROOT / cfg.get("export", {}).get("text_dir", DEFAULT_OUT_DIR)

    failed = export_all(
        cfg=cfg,
        splits_csv=ROOT / args.splits_csv,
        out_dir=out_dir,
        split=args.split,
        overwrite=args.overwrite,
    )

    if failed:
        failed_path = out_dir / "failed_clips.csv"
        pd.DataFrame(failed).to_csv(failed_path, index=False)
        logger.warning("%d clips failed. See %s", len(failed), failed_path)
    else:
        logger.info("All clips exported successfully.")


if __name__ == "__main__":
    main()