#!/usr/bin/env python3
"""
pipeline.py: sparse COLMAP reconstruction of a folder of images, exported to PLY.

Extracted from 03_using_pycolmap.ipynb. Steps:

  1. reset    delete the previous database and reconstruction (fresh run)
  2. extract  colmap feature_extractor
  3. match    colmap exhaustive_matcher
  4. map      colmap mapper
  5. export   colmap model_converter -> <model>/sparse.ply for every model

Usage:
    python pipeline.py assets/knife/images_clean
    python pipeline.py IMAGE_DIR --workspace out/knife --threads 4
    python pipeline.py IMAGE_DIR --dry-run        # print the commands only

By default the database (pycolmap.db) and the reconstruction
(pycolmap_output/) go next to IMAGE_DIR, as in the notebook
(e.g. assets/knife/pycolmap.db, assets/knife/pycolmap_output/0/sparse.ply).

Requires the COLMAP command-line tool on PATH (or pass --colmap).
Background blurring of the input images is a separate step
(notebook 01_adding_blur_around_objects.ipynb).
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
DB_NAME = "pycolmap.db"
OUTPUT_NAME = "pycolmap_output"


def run(cmd, dry_run=False):
    print("$ " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def find_images(image_dir):
    return sorted(p for p in Path(image_dir).iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def reset_outputs(db_path, output_dir, dry_run=False):
    """Delete a previous database and reconstruction so the run starts fresh."""
    if db_path.exists():
        print(f"Removing existing database {db_path}")
        if not dry_run:
            db_path.unlink()
    if output_dir.exists():
        print(f"Removing existing reconstruction {output_dir}")
        if not dry_run:
            shutil.rmtree(output_dir)


def extract_features(colmap, db_path, image_dir, dry_run=False):
    run([colmap, "feature_extractor",
         "--database_path", str(db_path),
         "--image_path", str(image_dir)], dry_run)


def match_features(colmap, db_path, dry_run=False):
    run([colmap, "exhaustive_matcher",
         "--database_path", str(db_path)], dry_run)


def map_sparse(colmap, db_path, image_dir, output_dir, threads=8,
               init_min_tri_angle=4, dry_run=False):
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    run([colmap, "mapper",
         "--database_path", str(db_path),
         "--image_path", str(image_dir),
         "--output_path", str(output_dir),
         "--Mapper.num_threads", str(threads),
         # a bit more robust on noisy data
         "--Mapper.init_min_tri_angle", str(init_min_tri_angle)], dry_run)


def model_dirs(output_dir):
    """Reconstructed models written by the mapper: <output_dir>/0, /1, ..."""
    return sorted((d for d in output_dir.iterdir() if d.is_dir() and d.name.isdigit()),
                  key=lambda d: int(d.name))


def export_ply(colmap, model_dir, dry_run=False):
    ply_path = model_dir / "sparse.ply"
    run([colmap, "model_converter",
         "--input_path", str(model_dir),
         "--output_path", str(ply_path),
         "--output_type", "PLY"], dry_run)
    return ply_path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    p.add_argument("image_dir", type=Path, help="folder with the input images")
    p.add_argument("--workspace", type=Path, default=None,
                   help="where to write the database and reconstruction "
                        "(default: the parent of IMAGE_DIR)")
    p.add_argument("--colmap", default="colmap", help="COLMAP executable (default: colmap)")
    p.add_argument("--threads", type=int, default=8, help="mapper threads (default: 8)")
    p.add_argument("--init-min-tri-angle", type=float, default=4,
                   help="mapper initial triangulation angle in degrees (default: 4)")
    p.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    image_dir = args.image_dir
    if not image_dir.is_dir():
        sys.exit(f"error: image folder not found: {image_dir}")
    images = find_images(image_dir)
    if len(images) < 2:
        sys.exit(f"error: need at least 2 images in {image_dir}, found {len(images)}")

    colmap = args.colmap
    if not args.dry_run and shutil.which(colmap) is None:
        sys.exit(f"error: '{colmap}' not found on PATH; install COLMAP "
                 f"(https://colmap.github.io/install.html) or pass --colmap")

    workspace = args.workspace or image_dir.resolve().parent
    db_path = workspace / DB_NAME
    output_dir = workspace / OUTPUT_NAME
    print(f"{len(images)} images in {image_dir}; workspace {workspace}")
    if not args.dry_run:
        workspace.mkdir(parents=True, exist_ok=True)

    reset_outputs(db_path, output_dir, args.dry_run)
    extract_features(colmap, db_path, image_dir, args.dry_run)
    match_features(colmap, db_path, args.dry_run)
    map_sparse(colmap, db_path, image_dir, output_dir, args.threads,
               args.init_min_tri_angle, args.dry_run)

    if args.dry_run:
        export_ply(colmap, output_dir / "0", dry_run=True)
        return 0

    models = model_dirs(output_dir)
    if not models:
        sys.exit("error: COLMAP mapper produced no reconstruction "
                 "(too few matches? try more / sharper images)")
    for model in models:
        ply = export_ply(colmap, model)
        print(f"Exported model {model.name} to {ply}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
