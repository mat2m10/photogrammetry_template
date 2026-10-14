#!/usr/bin/env python3
"""
preprocess.py: blur everything outside a rectangle, to make the subject
stand out before reconstruction.

Extracted from 01_adding_blur_around_objects.ipynb, with two fixes:
- failures are no longer silent: unreadable inputs and failed writes raise
  (the notebook swallowed cv2.imwrite failures with `pass`), and load_rgb()
  raises a clear error instead of returning None.
- the 255x255 Gaussian blur is approximated by blurring a downscaled copy
  (8x smaller by default) and upscaling it, which looks the same for a
  blur this strong and is much faster.

Usage:
    python preprocess.py assets/knife/images_raw assets/knife/images_clean \\
        --keep 450 500 1800 800
"""

import argparse
import os
import sys
from glob import glob

import cv2

DEFAULT_KSIZE = 255      # blur strength of the original notebook
DEFAULT_DOWNSCALE = 8


def opencv_sigma(ksize):
    """Sigma that cv2.GaussianBlur uses for kernel size `ksize` when sigma=0."""
    return 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8


def strong_blur(img, ksize=DEFAULT_KSIZE, downscale=DEFAULT_DOWNSCALE):
    """
    Approximate cv2.GaussianBlur(img, (ksize, ksize), 0).

    A blur of sigma s on an image downscaled by d, upscaled back, is close to
    a blur of sigma s*d on the full image, at about 1/d^2 of the cost.
    """
    if ksize % 2 == 0 or ksize < 1:
        raise ValueError(f"ksize must be a positive odd number, got {ksize}")
    if downscale <= 1:
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // downscale), max(1, h // downscale)),
                       interpolation=cv2.INTER_AREA)
    sigma = opencv_sigma(ksize) / downscale
    k = 2 * int(round(3 * sigma)) + 1
    small = cv2.GaussianBlur(small, (k, k), sigma)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def clip_region(region, shape):
    """Clip (x, y, w, h) to the image; raise if nothing is left."""
    x, y, w, h = region
    h_img, w_img = shape[:2]
    x, y = max(0, x), max(0, y)
    w, h = min(w, w_img - x), min(h, h_img - y)
    if w <= 0 or h <= 0:
        raise ValueError(f"region {region} is outside the {w_img}x{h_img} image")
    return x, y, w, h


def blur_background_keep_region(img, region, ksize=DEFAULT_KSIZE,
                                downscale=DEFAULT_DOWNSCALE):
    """Return a copy of `img` blurred everywhere except `region` (x, y, w, h)."""
    x, y, w, h = clip_region(region, img.shape)
    out = strong_blur(img, ksize, downscale)
    out[y:y+h, x:x+w] = img[y:y+h, x:x+w]
    return out


def blur_background_file(src, dst, region, ksize=DEFAULT_KSIZE,
                         downscale=DEFAULT_DOWNSCALE):
    img = cv2.imread(src)
    if img is None:
        raise IOError(f"could not read image {src}")
    out = blur_background_keep_region(img, region, ksize, downscale)
    if not cv2.imwrite(dst, out):
        raise IOError(f"could not write {dst}")


def blur_folder(src_dir, dst_dir, region, pattern="*.png",
                ksize=DEFAULT_KSIZE, downscale=DEFAULT_DOWNSCALE):
    """Process every `pattern` file of src_dir into dst_dir. Returns the count."""
    paths = sorted(glob(os.path.join(src_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"no {pattern} images in {src_dir}")
    os.makedirs(dst_dir, exist_ok=True)
    failures = []
    for src in paths:
        dst = os.path.join(dst_dir, os.path.basename(src))
        try:
            blur_background_file(src, dst, region, ksize, downscale)
        except (IOError, ValueError) as e:
            failures.append(str(e))
    if failures:
        raise RuntimeError(f"{len(failures)} of {len(paths)} images failed:\n  "
                           + "\n  ".join(failures))
    return len(paths)


def load_rgb(path):
    """Load an image as RGB for plotting; raise instead of returning None."""
    img = cv2.imread(path)
    if img is None:
        hint = (" (run the background-blur step first?)"
                if "images_clean" in path else "")
        raise FileNotFoundError(f"could not read image {path}{hint}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Blur everything outside a rectangle.")
    p.add_argument("src_dir")
    p.add_argument("dst_dir")
    p.add_argument("--keep", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                   required=True, help="sharp region in pixels")
    p.add_argument("--pattern", default="*.png", help="input glob (default *.png)")
    p.add_argument("--ksize", type=int, default=DEFAULT_KSIZE,
                   help=f"Gaussian kernel size, odd (default {DEFAULT_KSIZE})")
    p.add_argument("--downscale", type=int, default=DEFAULT_DOWNSCALE,
                   help="blur on an image this many times smaller; "
                        f"1 = exact full-size blur (default {DEFAULT_DOWNSCALE})")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    try:
        n = blur_folder(a.src_dir, a.dst_dir, tuple(a.keep), a.pattern,
                        a.ksize, a.downscale)
    except (FileNotFoundError, RuntimeError) as e:
        sys.exit(f"error: {e}")
    print(f"{n} images written to {a.dst_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
