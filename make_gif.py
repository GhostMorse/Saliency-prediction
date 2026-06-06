"""Overlay predicted saliency on a video and save an animated GIF (qualitative).

    python make_gif.py --video data/public_tests/01_test_file_input/test/0001/source.mp4 \
        --pred-dir outputs/0001 --output out.gif
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import imageio
import numpy as np

from saliency.utils import normalize_map, padding


def main() -> None:
    p = argparse.ArgumentParser(description="Make a saliency-overlay GIF.")
    p.add_argument("--video", type=Path, required=True, help="Path to source.mp4")
    p.add_argument("--pred-dir", type=Path, required=True, help="Folder with predicted NNNN.png maps")
    p.add_argument("--output", type=Path, default=Path("out.gif"))
    p.add_argument("--alpha", type=float, default=0.7)
    args = p.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    with imageio.get_writer(str(args.output), mode="I") as writer:
        for name in sorted(os.listdir(args.pred_dir)):
            if not name.endswith(".png"):
                continue
            ok, frame = cap.read()
            if not ok:
                break
            pred = cv2.imread(str(args.pred_dir / name), cv2.IMREAD_GRAYSCALE)
            heat = (1 - padding(normalize_map(pred), frame.shape[0], frame.shape[1])) * 255
            heatmap = cv2.applyColorMap(heat.astype(np.uint8), cv2.COLORMAP_JET)
            blended = np.clip(args.alpha * frame + (1 - args.alpha) * heatmap, 0, 255).astype(np.uint8)
            writer.append_data(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    cap.release()
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
