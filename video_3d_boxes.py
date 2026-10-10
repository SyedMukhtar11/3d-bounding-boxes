"""
Draw 3D bounding boxes on a YouTube video.

Pipeline: yt-dlp -> YOLO11-seg + tracking -> Depth Anything V2 (metric) ->
back-project object pixels to 3D -> fit oriented box -> project 8 corners -> draw.

Install:
    pip install yt-dlp ultralytics transformers torch opencv-python pillow numpy

Run:
    python video_3d_boxes.py "https://www.youtube.com/watch?v=XXXX" --out result.mp4
"""
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yt_dlp
from PIL import Image
from transformers import pipeline
from ultralytics import YOLO

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),      # bottom face
         (4, 5), (5, 6), (6, 7), (7, 4),      # top face
         (0, 4), (1, 5), (2, 6), (3, 7)]      # pillars

DEPTH_MODELS = {
    "outdoor": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    "indoor": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
}


def download(url, out="input.mp4"):
    opts = {"format": "mp4[height<=720]/best[height<=720]",
            "outtmpl": out, "quiet": True, "overwrites": True}
    with yt_dlp.YoutubeDL(opts) as y:
        y.download([url])
    return out


def intrinsics(w, h, hfov_deg):
    fx = w / (2 * np.tan(np.radians(hfov_deg) / 2))
    return fx, fx, w / 2, h / 2


def box_from_mask(mask, depth, K):
    """Back-project masked pixels to camera coords (X right, Y down, Z forward)
    and fit a yaw-only oriented box (ground = XZ plane)."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 200:
        return None
    z = depth[ys, xs]
    lo, hi = np.percentile(z, [10, 90])          # drop depth outliers / mask bleed
    keep = (z >= lo) & (z <= hi) & (z > 0.1)
    if keep.sum() < 100:
        return None
    xs, ys, z = xs[keep], ys[keep], z[keep]
    fx, fy, cx, cy = K
    X = (xs - cx) * z / fx
    Y = (ys - cy) * z / fy
    rect = cv2.minAreaRect(np.stack([X, z], 1).astype(np.float32))
    bottom = cv2.boxPoints(rect)                 # 4x2 in (X, Z)
    y_min, y_max = np.percentile(Y, [2, 98])
    return bottom, y_min, y_max


def smooth(prev, new, alpha=0.6):
    """EMA over (bottom, y_min, y_max); aligns corner order to previous frame."""
    if prev is None:
        return new
    pb, pmin, pmax = prev
    nb, nmin, nmax = new
    best = min(range(4), key=lambda s: np.linalg.norm(np.roll(nb, s, 0) - pb))
    nb = np.roll(nb, best, 0)
    return (alpha * pb + (1 - alpha) * nb,
            alpha * pmin + (1 - alpha) * nmin,
            alpha * pmax + (1 - alpha) * nmax)


def corners_3d(box):
    bottom, y_min, y_max = box
    low = [[x, y_max, z] for x, z in bottom]     # Y is down, so y_max = ground
    high = [[x, y_min, z] for x, z in bottom]
    return np.array(low + high, dtype=np.float32)


def project(pts, K):
    fx, fy, cx, cy = K
    if (pts[:, 2] <= 0.1).any():
        return None
    u = fx * pts[:, 0] / pts[:, 2] + cx
    v = fy * pts[:, 1] / pts[:, 2] + cy
    return np.stack([u, v], 1).astype(int)


def color_for(i):
    rng = np.random.RandomState(i * 7 + 3)
    return tuple(int(c) for c in rng.randint(80, 255, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--out", default="result.mp4")
    ap.add_argument("--scene", choices=["outdoor", "indoor"], default="outdoor")
    ap.add_argument("--hfov", type=float, default=65.0,
                    help="assumed horizontal field of view in degrees")
    ap.add_argument("--classes", nargs="*", type=int, default=None,
                    help="COCO class ids to keep, e.g. 0 2 5 7 (person car bus truck)")
    args = ap.parse_args()

    device = 0 if torch.cuda.is_available() else -1
    video = download(args.url)
    yolo = YOLO("yolo11n-seg.pt")
    depth_pipe = pipeline("depth-estimation", model=DEPTH_MODELS[args.scene], device=device)

    cap = cv2.VideoCapture(video)
    W, H = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    K = intrinsics(W, H, args.hfov)
    tracks = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # metric depth (meters), resized to frame size
        d = depth_pipe(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))["predicted_depth"]
        d = d.reshape(1, 1, *d.shape[-2:]).float().cpu()
        depth = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)[0, 0].numpy()

        res = yolo.track(frame, persist=True, verbose=False, classes=args.classes)[0]
        if res.masks is not None:
            ids = res.boxes.id.int().tolist() if res.boxes.id is not None else list(range(len(res.masks.xy)))
            clss = res.boxes.cls.int().tolist()
            for poly, tid, c in zip(res.masks.xy, ids, clss):
                if len(poly) < 3:
                    continue
                mask = np.zeros((H, W), np.uint8)
                cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
                box = box_from_mask(mask, depth, K)
                if box is None:
                    continue
                tracks[tid] = box = smooth(tracks.get(tid), box)
                pts2d = project(corners_3d(box), K)
                if pts2d is None:
                    continue
                col = color_for(tid)
                for a, b in EDGES:
                    cv2.line(frame, tuple(pts2d[a]), tuple(pts2d[b]), col, 2, cv2.LINE_AA)
                dist = float(np.mean(box[0][:, 1]))
                cv2.putText(frame, f"{res.names[c]} #{tid} {dist:.1f}m",
                            tuple(pts2d[4]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        writer.write(frame)

    cap.release()
    writer.release()
    print("Saved", args.out)


if __name__ == "__main__":
    main()
