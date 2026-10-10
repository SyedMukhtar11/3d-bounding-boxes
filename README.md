# YouTube Video 3D Bounding Boxes

Draw estimated 3D bounding boxes on any YouTube video (including Shorts) using only a single camera feed. No LiDAR, no calibration files, no 3D training data needed.

The script downloads a video, detects and tracks objects, estimates metric depth, fits an oriented 3D box to each object, and writes a new video with the boxes drawn on top.

## How it works

```
YouTube URL
    |
    v
yt-dlp (download, up to 720p)
    |
    v
YOLO11-seg + tracking  -->  object masks + persistent IDs
    |
    v
Depth Anything V2 (metric)  -->  per-pixel depth in meters
    |
    v
Back-project mask pixels to 3D camera coordinates
    |
    v
Fit yaw-rotated box (minAreaRect on the ground plane + height)
    |
    v
Temporal smoothing per track ID
    |
    v
Project 8 corners to the image and draw wireframe + label
```

## Installation

Python 3.9+ is recommended. A CUDA GPU is optional but makes processing much faster.

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install yt-dlp ultralytics transformers torch opencv-python pillow numpy
```

The first run downloads the YOLO and Depth Anything model weights automatically.

## Usage

```bash
python video_3d_boxes.py "https://www.youtube.com/watch?v=VIDEO_ID" --out result.mp4
```

Examples:

```bash
# YouTube Short (portrait video, narrower field of view)
python video_3d_boxes.py "https://youtube.com/shorts/VIDEO_ID" --hfov 50

# Indoor scene
python video_3d_boxes.py "URL" --scene indoor

# Only people, cars, buses and trucks (COCO class ids)
python video_3d_boxes.py "URL" --classes 0 2 5 7
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `url` | required | YouTube video or Shorts URL (keep it in quotes) |
| `--out` | `result.mp4` | Output video path |
| `--scene` | `outdoor` | `outdoor` or `indoor`, selects the metric depth model |
| `--hfov` | `65` | Assumed horizontal field of view in degrees |
| `--classes` | all | COCO class ids to keep, e.g. `0` = person, `2` = car |

## Tuning tips

- **Boxes look too deep or too flat:** adjust `--hfov`. Typical values are 40-70 for phones and dashcams. Portrait video usually needs a smaller value.
- **Wrong depth scale:** switch `--scene` between `indoor` and `outdoor`.
- **Too many irrelevant boxes:** use `--classes` to filter.
- **Slow on CPU:** use a short video, or lower the download resolution in the `download()` function.

## Limitations

- The boxes are **estimates**. A normal video has no camera intrinsics or true depth, so the field of view is assumed and depth is predicted by a neural network.
- Boxes only fit the **visible surface** of an object. The hidden far side is never observed, so size and depth can be underestimated.
- Orientation is ambiguous for round or symmetric objects.
- Detection uses COCO classes only (YOLO).
- Boxes are yaw-only (rotated around the vertical axis), so pitch and roll are not modeled.

## Better accuracy: alternatives

If you need true learned 3D detection rather than this geometric approach:

- [Cube R-CNN (Omni3D)](https://github.com/facebookresearch/omni3d) - monocular 3D detector for in-the-wild images
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) - SMOKE, MonoFlex, FCOS3D for driving-style footage
- [MediaPipe Objectron](https://github.com/google/mediapipe) - lightweight, limited to a few object categories

## Built with

- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) via Hugging Face Transformers
- [OpenCV](https://opencv.org/)

## Disclaimer

Only process videos you have the right to use. Respect YouTube's Terms of Service and the video creator's copyright.

## License

Add a license of your choice (for example MIT) as a `LICENSE` file. Note that Ultralytics YOLO is released under AGPL-3.0, which may affect how you can distribute this project.
