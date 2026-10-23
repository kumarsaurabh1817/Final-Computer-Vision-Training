# Day 4 Notes — Multi-Object Tracking

## Kaggle dataset
- Primary: [mnhlfuch/mot17-02-frcnn](https://www.kaggle.com/datasets/mnhlfuch/mot17-02-frcnn) (~96 MB). One MOT17 sequence: 600 frames, 1920×1080, 30 fps, with `gt/gt.txt` track IDs.
- Fallback (automatic): [wenhoujinjust/mot-17](https://www.kaggle.com/datasets/wenhoujinjust/mot-17). The notebook pulls only `MOT17/train/MOT17-04-FRCNN` when the directory download works, otherwise the full ~5.6 GB.
- Your own clip: set `VIDEO_PATH`. You get the tracked video and ID statistics, but no metrics without ground truth.

## Tracking-by-detection
1. **Detect** objects in every frame.
2. **Predict** where each existing track should be now (Kalman filter, constant-velocity model).
3. **Associate** detections to predicted tracks (Hungarian algorithm on an IoU and/or appearance cost).
4. **Manage** track lifecycles: start new tracks, keep lost ones for `track_buffer` frames, delete stale ones.

| Tracker | Motion | Appearance | Key trick |
|---|---|---|---|
| Naive IoU (built here) | none | none | Baseline |
| SORT | Kalman | none | Kalman + Hungarian |
| DeepSORT | Kalman | ReID CNN embedding | Re-identifies after occlusion |
| **ByteTrack** | Kalman | none | Second matching round with **low-score** boxes, which rescues occluded people |
| **BoT-SORT** | Kalman + camera-motion compensation | optional ReID | Robust when the camera moves |

## MOT metrics (implemented in the notebook)
- **MOTA** = 1 − (FN + FP + IDSW) / #GT. Mostly measures detection quality.
- **IDF1**: F1 of identity-consistent matches after a global one-to-one GT↔track ID assignment. This is the best single number for "stable IDs".
- **IDSW**: number of times a person's matched track ID changes. **Frag**: number of times a GT trajectory is interrupted.
- MOT17 GT includes heavily occluded people, so recall and MOTA stay well below 1 even for strong trackers. Compare trackers with each other, not against 1.0.

## What to check in your run
- IDF1 and IDSW: naive → ByteTrack → BoT-SORT.
- Track-lifetime plot: few long lines are good; many short ones mean fragmentation.
- Watch the video: where do IDs switch? Usually at crossings and occlusions.

## Tuning knobs (`*_custom.yaml` written by the notebook)
- `track_buffer`: how long a lost track survives. Raise it for long occlusions.
- `track_high_thresh` / `track_low_thresh`: score cut-offs for the two ByteTrack matching rounds.
- `match_thresh`: association strictness.
- BoT-SORT `with_reid: True`: appearance matching (slower, fewer switches after occlusion).

## Pitfalls
- Keep the detector `conf` low (0.1). ByteTrack needs the low-score boxes; it applies its own thresholds internally.
- `persist=True` keeps tracker state between calls. Use a fresh model per run, or IDs leak between experiments.
- OpenCV writes `mp4v`. Browsers need H.264, so the notebook re-encodes with ffmpeg.

## Materials
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [Ultralytics — Object tracking](https://docs.ultralytics.com/modes/track/)
