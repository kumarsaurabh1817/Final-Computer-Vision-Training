"""
Week 3 · Day 4 — Multi-object tracking: naive IoU tracker vs ByteTrack vs BoT-SORT on MOT17.

Colab-first script (percent-format cells). Build the notebook with:
    python ../tools/py_to_ipynb.py day4_object_tracking/day4_object_tracking.py
Kaggle dataset: https://www.kaggle.com/datasets/mnhlfuch/mot17-02-frcnn  (one MOT17 sequence, ~96 MB)
Fallback:       https://www.kaggle.com/datasets/wenhoujinjust/mot-17     (full MOT17, ~5.6 GB)
"""

# %% [markdown]
# # Week 3 · Day 4 — Multi-Object Tracking
#
# **Goal:** bolt a tracker onto a detector, produce a tracked video with **stable IDs**, and *measure* stability instead of eyeballing it.
#
# Tracking-by-detection in one sentence: run a detector on every frame, then **associate** detections across frames into tracks with a motion model (Kalman filter) plus a matching cost (IoU, optionally appearance).
#
# | Tracker | Association | Notes |
# |---|---|---|
# | Naive IoU tracker (built here) | Hungarian matching on IoU, high-confidence boxes only | Baseline: breaks on occlusion and fast motion |
# | **ByteTrack** | Kalman prediction + two-stage matching: high-score boxes first, then **low-score** boxes to rescue occluded tracks | Fast, no appearance model |
# | **BoT-SORT** | ByteTrack-style matching + camera-motion compensation (+ optional ReID appearance, the DeepSORT idea) | Usually fewer ID switches when the camera moves |
#
# **Dataset:** [MOT17-02 on Kaggle (mnhlfuch/mot17-02-frcnn)](https://www.kaggle.com/datasets/mnhlfuch/mot17-02-frcnn) — a 600-frame, 30 fps, 1920×1080 pedestrian sequence **with ground-truth track IDs** (~96 MB). That lets us compute MOTA, IDF1, and ID switches. The full [MOT17 Kaggle mirror](https://www.kaggle.com/datasets/wenhoujinjust/mot-17) is the fallback. You can also point `VIDEO_PATH` at your own clip (no GT → no metrics, just the tracked video).
#
# > Runtime: **T4 GPU**. Each tracker pass over 600 frames at `imgsz=1280` takes ~1–2 min.

# %% [markdown]
# ## 0 · Install dependencies

# %%
import subprocess
import sys


def pip_install(*packages: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


pip_install("ultralytics==8.4.182", "kagglehub==1.0.2", "lap==0.5.12")  # lap: linear assignment used by the trackers

# %% [markdown]
# ## 1 · Configuration

# %%
import configparser
import json
import shutil
import time
from collections import defaultdict, deque
from pathlib import Path

import matplotlib

IN_NOTEBOOK = "ipykernel" in sys.modules
if not IN_NOTEBOOK:
    matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from ultralytics.utils import ROOT as ULTRALYTICS_ROOT

KAGGLE_HANDLE = "mnhlfuch/mot17-02-frcnn"
FALLBACK_HANDLE = "wenhoujinjust/mot-17"
FALLBACK_SEQUENCE = "MOT17/train/MOT17-04-FRCNN"
VIDEO_PATH = None  # e.g. "/content/my_clip.mp4" to track your own video (skips GT metrics)

MODEL_WEIGHTS = "yolo26s.pt"  # COCO-pretrained; class 0 = person. Use Day 2's best.pt for your own classes
IMGSZ = 1280  # MOT pedestrians are small; 1280 recovers many of them
DET_CONF = 0.1  # keep low-score boxes: ByteTrack's second matching stage needs them
NAIVE_MIN_CONF = 0.4
MAX_FRAMES = None  # e.g. 300 for a quicker run
TRACK_BUFFER = 30  # frames a lost track is kept alive (≈1 s at 30 fps)
OUTPUT_WIDTH = 1280  # rendered video width
MATCH_IOU = 0.5  # IoU needed for a prediction to count as matching a GT box in the metrics

IN_COLAB = "google.colab" in sys.modules
try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()
OUTPUT_DIR = Path("/content/week3_outputs/day4") if IN_COLAB else HERE.parent / "results" / "day4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = 0 if torch.cuda.is_available() else "cpu"
print({"device": DEVICE, "model": MODEL_WEIGHTS, "imgsz": IMGSZ})


def save_fig(fig, name: str) -> None:
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    if IN_NOTEBOOK:
        plt.show()
    plt.close(fig)
    print(f"saved -> {path}")


# %% [markdown]
# ## 2 · Get the sequence (frames + ground truth)
#
# MOTChallenge layout: `img1/000001.jpg …`, `gt/gt.txt`, and `seqinfo.ini` (frame rate, resolution). Each `gt.txt` row is
# `frame, track_id, x, y, w, h, consider_flag, class, visibility`. Class 1 = pedestrian; classes 2/7/8/12 (person on vehicle, static person, distractor, reflection) are **ignore regions**: detections there are neither rewarded nor penalized.

# %%
import kagglehub


def find_sequence(root: Path) -> Path:
    for img_dir in sorted(root.rglob("img1")):
        if (img_dir.parent / "gt" / "gt.txt").exists():
            return img_dir.parent
    raise FileNotFoundError(f"No MOT sequence with gt/gt.txt below {root}")


SEQ_DIR = None
if VIDEO_PATH is None:
    try:
        SEQ_DIR = find_sequence(Path(kagglehub.dataset_download(KAGGLE_HANDLE)))
    except Exception as exc:
        print(f"Primary dataset unavailable ({exc}); downloading {FALLBACK_SEQUENCE} from {FALLBACK_HANDLE}")
        try:
            SEQ_DIR = find_sequence(Path(kagglehub.dataset_download(FALLBACK_HANDLE, path=FALLBACK_SEQUENCE)))
        except Exception as exc2:
            print(f"Directory download failed ({exc2}); downloading the full dataset (~5.6 GB)")
            SEQ_DIR = find_sequence(Path(kagglehub.dataset_download(FALLBACK_HANDLE)))
    print("Sequence:", SEQ_DIR)

if SEQ_DIR is not None:
    FRAME_PATHS = sorted((SEQ_DIR / "img1").glob("*.jpg"))
    info = configparser.ConfigParser()
    info.read(SEQ_DIR / "seqinfo.ini")
    FPS = float(info.get("Sequence", "frameRate", fallback="30"))
    if MAX_FRAMES:
        FRAME_PATHS = FRAME_PATHS[:MAX_FRAMES]
    first_frame = cv2.imread(str(FRAME_PATHS[0]))
    SEQ_NAME = SEQ_DIR.name
else:
    cap = cv2.VideoCapture(VIDEO_PATH)
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, first_frame = cap.read()
    cap.release()
    SEQ_NAME = Path(VIDEO_PATH).stem
H, W = first_frame.shape[:2]
print(f"{SEQ_NAME}: {W}x{H} @ {FPS:.0f} fps")


def iterate_frames():
    """Yield (frame_number starting at 1, BGR image) from the sequence or the custom video."""
    if SEQ_DIR is not None:
        for i, path in enumerate(FRAME_PATHS, start=1):
            yield i, cv2.imread(str(path))
    else:
        cap = cv2.VideoCapture(VIDEO_PATH)
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok or (MAX_FRAMES and i >= MAX_FRAMES):
                break
            i += 1
            yield i, frame
        cap.release()


N_FRAMES = len(FRAME_PATHS) if SEQ_DIR is not None else sum(1 for _ in iterate_frames())
print("frames:", N_FRAMES)

# %%
GT, IGNORE = {}, {}
if SEQ_DIR is not None:
    gt = np.loadtxt(SEQ_DIR / "gt" / "gt.txt", delimiter=",", ndmin=2)
    gt = gt[gt[:, 0] <= N_FRAMES]
    to_xyxy = lambda r: np.stack([r[:, 2], r[:, 3], r[:, 2] + r[:, 4], r[:, 3] + r[:, 5]], axis=1)
    peds = gt[(gt[:, 7] == 1) & (gt[:, 6] == 1)]
    ignore = gt[np.isin(gt[:, 7], [2, 7, 8, 12])]
    for f in range(1, N_FRAMES + 1):
        rows = peds[peds[:, 0] == f]
        GT[f] = (rows[:, 1].astype(int), to_xyxy(rows))
        IGNORE[f] = to_xyxy(ignore[ignore[:, 0] == f])
    n_gt_ids = len(np.unique(peds[:, 1]))
    print(f"GT: {len(peds)} pedestrian boxes, {n_gt_ids} unique identities, "
          f"median visibility {np.median(peds[:, 8]):.2f}")

    # Validate labels visually before trusting them
    frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(frame)
    ids, boxes = GT[1]
    for tid, b in zip(ids, boxes):
        ax.add_patch(plt.Rectangle(b[:2], b[2] - b[0], b[3] - b[1], fill=False, ec="lime", lw=1.5))
        ax.text(b[0], b[1] - 3, str(tid), color="lime", fontsize=8)
    for b in IGNORE[1]:
        ax.add_patch(plt.Rectangle(b[:2], b[2] - b[0], b[3] - b[1], fill=False, ec="orange", lw=1, ls="--"))
    ax.set_title("Frame 1 ground truth (green = pedestrians with IDs, orange dashed = ignore regions)")
    ax.axis("off")
    save_fig(fig, "day4_gt_frame1.png")

# %% [markdown]
# ## 3 · Shared helpers: IoU and drawing

# %%
def pairwise_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


def id_color(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(int(track_id) * 9973)
    return tuple(int(c) for c in rng.integers(60, 255, size=3))


# %% [markdown]
# ## 4 · Detection only: why we need a tracker
#
# Run the detector frame by frame and cache its boxes. A detector has no memory: there is no notion of "the same person as last frame", and the count flickers whenever someone is occluded.

# %%
detector = YOLO(MODEL_WEIGHTS)
DETECTIONS = {}
t0 = time.perf_counter()
for f, frame in iterate_frames():
    r = detector.predict(frame, conf=DET_CONF, classes=[0], imgsz=IMGSZ, device=DEVICE, verbose=False)[0]
    DETECTIONS[f] = (r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())
det_fps = N_FRAMES / (time.perf_counter() - t0)
print(f"Detector: {det_fps:.1f} FPS at imgsz={IMGSZ}")

det_counts = [int((DETECTIONS[f][1] >= NAIVE_MIN_CONF).sum()) for f in range(1, N_FRAMES + 1)]
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(det_counts, label=f"detections with conf ≥ {NAIVE_MIN_CONF}")
if GT:
    ax.plot([len(GT[f][0]) for f in range(1, N_FRAMES + 1)], label="GT pedestrians (incl. occluded)")
ax.set_xlabel("frame")
ax.set_ylabel("people")
ax.legend()
ax.grid(alpha=0.3)
ax.set_title("Per-frame counts: detections flicker; ground truth also includes heavily occluded people")
save_fig(fig, "day4_detection_counts.png")

# %% [markdown]
# ## 5 · Tracker 1: a naive IoU tracker (built from scratch)
#
# Each frame: match existing tracks to high-confidence detections with the Hungarian algorithm on IoU; unmatched detections start new IDs; tracks unseen for `max_age` frames die. No motion model, no low-score rescue.

# %%
class NaiveIoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 5, min_conf: float = NAIVE_MIN_CONF):
        self.iou_threshold, self.max_age, self.min_conf = iou_threshold, max_age, min_conf
        self.tracks: dict[int, dict] = {}
        self.next_id = 1

    def update(self, boxes: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        boxes = boxes[scores >= self.min_conf]
        track_ids = list(self.tracks)
        matched_tracks, matched_dets = set(), set()
        if track_ids and len(boxes):
            iou = pairwise_iou(np.array([self.tracks[t]["box"] for t in track_ids]), boxes)
            rows, cols = linear_sum_assignment(-iou)
            for r, c in zip(rows, cols):
                if iou[r, c] >= self.iou_threshold:
                    tid = track_ids[r]
                    self.tracks[tid].update(box=boxes[c], age=0)
                    matched_tracks.add(tid)
                    matched_dets.add(c)
        for tid in track_ids:
            if tid not in matched_tracks:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    del self.tracks[tid]
        out_ids, out_boxes = [], []
        for c in range(len(boxes)):
            if c not in matched_dets:
                self.tracks[self.next_id] = {"box": boxes[c], "age": 0}
                matched_tracks.add(self.next_id)
                self.next_id += 1
        for tid in matched_tracks:
            out_ids.append(tid)
            out_boxes.append(self.tracks[tid]["box"])
        return np.array(out_ids, dtype=int), np.array(out_boxes).reshape(-1, 4)


naive = NaiveIoUTracker()
TRACKS = {"naive_iou": {f: naive.update(*DETECTIONS[f]) for f in range(1, N_FRAMES + 1)}}
print("naive tracker: unique IDs =", len({i for ids, _ in TRACKS["naive_iou"].values() for i in ids}))

# %% [markdown]
# ## 6 · Trackers 2 and 3: ByteTrack and BoT-SORT (Ultralytics)
#
# `model.track(..., persist=True)` keeps tracker state between calls, so we feed frames one by one exactly like a live stream. Each run uses a fresh model so state never leaks between trackers. Tracker settings come from Ultralytics' default YAMLs; the cell writes editable copies so you can tune `track_buffer`, `match_thresh`, and friends.

# %%
def tracker_config(name: str) -> Path:
    cfg = yaml.safe_load((ULTRALYTICS_ROOT / "cfg" / "trackers" / f"{name}.yaml").read_text())
    cfg["track_buffer"] = TRACK_BUFFER
    path = OUTPUT_DIR / f"{name}_custom.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"{name}: " + ", ".join(f"{k}={v}" for k, v in cfg.items()))
    return path


def run_ultralytics_tracker(name: str) -> tuple[dict, float]:
    model = YOLO(MODEL_WEIGHTS)
    cfg = tracker_config(name)
    out = {}
    t0 = time.perf_counter()
    for f, frame in iterate_frames():
        r = model.track(frame, persist=True, tracker=str(cfg), conf=DET_CONF, classes=[0], imgsz=IMGSZ, device=DEVICE, verbose=False)[0]
        if r.boxes.id is None:
            out[f] = (np.zeros(0, dtype=int), np.zeros((0, 4)))
        else:
            out[f] = (r.boxes.id.int().cpu().numpy(), r.boxes.xyxy.cpu().numpy())
    return out, N_FRAMES / (time.perf_counter() - t0)


TRACK_FPS = {"naive_iou": det_fps}
for name in ["bytetrack", "botsort"]:
    TRACKS[name], TRACK_FPS[name] = run_ultralytics_tracker(name)
    print(f"{name}: {TRACK_FPS[name]:.1f} FPS, unique IDs = {len({i for ids, _ in TRACKS[name].values() for i in ids})}")

# %% [markdown]
# ## 7 · Measure ID stability: MOTA, IDF1, ID switches
#
# Implemented from the definitions so every number is explainable:
#
# - **CLEAR-MOT matching** per frame: keep last frame's GT↔track pairs if IoU ≥ 0.5, then Hungarian-match the rest.
# - **ID switch (IDSW)**: a GT person is matched to a different track ID than the last time they were matched.
# - **MOTA** = 1 − (FN + FP + IDSW) / #GT boxes — dominated by detection quality.
# - **IDF1** = 2·IDTP / (#GT boxes + #predicted boxes) after a *global* one-to-one assignment of GT identities to track identities — the metric that best reflects "same person keeps the same ID".
# - **Fragmentations**: how many times a GT trajectory is interrupted (matched → unmatched → matched).
#
# Predictions that only overlap ignore regions are removed before scoring, following the MOT17 protocol.

# %%
def drop_ignored(frame: int, ids: np.ndarray, boxes: np.ndarray):
    ign = IGNORE.get(frame, np.zeros((0, 4)))
    if len(ign) == 0 or len(boxes) == 0:
        return ids, boxes
    _, gt_boxes = GT[frame]
    hits_ignore = pairwise_iou(boxes, ign).max(1) >= MATCH_IOU
    hits_ped = pairwise_iou(boxes, gt_boxes).max(1) >= MATCH_IOU if len(gt_boxes) else np.zeros(len(boxes), bool)
    keep = ~(hits_ignore & ~hits_ped)
    return ids[keep], boxes[keep]


def evaluate_tracks(tracks: dict) -> dict:
    fp = fn = idsw = frag = n_gt = n_pred = 0
    current: dict[int, int] = {}  # gt_id -> track_id matched in the previous frame
    last_matched: dict[int, int] = {}  # gt_id -> last track_id ever matched
    was_tracked: dict[int, bool] = {}
    co_occurrence = defaultdict(int)  # (gt_id, track_id) -> frames matched at IoU >= threshold
    gt_counts, pred_counts = defaultdict(int), defaultdict(int)

    for f in range(1, N_FRAMES + 1):
        g_ids, g_boxes = GT[f]
        t_ids, t_boxes = drop_ignored(f, *tracks[f])
        n_gt += len(g_ids)
        n_pred += len(t_ids)
        for g in g_ids:
            gt_counts[g] += 1
        for t in t_ids:
            pred_counts[t] += 1
        iou = pairwise_iou(g_boxes, t_boxes)
        for gi, ti in zip(*np.where(iou >= MATCH_IOU)):
            co_occurrence[(g_ids[gi], t_ids[ti])] += 1

        matches = {}
        t_index = {t: j for j, t in enumerate(t_ids)}
        for gi, g in enumerate(g_ids):  # 1) keep previous correspondences
            t = current.get(g)
            if t in t_index and t not in matches.values() and iou[gi, t_index[t]] >= MATCH_IOU:
                matches[g] = t
        free_g = [gi for gi, g in enumerate(g_ids) if g not in matches]
        used_t = set(matches.values())
        free_t = [j for j, t in enumerate(t_ids) if t not in used_t]
        if free_g and free_t:  # 2) Hungarian on the rest
            sub = iou[np.ix_(free_g, free_t)]
            rows, cols = linear_sum_assignment(-sub)
            for r, c in zip(rows, cols):
                if sub[r, c] >= MATCH_IOU:
                    g, t = g_ids[free_g[r]], t_ids[free_t[c]]
                    if g in last_matched and last_matched[g] != t:
                        idsw += 1
                    matches[g] = t
        for g in g_ids:
            tracked = g in matches
            if tracked and g in was_tracked and not was_tracked[g]:
                frag += 1
            if g in was_tracked or tracked:
                was_tracked[g] = tracked
        for g, t in matches.items():
            last_matched[g] = t
        current = dict(matches)
        tp = len(matches)
        fn += len(g_ids) - tp
        fp += len(t_ids) - tp

    # IDF1: global one-to-one assignment of GT identities to predicted identities
    gt_list, tr_list = sorted(gt_counts), sorted(pred_counts)
    idtp = 0
    if gt_list and tr_list:
        gi = {g: i for i, g in enumerate(gt_list)}
        ti = {t: j for j, t in enumerate(tr_list)}
        mat = np.zeros((len(gt_list), len(tr_list)))
        for (g, t), c in co_occurrence.items():
            mat[gi[g], ti[t]] = c
        rows, cols = linear_sum_assignment(-mat)
        idtp = int(mat[rows, cols].sum())
    return {
        "MOTA": 1 - (fn + fp + idsw) / max(n_gt, 1),
        "IDF1": 2 * idtp / max(n_gt + n_pred, 1),
        "IDP": idtp / max(n_pred, 1),
        "IDR": idtp / max(n_gt, 1),
        "IDSW": idsw,
        "Frag": frag,
        "FP": fp,
        "FN": fn,
        "Recall": (n_gt - fn) / max(n_gt, 1),
        "Precision": (n_pred - fp) / max(n_pred, 1),
        "unique_IDs": len(tr_list),
        "GT_IDs": len(gt_list),
    }


if GT:
    rows = []
    for name, tracks in TRACKS.items():
        rows.append({"tracker": name, **evaluate_tracks(tracks), "FPS": TRACK_FPS[name]})
    metrics_df = pd.DataFrame(rows).set_index("tracker")
    print(metrics_df.round(3).to_string())
    metrics_df.to_csv(OUTPUT_DIR / "day4_tracking_metrics.csv")

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    for ax, col in zip(axes, ["IDF1", "MOTA", "IDSW"]):
        ax.bar(metrics_df.index, metrics_df[col], color=["gray", "tab:blue", "tab:orange"])
        ax.set_title(col + (" (lower is better)" if col == "IDSW" else " (higher is better)"))
        for i, v in enumerate(metrics_df[col]):
            ax.text(i, v, f"{v:.3f}" if col != "IDSW" else str(int(v)), ha="center", va="bottom")
    save_fig(fig, "day4_tracker_comparison.png")
else:
    metrics_df = None
    print("Custom video: no ground truth, so only qualitative output and ID statistics below.")

# %% [markdown]
# ## 8 · Visualize ID stability: track lifetimes
#
# Each horizontal line is one track ID, drawn over the frames where it exists. Stable tracking = few, long lines. Fragmented tracking = many short lines (the same person keeps getting new IDs).

# %%
fig, axes = plt.subplots(1, len(TRACKS), figsize=(6 * len(TRACKS), 6), sharey=False)
for ax, (name, tracks) in zip(np.atleast_1d(axes), TRACKS.items()):
    frames_by_id = defaultdict(list)
    for f, (ids, _) in tracks.items():
        for i in ids:
            frames_by_id[int(i)].append(f)
    lengths = [len(v) for v in frames_by_id.values()]
    for row, (tid, fr) in enumerate(sorted(frames_by_id.items(), key=lambda kv: kv[1][0])):
        ax.hlines(row, min(fr), max(fr), color=np.array(id_color(tid)) / 255, lw=2)
    ax.set_title(f"{name}: {len(frames_by_id)} IDs, median life {np.median(lengths):.0f} frames")
    ax.set_xlabel("frame")
    ax.set_ylabel("track (sorted by first appearance)")
save_fig(fig, "day4_track_lifetimes.png")

# %% [markdown]
# ## 9 · Render the tracked video
#
# Boxes are coloured by ID and each track draws a short trail of its recent centres. The best tracker by IDF1 is rendered (ByteTrack for custom videos). The file is re-encoded to H.264 so it plays in the browser.

# %%
best_tracker = metrics_df["IDF1"].idxmax() if metrics_df is not None else "bytetrack"
tracks = TRACKS[best_tracker]
scale = OUTPUT_WIDTH / W
out_size = (OUTPUT_WIDTH, int(round(H * scale / 2) * 2))
raw_video = OUTPUT_DIR / f"{SEQ_NAME}_{best_tracker}_raw.mp4"
final_video = OUTPUT_DIR / f"{SEQ_NAME}_{best_tracker}_tracked.mp4"
writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, out_size)
trails = defaultdict(lambda: deque(maxlen=30))

for f, frame in iterate_frames():
    frame = cv2.resize(frame, out_size)
    ids, boxes = tracks[f]
    for tid, box in zip(ids, boxes * scale):
        x1, y1, x2, y2 = box.astype(int)
        color = id_color(tid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {tid}", (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        trails[tid].append(((x1 + x2) // 2, y2))
        pts = np.array(trails[tid], dtype=np.int32)
        if len(pts) > 1:
            cv2.polylines(frame, [pts], False, color, 2)
    cv2.putText(frame, f"{best_tracker} | frame {f}/{N_FRAMES} | tracks {len(ids)}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    writer.write(frame)
writer.release()

if shutil.which("ffmpeg"):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_video), "-vcodec", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "26", str(final_video)], check=True)
    raw_video.unlink()
else:
    raw_video.rename(final_video)
print(f"Tracked video -> {final_video} ({final_video.stat().st_size / 1e6:.1f} MB)")

if IN_NOTEBOOK:
    from IPython.display import Video, display

    display(Video(str(final_video), embed=True, width=960))

# %% [markdown]
# ### Key frames from the tracked video
#
# Pick a few frames one second apart and check that the same people keep the same IDs.

# %%
picks = [int(x) for x in np.linspace(1, N_FRAMES, 4)]
cap_frames = {f: fr for f, fr in iterate_frames() if f in picks}
fig, axes = plt.subplots(2, 2, figsize=(20, 12))
for ax, f in zip(axes.flat, picks):
    frame = cv2.cvtColor(cap_frames[f], cv2.COLOR_BGR2RGB)
    ax.imshow(frame)
    for tid, b in zip(*tracks[f]):
        ax.add_patch(plt.Rectangle(b[:2], b[2] - b[0], b[3] - b[1], fill=False, ec=np.array(id_color(tid)) / 255, lw=2))
        ax.text(b[0], b[1] - 4, str(tid), color=np.array(id_color(tid)) / 255, fontsize=9, weight="bold")
    ax.set_title(f"frame {f}")
    ax.axis("off")
save_fig(fig, "day4_keyframes.png")

# %% [markdown]
# ## 10 · Save results

# %%
summary = {
    "dataset": KAGGLE_HANDLE if VIDEO_PATH is None else VIDEO_PATH,
    "sequence": SEQ_NAME,
    "frames": N_FRAMES,
    "fps": FPS,
    "detector": MODEL_WEIGHTS,
    "imgsz": IMGSZ,
    "rendered_tracker": best_tracker,
    "video": str(final_video),
    "metrics": metrics_df.round(4).reset_index().to_dict(orient="records") if metrics_df is not None else None,
}
with open(OUTPUT_DIR / "day4_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=float)
print(json.dumps(summary, indent=2, default=float))

if IN_COLAB:
    print("Download the video from the Files panel, or run: from google.colab import files; files.download(str(final_video))")

# %% [markdown]
# ## Takeaways (fill in with your numbers)
#
# - IDF1: naive ___ → ByteTrack ___ → BoT-SORT ___. IDSW: ___ → ___ → ___.
# - MOTA is capped by the detector (FN from occluded people dominate); IDF1 and IDSW isolate the tracker's job.
# - ByteTrack's low-score second matching keeps tracks alive through partial occlusion; the naive tracker throws those boxes away and spawns new IDs.
# - Things to try: raise `TRACK_BUFFER` for long occlusions, enable ReID in `botsort_custom.yaml` (`with_reid: True`) for appearance matching, or track your Day 2 hard-hat model on a site video.
#
# Next: Day 5 switches from "find objects" to "find defects" with anomaly detection.
