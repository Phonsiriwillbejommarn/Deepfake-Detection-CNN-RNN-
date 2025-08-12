#!/bin/bash

# Define the base directory of the project
# This assumes the script is run from the project's root directory.
PROJECT_DIR="$(pwd)"
SRC_DIR="$PROJECT_DIR/src"
DATA_DIR="$PROJECT_DIR/data"
CHECKPOINTS_DIR="$PROJECT_DIR/checkpoints"

# --- Step 1: Data Preparation ---
echo "Starting data preparation..."

# Run the data extraction script
python - << 'PY'
import os, cv2, random
from pathlib import Path

# Get the base directory from the environment variable (or set a default)
PROJECT_DIR = os.getenv("PROJECT_DIR", str(Path.cwd()))
DATA_DIR = Path(PROJECT_DIR) / "data"

REAL_DIR = DATA_DIR / "real"
FAKE_DIR = DATA_DIR / "fake"
OUT_FRAMES = DATA_DIR / "frames"
LIST_TRAIN = DATA_DIR / "train_list.txt"
LIST_VAL   = DATA_DIR / "val_list.txt"

IMG_SIZE   = 224
FPS_SAMPLE = 5
SPLIT      = 0.8
VIDEO_EXT  = {".mp4",".mov",".mkv",".avi",".webm",".m4v"}

def iter_videos(root, label, cname):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXT:
            yield p, label, cname

def extract_frames(video_path: Path, out_dir: Path, img_size=224, fps_sample=5):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] open fail: {video_path}")
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    step = max(1, round(fps / float(fps_sample))) if fps_sample and fps>0 else 6
    idx = saved = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % step == 0:
            frame = cv2.resize(frame, (img_size, img_size))
            cv2.imwrite(str(out_dir / f"{saved:06d}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            saved += 1
        idx += 1
    cap.release()
    return saved

items = []
for root,label,cname in [(REAL_DIR,0,"real"), (FAKE_DIR,1,"fake")]:
    for vid,label,cname in iter_videos(root,label,cname):
        out_dir = OUT_FRAMES / cname / vid.stem
        n = extract_frames(vid, out_dir, IMG_SIZE, FPS_SAMPLE)
        if n>0:
            items.append((str(out_dir), label))

random.Random(1337).shuffle(items)
n_train = int(len(items)*SPLIT)
train, val = items[:n_train], items[n_train:]

LIST_TRAIN.parent.mkdir(parents=True, exist_ok=True)
with open(LIST_TRAIN,"w",encoding="utf-8") as f:
    for p,lab in train: f.write(f"{p}\t{lab}\n")
with open(LIST_VAL,"w",encoding="utf-8") as f:
    for p,lab in val: f.write(f"{p}\t{lab}\n")

print(f"✅ Done. train={len(train)}  val={len(val)}")
print(f"train_list: {LIST_TRAIN}")
print(f"val_list  : {LIST_VAL}")
PY

# Check if the data list files were created
if [ ! -f "$DATA_DIR/train_list.txt" ] || [ ! -f "$DATA_DIR/val_list.txt" ]; then
    echo "Error: Data list files were not created. Exiting."
    exit 1
fi

# --- Step 2: Model Training ---
echo "Starting model training..."
cd "$SRC_DIR" || { echo "Failed to change directory. Exiting."; exit 1; }

# Set environment variables for the training script
export TRAIN_LIST="$DATA_DIR/train_list.txt"
export VAL_LIST="$DATA_DIR/val_list.txt"
export MODEL_OUT="$CHECKPOINTS_DIR/varlen_best.h5"

# Run the training script
python train_varlen.py

echo "Script finished."