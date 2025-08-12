
import os
import cv2
import numpy as np
import tensorflow as tf
from model_varlen import create_cnn_rnn_model_varlen

def read_video_to_array(video_path, img_size=224, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = 1
    if max_frames and total > max_frames:
        step = max(1, total // max_frames)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (img_size, img_size))
            frame = frame.astype("float32") / 255.0
            frames.append(frame)
        idx += 1
    cap.release()
    if not frames:
        return np.zeros((1, img_size, img_size, 3), dtype="float32")
    return np.stack(frames, axis=0)

def predict_video(model_path, video_path, img_size=224, max_frames=None, threshold=0.5):
    print("[predict_varlen] Loading model:", model_path)
    try:
        model = tf.keras.models.load_model(model_path, compile=False,
                                           custom_objects={"FramePresence": tf.keras.layers.Layer})
    except Exception:
        # If custom layer not registered, import and retry
        from model_varlen import FramePresence
        model = tf.keras.models.load_model(model_path, compile=False,
                                           custom_objects={"FramePresence": FramePresence})

    arr = read_video_to_array(video_path, img_size=img_size, max_frames=max_frames)
    x = np.expand_dims(arr, 0)  # (1, T, H, W, C)
    prob = float(model.predict(x, verbose=0)[0][0])
    pred = 1 if prob >= threshold else 0
    print(f"[predict_varlen] prob={prob:.4f} pred={pred}")
    return prob, pred

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--max-frames", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    predict_video(args.model, args.video, img_size=args.img_size,
                  max_frames=args.max_frames, threshold=args.threshold)
