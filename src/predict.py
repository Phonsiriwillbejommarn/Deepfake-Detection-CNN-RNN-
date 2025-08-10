import os
import sys
import shutil
import argparse
import importlib.util
import builtins
import numpy as np
import tensorflow as tf

from data_utils import extract_frames_from_video

IMG_SIZE = 224
SEQUENCE_LENGTH = 100
# --- แก้ไขบรรทัดนี้ ---
MODEL_PATH = '/Users/phonsirithabunsri/Desktop/AI/models/deepfake_detector_best_v4.h5'
TEMP_FOLDER = '/Users/phonsirithabunsri/Desktop/AI/temp_prediction_frames'

def _patch_lambda_after_load(model):
    """
    แพตช์ให้ Lambda layer มองเห็น tf และชี้ฟังก์ชันไปที่ tf.reduce_sum(axis=1)
    ใช้กับโมเดล .h5 ที่เซฟ Lambda(lambda x: tf.reduce_sum(...)) ไว้ ซึ่งอาจไม่มี tf ในสโคปตอนโหลด
    """
    # ให้ 'tf' อยู่ใน builtins เผื่อฟังก์ชันที่ deserialize มาอ้างชื่อ 'tf'
    builtins.tf = tf

    # หาด้วยชื่อก่อน
    lambda_layer = None
    try:
        lambda_layer = model.get_layer('temporal_sum')
    except Exception:
        pass

    # ถ้าไม่เจอชื่อ ให้หาเลเยอร์แรกที่เป็น Lambda
    if lambda_layer is None:
        for lyr in model.layers:
            if isinstance(lyr, tf.keras.layers.Lambda):
                lambda_layer = lyr
                break

    if lambda_layer is not None:
        try:
            lambda_layer.function = (lambda x: tf.reduce_sum(x, axis=1))
            # บางเวอร์ชันของ Keras ต้องให้ init ชื่อใหม่เพื่อเคลียร์แคชภายใน
            try:
                lambda_layer._init_set_name(lambda_layer.name)
            except Exception:
                pass
            print(f"[PATCH] Patched Lambda layer '{lambda_layer.name}' to use tf.reduce_sum(axis=1).")
        except Exception as e:
            print(f"[PATCH] Failed to patch Lambda function: {e}")
    else:
        print("[PATCH] No Lambda layer found to patch.")

def predict_single_video(video_path, model):
    print(f"--- Starting prediction for: {video_path} ---")

    os.makedirs(TEMP_FOLDER, exist_ok=True)
    extract_frames_from_video(
        video_path,
        TEMP_FOLDER,
        frames_to_extract=SEQUENCE_LENGTH,
        image_size=(IMG_SIZE, IMG_SIZE)
    )

    # ใช้เฉพาะไฟล์รูป
    image_files = sorted(
        [f for f in os.listdir(TEMP_FOLDER)
         if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    )
    if len(image_files) == 0:
        print("Error: No frames were extracted. Please check the video or extractor.")
        shutil.rmtree(TEMP_FOLDER, ignore_errors=True)
        return None, None

    # ให้ครบ 100 เฟรมเสมอ
    if len(image_files) < SEQUENCE_LENGTH:
        print(f"Warning: Not enough frames. Found {len(image_files)}. Sampling with replacement.")
        rng = np.random.default_rng(42)
        selected_files = list(rng.choice(image_files, SEQUENCE_LENGTH, replace=True))
    else:
        selected_files = image_files[:SEQUENCE_LENGTH]

    # โหลดภาพ -> เทนเซอร์
    frames = []
    for file_name in selected_files:
        img_path = os.path.join(TEMP_FOLDER, file_name)
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        frames.append(img_array)

    input_data = np.expand_dims(np.array(frames, dtype=np.float32), axis=0)  # (1, T, H, W, C)
    print(f"Input shape -> {input_data.shape} | Model expects -> {getattr(model, 'input_shape', 'unknown')}")

    # พยากรณ์
    prediction_score = model.predict(input_data, verbose=0)[0][0]
    result = "REAL" if prediction_score > 0.5 else "FAKE"

    print(f"\n--- Prediction Result ---")
    print(f"Model Score: {prediction_score:.4f}")
    print(f"Conclusion: The video is likely {result}")

    shutil.rmtree(TEMP_FOLDER, ignore_errors=True)
    return result, prediction_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict if a video is a deepfake.')
    parser.add_argument('video_path', type=str, help='Path to the video file you want to predict.')
    args = parser.parse_args()

    print("\nDEBUG: Running the latest version of predict.py\n")

    # โหลด src/model.py เพื่อให้แน่ใจว่า symbol ต่าง ๆ ในไฟล์นั้นอยู่ใน sys.modules
    src_dir = os.path.dirname(os.path.abspath(__file__))      # .../AI/src
    model_py_path = os.path.join(src_dir, 'model.py')         # .../AI/src/model.py
    if os.path.exists(model_py_path):
        spec = importlib.util.spec_from_file_location("model", model_py_path)
        model_module = importlib.util.module_from_spec(spec)
        sys.modules["model"] = model_module
        spec.loader.exec_module(model_module)
        # ใส่ tf เข้า scope ของโมดูลนั้นด้วย (กันเหนียว)
        if not hasattr(model_module, 'tf'):
            model_module.tf = tf
    else:
        print(f"WARNING: model.py not found at {model_py_path}.")

    print("Loading trained model...")
    try:
        # safe_mode=False จำเป็นสำหรับ .h5 ที่มี Lambda/custom (Keras 3)
        model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
    except TypeError:
        # เผื่อ environment ที่ยังไม่มีพารามิเตอร์ safe_mode
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    print("Model loaded successfully.")

    # แพตช์ Lambda หลังโหลด (กัน NameError 'tf')
    _patch_lambda_after_load(model)

    # รันพยากรณ์
    if os.path.exists(args.video_path):
        predict_single_video(args.video_path, model)
    else:
        print(f"Error: Video file not found at '{args.video_path}'")