
import os
import numpy as np
import tensorflow as tf

from model_varlen import create_cnn_rnn_model_varlen
from sequence_generator_varlen import VideoSequenceGeneratorVarLen

# ---- Config (env override-able) ----
IMG_SIZE   = int(os.getenv("IMG_SIZE", "224"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
EPOCHS     = int(os.getenv("EPOCHS", "20"))
MAX_FRAMES = os.getenv("MAX_FRAMES", "")  # cap frames per video (optional)
MAX_FRAMES = int(MAX_FRAMES) if str(MAX_FRAMES).strip().isdigit() else None

TRAIN_LIST = os.getenv("TRAIN_LIST", "train_list.txt")  # each line: <folder>\t<label>
VAL_LIST   = os.getenv("VAL_LIST", "val_list.txt")
MODEL_OUT  = os.getenv("MODEL_OUT", "checkpoints/varlen_best.h5")

# ---- Utils ----
def read_list(path):
    folders, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            p, y = ln.split("\t")
            folders.append(p)
            labels.append(int(y))
    return folders, labels

# ---- Datasets ----
train_folders, train_labels = read_list(TRAIN_LIST)
val_folders,   val_labels   = read_list(VAL_LIST)

train_gen = VideoSequenceGeneratorVarLen(train_folders, train_labels, image_size=IMG_SIZE,
                                         batch_size=BATCH_SIZE, shuffle=True, max_frames=MAX_FRAMES)
val_gen   = VideoSequenceGeneratorVarLen(val_folders, val_labels, image_size=IMG_SIZE,
                                         batch_size=BATCH_SIZE, shuffle=False, max_frames=MAX_FRAMES)

# ---- Model ----
model = create_cnn_rnn_model_varlen(input_shape=(None, IMG_SIZE, IMG_SIZE, 3))
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss="binary_crossentropy",
              metrics=[tf.keras.metrics.AUC(name="auc")])

os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
ckpt = tf.keras.callbacks.ModelCheckpoint(MODEL_OUT, monitor="val_auc", mode="max",
                                          save_best_only=True, verbose=1)
early = tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                         patience=8, restore_best_weights=True, verbose=1)
reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                              patience=4, verbose=1, min_lr=1e-6)

# ---- Train ----
history = model.fit(train_gen,
                    validation_data=val_gen,
                    epochs=EPOCHS,
                    callbacks=[ckpt, early, reduce])

print("[train_varlen] Done. Best model at:", MODEL_OUT)
