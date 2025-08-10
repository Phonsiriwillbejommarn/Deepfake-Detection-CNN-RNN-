import os
import tensorflow as tf
from model import create_cnn_rnn_model # ใช้ model.py ตัวเดิมที่ง่ายลง
from sequence_generator import VideoSequenceGenerator
import numpy as np
import random
from sklearn.utils.class_weight import compute_class_weight

# --- ตั้งค่าพื้นฐาน ---
IMG_SIZE = 224
SEQUENCE_LENGTH = 100
BATCH_SIZE = 4
EPOCHS = 50 
PROCESSED_DATA_DIR = '/Users/phonsirithabunsri/Desktop/AI/data/processed_full_frame'

# --- 1. การแบ่งข้อมูล 
print("Splitting data into training and validation sets...")
all_video_folders = []
all_labels = []
for class_name in ['real', 'fake']:
    class_dir = os.path.join(PROCESSED_DATA_DIR, class_name)
    label = 1 if class_name == 'real' else 0
    if not os.path.isdir(class_dir): continue
    for video_folder in os.listdir(class_dir):
        video_folder_path = os.path.join(class_dir, video_folder)
        if os.path.isdir(video_folder_path):
            all_video_folders.append(video_folder_path)
            all_labels.append(label)
if not all_video_folders: raise ValueError("No processed video data found.")
combined = list(zip(all_video_folders, all_labels))
random.shuffle(combined)
all_video_folders, all_labels = zip(*combined)
split_index = int(len(all_video_folders) * 0.8)
train_folders, train_labels = list(all_video_folders[:split_index]), list(all_labels[:split_index])
val_folders, val_labels = list(all_video_folders[split_index:]), list(all_labels[split_index:])

# --- 2. สร้าง Data Generator 
print("Initializing Data Generators...")
train_generator = VideoSequenceGenerator(
    video_folders=train_folders, labels=train_labels, batch_size=BATCH_SIZE,
    frame_count=SEQUENCE_LENGTH, image_size=IMG_SIZE, augment=True
)
validation_generator = VideoSequenceGenerator(
    video_folders=val_folders, labels=val_labels, batch_size=BATCH_SIZE,
    frame_count=SEQUENCE_LENGTH, image_size=IMG_SIZE, shuffle=False
)


# --- 4. สร้างและคอมไพล์โมเดล 
print("Creating model for stability training...")
model = create_cnn_rnn_model(input_shape=(SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3))

# --- กลยุทธ์ Learning Rate ที่ช้าและมั่นคง ---
total_steps = len(train_generator) * EPOCHS
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-5, # <--- ลดลงอย่างมาก เพื่อเริ่มอย่างช้าๆ
    decay_steps=total_steps,
    alpha=0.1
)

optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4)
loss_function = tf.keras.losses.BinaryCrossentropy() # <-- ไม่ใช้ label_smoothing ชั่วคราว

model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
print("Model created and compiled for STABLE training.")

# --- 5. เริ่มการฝึกโมเดล ---
print(f"\n--- Starting Model Training ---")
os.makedirs('models', exist_ok=True)

checkpoint_path = 'models/deepfake_detector_best_v6.h5' # <-- เปลี่ยนเป็น v6

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
)

# คำนวณ class_weight เพื่อจัดการกับความไม่สมดุล
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stopping]
)

print("\n--- Training Finished ---")