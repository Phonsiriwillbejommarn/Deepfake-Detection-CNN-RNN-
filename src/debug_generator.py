import os
import random
from sequence_generator import VideoSequenceGenerator

# --- ตั้งค่าให้ตรงกับ train.py ---
PROCESSED_DATA_DIR = '/Users/phonsirithabunsri/Desktop/AI/data/processed_full_frame'
IMG_SIZE = 224
SEQUENCE_LENGTH = 100
BATCH_SIZE = 10 # ดูทีละ 4 วิดีโอ

# --- โค้ดสำหรับดึงข้อมูล ---
all_video_folders = []
all_labels = []

for class_name in ['real', 'fake']:
    class_dir = os.path.join(PROCESSED_DATA_DIR, class_name)
    label = 1 if class_name == 'real' else 0
    for video_folder in os.listdir(class_dir):
        video_folder_path = os.path.join(class_dir, video_folder)
        if os.path.isdir(video_folder_path):
            all_video_folders.append(video_folder_path)
            all_labels.append(label)

# สุ่มข้อมูลเพื่อให้เห็นทั้ง real และ fake
combined = list(zip(all_video_folders, all_labels))
random.shuffle(combined)
all_video_folders, all_labels = zip(*combined)

# --- สร้าง Generator ---
debug_generator = VideoSequenceGenerator(
    video_folders=list(all_video_folders), 
    labels=list(all_labels), 
    batch_size=BATCH_SIZE,
    frame_count=SEQUENCE_LENGTH, 
    image_size=IMG_SIZE,
    shuffle=False # ปิดการสุ่มเพื่อให้ผลลัพธ์คงที่
)

print("--- Getting the first batch of data from generator ---")
X, y = debug_generator[0] # เอาข้อมูลชุดที่ 1

print(f"Shape of image data (X): {X.shape}") # ควรเป็น (4, 100, 224, 224, 3)
print(f"Shape of labels (y): {y.shape}")     # ควรเป็น (4,)
print(f"Labels for this batch (y): {y}")     # <-- ดูค่านี้! 1 = real, 0 = fake

# แสดง Path และ Label ของแต่ละวิดีโอใน Batch นี้
print("\n--- Videos in this batch ---")
for i in range(BATCH_SIZE):
    video_path = debug_generator.video_folders[i]
    label = debug_generator.labels[i]
    print(f"Path: {video_path}, Label: {label}")