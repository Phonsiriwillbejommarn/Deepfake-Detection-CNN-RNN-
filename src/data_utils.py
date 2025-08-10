import cv2
import os
import numpy as np

# ไม่ต้องใช้ MTCNN หรือ Haar Cascade อีกต่อไป

def extract_frames_from_video(video_path, output_folder, frames_to_extract=100, image_size=(224, 224)):
    """
    ฟังก์ชันสำหรับแยกเฟรมจากวิดีโอ, ปรับขนาด, และบันทึกเป็นไฟล์ภาพ (เวอร์ชันเต็มเฟรม)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        print(f"Warning: Video file {video_path} has zero frames.")
        return
        
    # เลือกเฟรมแบบกระจายตัวทั่วทั้งวิดีโอ
    indices = np.linspace(0, frame_count - 1, frames_to_extract, dtype=int)
    
    saved_count = 0
    for i, frame_index in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # ปรับขนาดเฟรมให้เป็นขนาดมาตรฐาน
        resized_frame = cv2.resize(frame, image_size)
        
        output_path = os.path.join(output_folder, f"frame_{saved_count}.jpg")
        cv2.imwrite(output_path, resized_frame)
        saved_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} full frames from {os.path.basename(video_path)}")

def process_all_videos(raw_data_dir, processed_data_dir, image_size=(224, 224)):
    """
    ฟังก์ชันหลักสำหรับวนลูปประมวลผลวิดีโอทั้งหมด
    """
    for class_name in ['real', 'fake']:
        source_dir = os.path.join(raw_data_dir, class_name)
        output_class_dir = os.path.join(processed_data_dir, class_name)
        
        if not os.path.exists(source_dir):
            print(f"Directory not found: {source_dir}")
            continue

        for video_filename in os.listdir(source_dir):
            if video_filename.lower().endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(source_dir, video_filename)
                
                video_name_without_ext = os.path.splitext(video_filename)[0]
                output_video_dir = os.path.join(output_class_dir, video_name_without_ext)
                
                print(f"\nProcessing {video_path}...")
                extract_frames_from_video(video_path, output_video_dir, image_size=image_size)

# --- ส่วนของการรันโปรแกรม ---
if __name__ == '__main__':
    RAW_DATA_PATH = 'data'
    PROCESSED_DATA_PATH = 'data/processed_full_frame' # <-- สร้างโฟลเดอร์ใหม่เพื่อไม่ให้ปนกับข้อมูลเก่า
    IMG_SIZE = 224 # <-- ขนาดมาตรฐานสำหรับโมเดลส่วนใหญ่

    os.makedirs(os.path.join(PROCESSED_DATA_PATH, 'real'), exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_DATA_PATH, 'fake'), exist_ok=True)

    process_all_videos(RAW_DATA_PATH, PROCESSED_DATA_PATH, image_size=(IMG_SIZE, IMG_SIZE))
    
    print("\n--- All videos have been processed into full frames. ---")