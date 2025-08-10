import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array, random_rotation, random_zoom

# --- ปรับปรุงฟังก์ชัน Augmentation (นำ @tf.function ออก) ---
def augment_image(image_array):
    """
    ฟังก์ชันสำหรับทำ Data Augmentation บน NumPy array
    """
    # ฟังก์ชันของ tf.image สามารถทำงานกับ NumPy array และ trả về EagerTensor ได้
    # เราจึงต้องแปลงกลับเป็น .numpy()
    image_array = tf.image.random_flip_left_right(image_array).numpy()
    image_array = tf.image.random_brightness(image_array, max_delta=0.1).numpy()
    image_array = tf.image.random_contrast(image_array, lower=0.9, upper=1.1).numpy()

    # ฟังก์ชันรุ่นเก่าเหล่านี้ทำงานกับ NumPy array โดยตรง
    image_array = random_rotation(image_array, 10, row_axis=0, col_axis=1, channel_axis=2)
    image_array = random_zoom(image_array, (0.9, 1.1), row_axis=0, col_axis=1, channel_axis=2)

    # ใช้ np.clip เพื่อให้ทำงานกับ array ต่อเนื่องได้เลย
    image_array = np.clip(image_array, 0.0, 1.0)
    return image_array

class VideoSequenceGenerator(tf.keras.utils.Sequence):
    def __init__(self, video_folders, labels, batch_size, frame_count, image_size, shuffle=True, augment=False):
        self.video_folders = video_folders
        self.labels = labels
        self.batch_size = batch_size
        self.frame_count = frame_count
        self.image_size = image_size
        self.shuffle = shuffle
        self.augment = augment
        
        self.indices = np.arange(len(self.video_folders))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.video_folders) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        X = np.empty((self.batch_size, self.frame_count, self.image_size, self.image_size, 3))
        y = np.empty((self.batch_size), dtype=int)
        
        for i, video_index in enumerate(batch_indices):
            video_folder_path = self.video_folders[video_index]
            frames = self.load_video_frames(video_folder_path)
            X[i,] = frames
            y[i] = self.labels[video_index]
            
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def load_video_frames(self, video_folder_path):
        frames = []
        frame_files = sorted(os.listdir(video_folder_path))
        
        if len(frame_files) >= self.frame_count:
            selected_files = np.random.choice(frame_files, self.frame_count, replace=False)
        else:
            selected_files = np.random.choice(frame_files, self.frame_count, replace=True)

        for file_name in sorted(selected_files):
            img_path = os.path.join(video_folder_path, file_name)
            img = load_img(img_path, target_size=(self.image_size, self.image_size))
            img_array = img_to_array(img) / 255.0
            
            # --- แก้ไขการเรียกใช้ฟังก์ชัน Augmentation ---
            if self.augment:
                # ส่ง NumPy array เข้าไปตรงๆ ได้เลย
                img_array = augment_image(img_array)
            
            frames.append(img_array)
        
        return np.array(frames)