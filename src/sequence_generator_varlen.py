import os
import numpy as np
from typing import List, Optional
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class VideoSequenceGeneratorVarLen(Sequence):
    """
    Variable-length sequence generator.
    - แต่ละรายการคือโฟลเดอร์ที่มีเฟรมภาพ
    - Padding ด้วยศูนย์เป็นความยาวคงที่ pad_T ทั้ง training
    - โมเดลจะสร้าง mask จากเฟรมศูนย์เอง
    """
    def __init__(
        self,
        folders: List[str],
        labels: List[int],
        image_size: int = 224,
        batch_size: int = 4,
        shuffle: bool = True,
        max_frames: Optional[int] = None,   # จำกัดเฟรมต่อวิดีโอ (แนะนำตั้ง เช่น 256)
        seed: int = 1337,
        augment: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert len(folders) == len(labels)
        self.folders = list(folders)
        self.labels = np.asarray(labels, dtype="float32").reshape(-1, 1)
        self.image_size = int(image_size)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.max_frames = int(max_frames) if max_frames else None
        self.rng = np.random.RandomState(seed)
        self.augment = augment

        # pad_T จะคงที่ตลอดการเทรน:
        # - ถ้ามี max_frames -> ใช้เป็น pad_T เลย
        # - ถ้าไม่มี -> จะเซ็ตจาก batch แรกครั้งเดียว
        self._pad_T = self.max_frames if self.max_frames is not None else None

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.folders) / self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.folders))
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def _list_frames(self, folder: str) -> List[str]:
        exts = (".jpg", ".jpeg", ".png")
        files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
        files.sort()  # assume alphabetical naming
        # ถ้ามี max_frames ให้ทำ uniform sampling ให้ไม่เกิน
        if self.max_frames and len(files) > self.max_frames:
            idx = np.linspace(0, len(files)-1, self.max_frames).round().astype(int)
            files = [files[i] for i in idx]
        return [os.path.join(folder, f) for f in files]

    def _load_frames(self, file_list: List[str]) -> np.ndarray:
        T = len(file_list)
        frames = np.zeros((T, self.image_size, self.image_size, 3), dtype="float32")
        for i, fp in enumerate(file_list):
            img = load_img(fp, target_size=(self.image_size, self.image_size))
            arr = img_to_array(img).astype("float32") / 255.0
            # TODO: augmentation ถ้าต้องการ
            frames[i] = arr
        return frames

    def __getitem__(self, idx):
        batch_idx = self.indices[idx*self.batch_size : (idx+1)*self.batch_size]
        batch_folders = [self.folders[i] for i in batch_idx]
        batch_labels = self.labels[batch_idx]

        # โหลดเฟรมของแต่ละวิดีโอใน batch
        seqs, lengths = [], []
        for folder in batch_folders:
            files = self._list_frames(folder)
            arr = self._load_frames(files)
            seqs.append(arr)
            lengths.append(arr.shape[0])

        # ---- กำหนด pad_T แบบคงที่ ----
        if self._pad_T is None:
            # ล็อกจาก batch แรก (ถ้าไม่ได้ตั้ง max_frames)
            self._pad_T = max(lengths) if lengths else 1
        pad_T = int(self._pad_T)

        # Padding ทุก sequence ให้ยาวเท่ากับ pad_T
        B = len(seqs)
        X = np.zeros((B, pad_T, self.image_size, self.image_size, 3), dtype="float32")
        for b, arr in enumerate(seqs):
            keep = min(arr.shape[0], pad_T)
            X[b, :keep] = arr[:keep]

        y = batch_labels
        return X, y
