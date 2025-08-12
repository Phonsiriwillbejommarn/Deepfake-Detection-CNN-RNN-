
# Deepfake-Detection-CNN-RNN
---

## สารบัญ

- [ภาพรวมโปรเจกต์](#ภาพรวมโปรเจกต์)
- [โครงสร้างโปรเจกต์](#โครงสร้างโปรเจกต์)
- [การติดตั้งและการเตรียมสภาพแวดล้อม](#การติดตั้งและการเตรียมสภาพแวดล้อม)
- [การเตรียมข้อมูล](#การเตรียมข้อมูล)
- [การฝึกโมเดล (Training)](#การฝึกโมเดล-training)
- [การทำนายผล (Inference)](#การทำนายผล-inference)



---

## ภาพรวมโปรเจกต์

โปรเจกต์นี้ใช้สถาปัตยกรรมแบบ CNN-RNN สำหรับตรวจจับ Deepfake จากวิดีโอ รองรับความยาววิดีโอที่ไม่เท่ากันด้วยกลไก padding/masking ภายใน data generator และโมเดลจะบันทึกเวอร์ชันที่ดีที่สุดเป็นไฟล์ `checkpoints/varlen_best.h5` หลังการฝึก

สำคัญ!! ผู้ใช้งานจะต้องมีชุดข้อมูลสำหรับการฝึกอยู่เเล้ว

---

## โครงสร้างโปรเจกต์
### 1. การจัดเตรียมโครงสร้างไฟล์

โครงสร้างควรมีลักษณะคล้ายกับนี้:

```text
Deepfake-Detection-CNN-RNN-/
├── checkpoints/
│   └── varlen_best.h5  (ไฟล์นี้จะถูกสร้างขึ้นหลังจากการฝึกโมเดล)
├── data/
│   ├── fake/  (โฟลเดอร์นี้สำหรับใส่วิดีโอ deepfake ของคุณ)
│   ├── real/  (โฟลเดอร์นี้สำหรับใส่วิดีโอจริงของคุณ)
├── README.md
├── run_all.sh
└── src/
    ├── model_varlen.py
    ├── predict_varlen.py
    ├── sequence_generator_varlen.py
    └── train_varlen.py

```

## การเตรียมข้อมูล

วางไฟล์วิดีโอของคุณลงใน `data/real` และ `data/fake` จากนั้นสคริปต์จะสกัดเฟรม แบ่ง train/val และสร้างไฟล์รายการ `train_list.txt` และ `val_list.txt` อัตโนมัติเมื่อเรียก `run_all.sh`

---

## การฝึกโมเดล (Training)

ใช้ `run_all.sh` เพื่อสั่ง *ทั้งขั้นตอนเตรียมข้อมูลและฝึกโมเดล* ในครั้งเดียว เมื่อฝึกเสร็จจะได้ไฟล์โมเดลที่ดีที่สุดอยู่ที่ `checkpoints/varlen_best.h5`

**คัดลอกคำสั่ง:** รันสคริปต์หลัก

```bash
sh run_all.sh
```

หากต้องการรันไฟล์ฝึกโดยตรง (ข้าม run\_all.sh) ให้ใช้ `src/train_varlen.py` และกำหนดอาร์กิวเมนต์เองตามที่โค้ดรองรับ

**คัดลอกคำสั่ง (ตัวอย่าง):**

```bash
python src/train_varlen.py \
  --data_root ./data \
  --epochs 20 \
  --batch_size 4 \
  --image_size 224 \
  --checkpoint_path ./checkpoints/varlen_best.h5
```

---

## การทำนายผล (Inference)

หลังฝึกเสร็จ ใช้สคริปต์ `src/predict_varlen.py` เพื่อทำนายวิดีโอใหม่ โดยต้องระบุพาธโมเดล (`--model`) และไฟล์วิดีโอ (`--video`)

**คัดลอกคำสั่ง:**

```bash
cd src
python predict_varlen.py --model ../checkpoints/varlen_best.h5 --video /path/to/your/video.mp4
```

**อธิบาย:**

- `--model` : พาธไฟล์โมเดลที่ฝึกแล้ว (เช่น `../checkpoints/varlen_best.h5`)
- `--video` : พาธไฟล์วิดีโอที่ต้องการตรวจสอบ

---

## ผลลัพธ์ของโมเดล

โมเดลจะทำนาย เเละแสดง *probability* และ *คลาส* ที่คาดการณ์  `0 = real`, `1 = deepfake` พร้อมสรุปผลทางหน้าจอ&#x20;

---

