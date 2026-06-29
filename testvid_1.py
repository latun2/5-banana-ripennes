# ============================================
# TEST DETEKSI VIDEO DENGAN YOLOv5
# ============================================

import sys
import os
import cv2
import torch
import numpy as np

# 1. SETUP PATH YOLOv5
yolov5_path = r'D:\semester_6\compvist\yolo5_test\yolov5'
if yolov5_path not in sys.path:
    sys.path.insert(0, yolov5_path)
print("✅ Path YOLOv5 ditambahkan")

# 2. IMPORT MODUL
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device
from utils.plots import Annotator, colors
from utils.augmentations import letterbox

print("✅ Modul berhasil di-import")

# 3. LOAD MODEL
model_path = r'D:\semester_6\compvist\yolo5_test\best.pt'
print("🔄 Memuat model...")
device = select_device('cpu')
model = DetectMultiBackend(model_path, device=device, dnn=False)
stride = model.stride
imgsz = check_img_size(640, s=stride)
print(f"✅ Model siap! Kelas: {model.names}")

# 4. PARAMETER DETEKSI
conf_thres = 0.25   # Confidence threshold
iou_thres = 0.45    # IOU threshold

# 5. CARI SEMUA VIDEO DI FOLDER
folder_video = r'D:\semester_6\compvist\yolo5_test'

video_list = []
for file in os.listdir(folder_video):
    if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        video_list.append(os.path.join(folder_video, file))

if not video_list:
    print(f"\n❌ Tidak ada video di folder: {folder_video}")
    exit()

print(f"\n✅ Ditemukan {len(video_list)} video:")
for i, v in enumerate(video_list, 1):
    print(f"   {i}. {os.path.basename(v)}")

# 6. PILIH VIDEO YANG AKAN DIPROSES
print("\n" + "="*50)
pilih = input(f"Pilih video (1-{len(video_list)}) atau 'all' untuk semua: ")

if pilih.lower() == 'all':
    video_terpilih = video_list
else:
    try:
        idx = int(pilih) - 1
        video_terpilih = [video_list[idx]]
    except:
        print("❌ Pilihan tidak valid!")
        exit()

# 7. BUAT FOLDER HASIL
folder_hasil = r'D:\semester_6\compvist\yolo5_test\hasil_test'
os.makedirs(folder_hasil, exist_ok=True)

# 8. PROSES VIDEO SATU PER SATU
for video_path in video_terpilih:
    nama_video = os.path.basename(video_path)
    nama_depan, ekstensi = os.path.splitext(nama_video)
    output_video = os.path.join(folder_hasil, f'{nama_depan}_hasil{ekstensi}')
    
    print(f"\n🎬 Memproses: {nama_video}")
    print(f"   Output: {output_video}")
    
    # Buka video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    frame_count = 0
    print(f"   Total frame: {total_frames}, FPS: {fps}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Proses frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = letterbox(img, imgsz, stride=stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        if len(img.shape) == 3:
            img = img[None]
        
        # Deteksi
        pred = model(img)
        pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=1000)
        
        # Gambar bounding box
        for det in pred:
            im0 = frame.copy()
            annotator = Annotator(im0, line_width=2, example=str(model.names))
            
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))
            
            out.write(annotator.result())
        
        # Progress setiap 100 frame
        if frame_count % 100 == 0:
            print(f"   Progress: {frame_count}/{total_frames} ({100*frame_count/total_frames:.1f}%)")
    
    cap.release()
    out.release()
    print(f"   ✅ Selesai! Hasil: {output_video}")

print("\n" + "="*50)
print(f"✅ SELESAI! {len(video_terpilih)} video diproses.")
print(f"📁 Semua hasil di: {folder_hasil}")
print("="*50)