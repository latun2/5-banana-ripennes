# ============================================
# TEST DETEKSI PISANG - PROSES SEMUA GAMBAR
# ============================================

import sys
import os
import cv2
import torch

# 1. SETUP PATH YOLOv5
yolov5_path = r'D:\semester_6\compvist\yolo5_test\yolov5'
if yolov5_path not in sys.path:
    sys.path.insert(0, yolov5_path)
print("✅ Path YOLOv5 ditambahkan")

# 2. IMPORT MODUL
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device
from utils.plots import Annotator, colors

print("✅ Modul berhasil di-import")

# 3. LOAD MODEL
model_path = r'D:\semester_6\compvist\yolo5_test\best.pt'
print("🔄 Memuat model...")
device = select_device('cpu')
model = DetectMultiBackend(model_path, device=device, dnn=False)
stride = model.stride
imgsz = check_img_size(640, s=stride)
print(f"✅ Model siap! Kelas: {model.names}")

# 4. BUAT FOLDER HASIL
folder_hasil = r'D:\semester_6\compvist\yolo5_test\hasil_test'
os.makedirs(folder_hasil, exist_ok=True)
print(f"📁 Folder hasil: {folder_hasil}")

# 5. CARI SEMUA GAMBAR
folder_gambar = r'D:\semester_6\compvist\yolo5_test'

gambar_list = []
for file in os.listdir(folder_gambar):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')) and 'hasil' not in file.lower():
        gambar_list.append(os.path.join(folder_gambar, file))

if not gambar_list:
    print(f"\n❌ Tidak ada gambar!")
    exit()

print(f"\n✅ Ditemukan {len(gambar_list)} gambar:")
for g in gambar_list:
    print(f"   - {os.path.basename(g)}")

# 6. PROSES SEMUA GAMBAR
print("\n" + "="*50)
print("MEMULAI PROSES DETEKSI (Confidence = 10%)")
print("="*50)

for i, gambar_path in enumerate(gambar_list, 1):
    nama_file = os.path.basename(gambar_path)
    nama_depan, ekstensi = os.path.splitext(nama_file)
    hasil_path = os.path.join(folder_hasil, f'{nama_depan}_hasil{ekstensi}')
    
    print(f"\n[{i}/{len(gambar_list)}] 🔍 Memproses: {nama_file}")
    
    dataset = LoadImages(gambar_path, img_size=imgsz, stride=stride)
    
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.float() / 255.0
        if len(im.shape) == 3:
            im = im[None]
        
        pred = model(im)
        # 🔥 PERUBAHAN DI SINI: conf_thres dari 0.25 jadi 0.1
        pred = non_max_suppression(pred, 0.1, 0.45, None, False, max_det=1000)
        
        for det in pred:
            im0 = im0s.copy()
            annotator = Annotator(im0, line_width=3, example=str(model.names))
            
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))
            
            cv2.imwrite(hasil_path, annotator.result())
            
            # Tampilkan jumlah objek yang terdeteksi
            jumlah_objek = len(det) if det is not None else 0
            print(f"   ✅ Terdeteksi {jumlah_objek} objek pisang")
            print(f"   📁 Hasil: {os.path.basename(hasil_path)}")
            
            cv2.imshow(f'Hasil Deteksi - {nama_file}', annotator.result())
            print("   📸 Tekan tombol apa saja untuk lanjut...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

print("\n" + "="*50)
print(f"✅ SELESAI!")
print("="*50)
 