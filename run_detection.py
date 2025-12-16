from ultralytics import YOLO
import cv2
import math
import pygame
import os

# Initialize pygame mixer
pygame.mixer.init()
SOUND_PATH = 'sensor-beep.mp3'
if os.path.exists(SOUND_PATH):
    pygame.mixer.music.load(SOUND_PATH)
else:
    print(f"WARNING: Sound file {SOUND_PATH} not found!")

# --- KONFIGURASI ---
# Pastikan file best.pt sudah kamu copy ke folder yang sama dengan script ini
MODEL_PATH = 'best.pt' 

# Ganti 0 untuk Webcam Laptop. 
# Jika ingin tes pakai video file, ganti jadi 'path/to/video.mp4'
VIDEO_SOURCE = 1 

# Tingkat keyakinan (0.0 - 1.0). 
# Jika AI ragu-ragu (< 0.5), tidak usah dideteksi biar gak false alarm.
CONF_THRESHOLD = 0.5 
# -------------------

# 1. Load Model Custom Kamu
print(f"Memuat model dari: {MODEL_PATH}...")
try:
    model = YOLO(MODEL_PATH)
    print("Model berhasil dimuat!")
    print("Class names yang dikenali model:", model.names) # Cek nama kelas di sini (0: Fall, 1: Standing, dst)
except Exception as e:
    print(f"ERROR: Gagal memuat model. Pastikan file '{MODEL_PATH}' ada.")
    exit()

# 2. Setup Kamera
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("ERROR: Tidak bisa membuka kamera/video.")
    exit()

# Warna (B, G, R)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
ORANGE = (0, 165, 255)

print("--- DETEKSI DIMULAI ---")
print("Tekan 'q' untuk keluar.")

while True:
    success, img = cap.read()
    if not success:
        break

    # 3. Proses Deteksi (Inference)
    # stream=True membuat proses lebih ringan
    results = model(img, stream=True, conf=CONF_THRESHOLD, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Ambil koordinat kotak
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Ambil Confidence & Class
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            current_class = model.names[cls]

            # 4. Logika Warna & Peringatan
            # Default warna hijau
            color = GREEN
            label_text = f"{current_class} {conf}"
            
            # Cek label (Sesuaikan string ini dengan nama di dataset Roboflow kamu!)
            # Cek label: 'Fall', 'No fall'
            class_lower = current_class.lower()
            
            if ('fall' in class_lower) and 'no' not in class_lower:
                color = RED
                # Tambah teks peringatan besar di layar
                cv2.putText(img, "WARNING: FALL DETECTED!", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, RED, 3)
                
                # Play sound if not already playing
                if not pygame.mixer.music.get_busy() and os.path.exists(SOUND_PATH):
                    pygame.mixer.music.play()


            # 5. Gambar Kotak & Label
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            
            # Background untuk teks label biar terbaca
            t_size = cv2.getTextSize(label_text, 0, fontScale=0.6, thickness=1)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, color, -1) 
            cv2.putText(img, label_text, (x1, y1 - 2), 0, 0.6, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    # Tampilkan
    cv2.imshow("Fall Detection System", img)

    # Keluar dengan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()