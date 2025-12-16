from ultralytics import YOLO
import torch
import os

def main():
    # 1. Cek Hardware (GPU vs CPU)
    # Ini penting biar kamu tau ekspektasi waktunya
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- SYSTEM CHECK ---")
    print(f"Training using: {device.upper()}")
    if device == 'cpu':
        print("WARNING: Training menggunakan CPU akan sangat lambat.")
        print("Saran: Kurangi 'epochs' menjadi 5-10 saja untuk tes.")
    else:
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
    print("--------------------")

    # 2. Load Model
    # Kita mulai dari 'yolov8n.pt' (nano) karena paling ringan buat laptop
    model = YOLO('yolov8n.pt') 

    # 3. Path ke data.yaml
    # Ganti string di bawah ini sesuai lokasi file data.yaml hasil download kamu!
    # Contoh: 'datasets/data.yaml' atau 'C:/Projects/FallDetection/data.yaml'
    data_path = 'datasets-1/data.yaml' 

    # Cek apakah file data ada
    if not os.path.exists(data_path):
        print(f"ERROR: File {data_path} tidak ditemukan!")
        print("Pastikan kamu sudah download dataset dan path-nya benar.")
        return

    # 4. Mulai Training
    # imgsz=640: Resolusi standar
    # epochs=50: Jumlah putaran belajar
    # batch=8: Jumlah gambar diproses sekali jalan (Kecilkan jadi 4 kalau laptop nge-lag)
    results = model.train(
        data=data_path,
        imgsz=640,
        epochs=100, 
        batch=8,
        name='fall_detection_model',
        device=0 if device == 'cuda' else 'cpu'
    )

    print("--- TRAINING SELESAI ---")
    print(f"Model terbaik tersimpan di: runs/detect/fall_detection_model/weights/best.pt")

if __name__ == '__main__':
    # Windows butuh ini untuk multiprocessing
    main()