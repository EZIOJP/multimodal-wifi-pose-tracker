import os
import urllib.request
import sys

MODELS = {
    "pose_landmarker.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "gesture_recognizer.task": "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task"
}

def download_models(target_dir="models"):
    os.makedirs(target_dir, exist_ok=True)
    for name, url in MODELS.items():
        path = os.path.join(target_dir, name)
        if os.path.exists(path) and os.path.getsize(path) > 100000:
            print(f"[-] {name} already exists ({os.path.getsize(path)} bytes)")
            continue
        print(f"[+] Downloading {name} from {url}...")
        try:
            urllib.request.urlretrieve(url, path)
            print(f"[OK] Downloaded {name} ({os.path.getsize(path)} bytes)")
        except Exception as e:
            print(f"[ERROR] Failed to download {name}: {e}")
            return False
    return True

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    if download_models(models_dir):
        print("\nAll models ready!")
    else:
        sys.exit(1)
