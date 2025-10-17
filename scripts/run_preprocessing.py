import os
from src.preprocessing.preprocess_pipeline import process_trial

DATA_ROOT = "/content/drive/MyDrive/RA/data"
CAR_CACHE = os.path.join(DATA_ROOT, "car_cache")
SAVE_DIR  = os.path.join(DATA_ROOT, "processed_ecog_final")

os.makedirs(CAR_CACHE, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

subjects = [f"sub_{i}" for i in range(1, 11)]
trials = [f"trial{str(i).zfill(3)}" for i in range(7)]

for subj in subjects:
    for trial in trials:
        print(f"\n=== Processing {subj}_{trial} ===")
        process_trial(subj, trial, DATA_ROOT, CAR_CACHE, SAVE_DIR)
        
