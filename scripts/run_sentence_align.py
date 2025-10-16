import os
from src.alignment.sentence_align import batch_align_all_sentences

DATA_ROOT = "/content/drive/MyDrive/RA/data"
SAVE_DIR = os.path.join(DATA_ROOT, "processed_ecog_final")
ALIGN_DIR = os.path.join(DATA_ROOT, "sentence_aligned_final")

batch_align_all_sentences(
    subject_trial_map_path=os.path.join(DATA_ROOT, "subject_trial_movie_mapping.csv"),
    sentence_path=os.path.join(DATA_ROOT, "cleaned_sentences.csv"),
    trigger_root=os.path.join(DATA_ROOT, "subject_timings", "subject_timings"),
    ecog_data_dir=SAVE_DIR,
    save_root=ALIGN_DIR,
    data_root=DATA_ROOT,
    fs=2048,
    debug=False
)
