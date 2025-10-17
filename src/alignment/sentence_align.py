"""
Sentence-level ECoG Alignment Pipeline
--------------------------------------
Aligns preprocessed ECoG data (median-CAR, Hilbert) with sentence-level annotations.

Functions:
- load_ecog_matrix: load preprocessed ECoG signals for a trial
- process_one_sentence: extract z-scored clip around each sentence
- align_sentences_with_ecog: align all sentences in a trial (parallel)
- batch_align_all_sentences: batch process all subjects and trials
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from src.preprocessing.preprocess_pipeline import get_clean_electrodes

# ==========================================
# Load preprocessed ECoG data for one trial
# ==========================================
def load_ecog_matrix(subject, trial, save_dir, data_root):
    """
    Load preprocessed ECoG data for one trial.
    Args:
        subject (str): subject ID (e.g., "sub_1")
        trial (str): trial ID (e.g., "trial000")
        save_dir (str): directory containing processed electrode .npy files
        data_root (str): root directory containing metadata and labels
    Returns:
        ecog (ndarray): shape [n_channels, n_time]
        clean_electrodes (list): names of clean electrodes
    """
    trial_dir = os.path.join(save_dir, f"{subject}_{trial}")
    if not os.path.exists(trial_dir):
        raise FileNotFoundError(f"No preprocessed data found for {subject}_{trial}")

    clean_electrodes = get_clean_electrodes(subject, data_root)
    ecog = []
    missing = []

    for elec in clean_electrodes:
        fpath = os.path.join(trial_dir, f"{subject}_{trial}_{elec}_median.npy")
        if os.path.exists(fpath):
            ecog.append(np.load(fpath, mmap_mode="r"))
        else:
            missing.append(elec)

    if missing:
        print(f"[WARN] {subject}_{trial}: missing {len(missing)} electrodes, skipped")

    if not ecog:
        return None, []

    ecog = np.stack(ecog, axis=0)
    return ecog, clean_electrodes


# ==========================================
# Sentence-level processing (for parallel use)
# ==========================================
def process_one_sentence(onset, end, ecog_data, fs):
    """
    Extract and z-score an ECoG clip around a sentence.
    Includes 1 s pre-onset, baseline normalization (-600 to -100 ms).
    """
    win_start = int(onset - 1.0 * fs)
    win_end = int(end)
    if win_start < 0 or win_end > ecog_data.shape[1] or win_end <= win_start:
        return None

    clip = ecog_data[:, win_start:win_end]

    b_start = int(onset - 0.6 * fs)
    b_end = int(onset - 0.1 * fs)
    if b_start < 0 or b_end > ecog_data.shape[1] or b_end <= b_start:
        return None

    baseline = ecog_data[:, b_start:b_end]
    mean = baseline.mean(axis=1, keepdims=True)
    std = baseline.std(axis=1, keepdims=True) + 1e-6

    return (clip - mean) / std


# ==========================================
# Parallel sentence alignment for one trial
# ==========================================
def align_sentences_with_ecog(subject, trial, ecog_data, sentence_table, fs=2048, n_jobs=-1):
    """
    Align ECoG data with all sentences for one trial.
    Returns a dataframe with z-scored ECoG clips for each sentence.
    """
    starts = sentence_table["start_idx"].values
    ends = sentence_table["end_idx"].values

    clips = Parallel(n_jobs=n_jobs)(
        delayed(process_one_sentence)(st, ed, ecog_data, fs)
        for st, ed in tqdm(zip(starts, ends), total=len(starts), desc=f"{subject}_{trial}")
    )

    sentence_df = sentence_table.copy()
    sentence_df["ecog_clip"] = clips
    sentence_df["subject"] = subject
    sentence_df["trial"] = trial

    keep_cols = [
        "subject", "trial", "movie", "speaker", "sentence_idx", "sentence",
        "start", "end", "start_idx", "end_idx", "ecog_clip"
    ]
    return sentence_df[[c for c in keep_cols if c in sentence_df.columns]]


# ==========================================
# Batch alignment across all subjects/trials
# ==========================================
def batch_align_all_sentences(subject_trial_map_path, sentence_path,
                              trigger_root, ecog_data_dir, save_root,
                              data_root, fs=2048, debug=False, max_sentences=None):
    """
    Batch-align ECoG data with all sentences across subjects and trials.
    Saves a pickle file per trial with z-scored clips per sentence.
    """
    os.makedirs(save_root, exist_ok=True)
    df_map = pd.read_csv(subject_trial_map_path)
    sentence_df = pd.read_csv(sentence_path)

    for _, row in df_map.iterrows():
        subject, trial = row["subject"], row["trial"]

        if debug and not (subject == "sub_1" and trial == "trial000"):
            continue

        save_path = os.path.join(save_root, f"{subject}_{trial}_aligned.pkl")
        if os.path.exists(save_path):
            print(f"[skip] Already aligned: {save_path}")
            continue

        print(f"\n=== Aligning {subject}_{trial} ===")

        # 1. Load ECoG
        try:
            ecog, electrodes = load_ecog_matrix(subject, trial, ecog_data_dir, data_root)
        except FileNotFoundError:
            print(f"[skip] No preprocessed data for {subject}_{trial}")
            continue

        # 2. Load trigger
        trig_file = os.path.join(trigger_root, f"{subject}_{trial}_timings.csv")
        if not os.path.exists(trig_file):
            print(f"[skip] Missing trigger: {trig_file}")
            continue
        trigger_df = pd.read_csv(trig_file)

        # 3. Load sentence annotations
        sents = sentence_df[(sentence_df["subject"] == subject) & (sentence_df["trial"] == trial)].copy()
        sents = sents.dropna(subset=["start", "end"])
        if sents.empty:
            print(f"[skip] No valid sentences for {subject}-{trial}")
            continue
        if max_sentences:
            sents = sents.head(max_sentences)

        # 4. Compute sample indices
        def estimate_index_from_trigger(movie_time):
            diffs = np.abs(trigger_df["movie_time"] - movie_time)
            nearest_idx = diffs.idxmin()
            t_trigger = trigger_df.loc[nearest_idx, "movie_time"]
            i_trigger = trigger_df.loc[nearest_idx, "index"]
            return int(round(i_trigger + (movie_time - t_trigger) * fs))

        sents["start_idx"] = sents["start"].apply(lambda t: estimate_index_from_trigger(t))
        sents["end_idx"] = sents["end"].apply(lambda t: estimate_index_from_trigger(t))

        # 5. Align ECoG clips
        aligned = align_sentences_with_ecog(subject, trial, ecog, sents, fs=fs)

        # 6. Save results
        aligned.to_pickle(save_path)
        print(f"[✓] Saved {len(aligned)} sentences → {save_path}")
