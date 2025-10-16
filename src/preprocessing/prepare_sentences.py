"""
Prepare sentence-level annotation files for ECoG alignment
----------------------------------------------------------
This script:
1. Loads transcript feature files and timing trigger files.
2. Detects playback interruptions in timing data.
3. Extracts valid (continuous) sentence ranges.
4. Outputs:
   - cleaned_sentences.csv : valid sentences with movie/subject/trial info
   - playback_interrupts.csv : detected stop/start gaps in movie playback
"""

import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from google.colab import drive

# Mount Google Drive if using Colab
drive.mount('/content/drive')

# ==========================================
# Path setup
# ==========================================
BASE_DIR = "/content/drive/MyDrive/RA"
TRANSCRIPT_ROOT = os.path.join(BASE_DIR, "data/transcripts/transcripts")
TIMING_ROOT = os.path.join(BASE_DIR, "data/subject_timings/subject_timings")
MAPPING_CSV = os.path.join(BASE_DIR, "subject_trial_movie_mapping.csv")
SAVE_DIR = os.path.join(BASE_DIR, "processed")

os.makedirs(SAVE_DIR, exist_ok=True)

# Load subject-trial mapping
df_mapping = pd.read_csv(MAPPING_CSV)

# ==========================================
# Detect playback discontinuities
# ==========================================
def find_discontinuity_ranges(timing_df, threshold_multiplier=10):
    """
    Detect playback gaps by looking for large index jumps in trigger timings.

    Args:
        timing_df (DataFrame): timing file containing "index" and "start_time"
        threshold_multiplier (float): how many SDs above mean to mark as discontinuity

    Returns:
        list of (start_time, end_time, duration)
    """
    idx_diff = timing_df['index'].diff().fillna(0)

    # Calculate mean and std ignoring extreme outliers
    filtered = idx_diff[(idx_diff > 0) & (idx_diff < idx_diff.quantile(0.99))]
    mean_val = filtered.mean()
    std_val = filtered.std()
    threshold = mean_val + threshold_multiplier * std_val

    discontinuity = idx_diff > threshold
    discontinuity_indices = discontinuity[discontinuity].index

    discontinuity_ranges = []
    for idx in discontinuity_indices:
        start_time = timing_df.loc[idx - 1, 'start_time']
        end_time = timing_df.loc[idx, 'start_time']
        duration = end_time - start_time
        discontinuity_ranges.append((start_time, end_time, duration))
    return discontinuity_ranges


# ==========================================
# Extract sentence onset/offset per sentence_idx
# ==========================================
def extract_sentence_ranges(feature_df):
    """
    Group word-level transcript features into sentence-level time ranges.

    Args:
        feature_df (DataFrame): CSV containing start, end, sentence_idx, sentence

    Returns:
        DataFrame: columns [sentence_idx, start, end, sentence, speaker]
    """
    sentence_groups = feature_df.groupby("sentence_idx")
    ranges = []
    for sid, group in sentence_groups:
        onset = group['start'].min()
        offset = group['end'].max()
        sentence = group.iloc[0]['sentence']
        speaker = group.iloc[0]['speaker'] if 'speaker' in group else None
        ranges.append({
            "sentence_idx": sid,
            "start": onset,
            "end": offset,
            "sentence": sentence,
            "speaker": speaker
        })
    return pd.DataFrame(ranges)


# ==========================================
# Main loop for all subject-trials
# ==========================================
output_cleaned_sentences = []
output_interrupts = []

for _, row in tqdm(df_mapping.iterrows(), total=len(df_mapping), desc="Processing trials"):
    subject = row['subject']
    trial = row['trial']
    movie_name = row['filename']

    # Paths to transcript and timing files
    feature_file = os.path.join(TRANSCRIPT_ROOT, movie_name, "features.csv")
    timing_file = os.path.join(TIMING_ROOT, f"{subject}_{trial}_timings.csv")

    # Skip if files not found
    if not os.path.exists(feature_file):
        print(f"[Skip] Feature file not found for {movie_name}")
        continue
    if not os.path.exists(timing_file):
        print(f"[Skip] Timing file not found for {subject}-{trial}")
        continue

    print(f"[Processing] {subject}-{trial} | {movie_name}")

    # Load transcript features and timing data
    feature_df = pd.read_csv(feature_file)
    timing_df = pd.read_csv(timing_file)
    timing_df = timing_df[timing_df['type'] == 'trigger'].reset_index(drop=True)

    # Get sentence ranges
    sentence_df = extract_sentence_ranges(feature_df)

    # Detect playback gaps
    gaps = find_discontinuity_ranges(timing_df)
    gap_ranges = [(s, e) for s, e, _ in gaps]

    # Check if each sentence overlaps with playback gaps
    def in_gap(start, end, gap_ranges):
        for gs, ge in gap_ranges:
            if start <= ge and end >= gs:
                return True
        return False

    sentence_df['in_gap'] = sentence_df.apply(
        lambda row: in_gap(row['start'], row['end'], gap_ranges), axis=1
    )

    # Keep only clean (non-gap) sentences
    clean_df = sentence_df[~sentence_df['in_gap']].copy()
    clean_df['movie'] = movie_name
    clean_df['subject'] = subject
    clean_df['trial'] = trial
    output_cleaned_sentences.append(clean_df)

    # Record gap metadata
    for start, end, duration in gaps:
        output_interrupts.append({
            "subject": subject,
            "trial": trial,
            "movie": movie_name,
            "gap_start": start,
            "gap_end": end,
            "gap_duration": duration
        })


# ==========================================
# Save outputs
# ==========================================
df_cleaned = pd.concat(output_cleaned_sentences, ignore_index=True)
df_interrupts = pd.DataFrame(output_interrupts)

df_cleaned.to_csv(os.path.join(BASE_DIR, "cleaned_sentences.csv"), index=False)
df_interrupts.to_csv(os.path.join(BASE_DIR, "playback_interrupts.csv"), index=False)

print(f"Saved cleaned_sentences.csv ({len(df_cleaned)} rows)")
print(f"Saved playback_interrupts.csv ({len(df_interrupts)} rows)")
