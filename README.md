# Visual-Audio-Speech: ECoG Alignment and Preprocessing Pipeline

This repository contains the complete preprocessing and alignment pipeline for intracranial ECoG recordings collected during movie-watching experiments (Brain Treebank dataset). The goal is to clean, preprocess, and align high-gamma ECoG activity with sentence-level annotations from audiovisual movie stimuli.   
Below gives the instructions on how to use the code. For detailed explanations of preprocessing and alignment techniques, see the [DATA_PROCESSING.md](./DATA_PROCESSING.md) file.


## Overview

The pipeline includes three main stages:

1. **Sentence Metadata Preparation**  
   Extracts sentence-level time ranges and removes playback interruptions.  
   - Input: raw transcript features and trigger timing files  
   - Output: `cleaned_sentences.csv` and `playback_interrupts.csv`

2. **ECoG Preprocessing**  
   - Median common average referencing (CAR)  
   - Notch filtering (60, 120, 180, 240 Hz)  
   - Hilbert transform and high-gamma envelope extraction  
   - Despiking and saving clean electrode signals as `.npy` files

3. **Sentence-Level ECoG Alignment**  
   Aligns z-scored ECoG clips with each sentence timestamp.


## Data Description

All input data comes from the [Brain Treebank dataset (MIT CSAIL, 2024)](https://braintreebank.dev).

Directory structure (not uploaded to GitHub):

```
data/
 ├── subject_metadata/
 │    ├── sub_1_trial000.json
 │    ├── sub_2_trial001.json
 │    └── ...
 ├── subject_timings/
 │    ├── sub_1_trial000_timings.csv
 │    ├── sub_2_trial001_timings.csv
 │    └── ...
 ├── transcripts/
 │    └── <movie_name>/features.csv
 ├── electrode_labels/
 │    └── electrode_labels.json
 └── corrupted_elec.json
```


## Project Structure

```
Visual-Audio-Speech/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── preprocessing/
│   │   ├── prepare_sentences.py         # Extract and clean sentence ranges
│   │   └── preprocess_pipeline.py       # ECoG preprocessing (filtering, CAR, Hilbert)
│   └── alignment/
│       └── sentence_align.py            # Align ECoG clips with sentences
│
├── scripts/
│   ├── run_preprocessing.py             # Entry for ECoG preprocessing
│   └── run_sentence_align.py            # Entry for alignment step
│
└── data/                                # (not tracked; local only)
```


## Installation

Create a new Python environment and install dependencies:

```bash
pip install -r requirements.txt
```


## Running the Pipeline

### Step 1 — Generate Sentence Metadata

```bash
python src/preprocessing/prepare_sentences.py
```

Output:
- `cleaned_sentences.csv`  
- `playback_interrupts.csv`


### Step 2 — Run ECoG Preprocessing

```bash
python scripts/run_preprocessing.py
```

This will:
- Apply median-CAR referencing  
- Perform notch and Hilbert filtering  
- Save each electrode as `.npy` under `results/processed_ecog_final/`


### Step 3 — Align ECoG with Sentences

```bash
python scripts/run_sentence_align.py
```

Output (per trial):

```
results/sentence_aligned_final/
 ├── sub_1_trial000_aligned.pkl
 ├── sub_1_trial001_aligned.pkl
 └── ...
```

Each `.pkl` file contains one row per sentence with:

| Column | Description |
|--------|-------------|
| subject | The experiment subject |
| trial | The experiment trial |
| movie | Which movie the subject in this trial watched |
| speaker | Who spoke the sentence |
| sentence | Text of the spoken sentence |
| start, end | Start and end time in movie |
| start_idx, end_idx | Sample indices in ECoG |
| ecog_clip | Z-scored neural segment array |

