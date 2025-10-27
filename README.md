# Visual-Audio-Speech: ECoG Preprocessing and Alignment Pipeline

This repository contains the complete preprocessing and alignment pipeline for intracranial ECoG recordings collected during movie-watching experiments (Brain Treebank dataset). The goal is to clean, preprocess, and align high-gamma ECoG activity with sentence-level annotations from audiovisual movie stimuli.   
Below gives the instructions on how to use the code. For detailed explanations of preprocessing and alignment techniques, see the [DATA_PROCESSING.md](./DATA_PROCESSING.md) file.

The official Brain Treebank code release already includes several useful [preprocessing](https://github.com/czlwang/brain_treebank_code_release/blob/master/data/h5_data_reader.py) steps such as notch filtering, band-pass filtering, rereferencing, and a simple despike function. In their implementation, spikes are detected when the z-score exceeds four, and nearby samples are reduced by a fixed factor. While this removes large artifacts, it uses one global threshold and can either miss small transients or suppress real neural bursts.

To make this step more physiologically adaptive, we applied a soft-clipping despiking method based on a scaled hyperbolic tangent function:
x_clean = n × tanh(x / n), where n = 6.0 defines the soft saturation threshold in standard-deviation units. This smooth nonlinear compression limits extreme values without hard thresholding and maintains the natural temporal structure of neural signals. Compared with the simple z-score–based clipping in the original Brain Treebank preprocessing, our approach better preserves genuine high-gamma fluctuations while suppressing brief electrode-saturation artifacts.

We also noticed that some movie sessions in the Brain Treebank dataset contain playback interruptions. Since the original release did not describe how these gaps were handled, we remove sentences that occur near interruptions to reduce timing drift and ensure stable alignment between the movie and the neural data.

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
- Despike  
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
