# Visual-Audio-Speech: ECoG Alignment and Preprocessing Pipeline

This repository contains the complete preprocessing and alignment pipeline for intracranial ECoG recordings collected during movie-watching experiments (Brain Treebank dataset).  
The goal is to clean, preprocess, and align high-gamma ECoG activity with sentence-level annotations from audiovisual movie stimuli.


## Overview

The pipeline includes three main stages:

1. **Sentence Metadata Preparation**  
   Extracts sentence-level time ranges and removes playback interruptions.  
   - Input: raw transcript features and trigger timing files  
   - Output: cleaned_sentences.csv and playback_interrupts.csv

2. **ECoG Preprocessing**  
   - Median common average referencing (CAR)  
   - Notch filtering (60, 120, 180, 240 Hz)  
   - Hilbert transform and high-gamma envelope extraction  
   - Despiking and saving clean electrode signals as .npy files

3. **Sentence-Level ECoG Alignment**  
   Aligns z-scored ECoG clips with each sentence timestamp.


## Data Description

All input data comes from the Brain Treebank dataset (MIT CSAIL, 2024): https://braintreebank.dev

Directory structure (not uploaded to GitHub):
data/
├── subject_metadata/
│ ├── sub_1_trial000.json
│ ├── sub_2_trial001.json
│ └── ...
├── subject_timings/
│ ├── sub_1_trial000_timings.csv
│ ├── sub_2_trial001_timings.csv
│ └── ...
├── transcripts/
│ └── <movie_name>/features.csv
├── electrode_labels/
│ └── electrode_labels.json
└── corrupted_elec.json


## Project Structure
Visual-Audio-Speech/
│
├── README.md
├── requirements.txt
│
├── src/
│ ├── preprocessing/
│ │ ├── prepare_sentences.py # Extract and clean sentence ranges
│ │ └── preprocess_pipeline.py # ECoG preprocessing (filtering, CAR, Hilbert)
│ └── alignment/
│ └── sentence_align.py # Align ECoG clips with sentences
│
├── scripts/
│ ├── run_preprocessing.py # Entry for ECoG preprocessing
│ └── run_sentence_align.py # Entry for alignment step
│
├── results/
│ ├── processed_ecog_final/ # Cleaned electrode data (.npy)
│ └── sentence_aligned_final/ # Sentence-level aligned pickle files (.pkl)
└── data/ # (not tracked; local only)



## Installation

Create a new Python environment and install dependencies:

```bash
pip install -r requirements.txt

Running the Pipeline
Step 1 — Generate Sentence Metadata
python src/preprocessing/prepare_sentences.py


Output:

cleaned_sentences.csv

playback_interrupts.csv

Step 2 — Run ECoG Preprocessing
python scripts/run_preprocessing.py


This will:

Apply median-CAR referencing

Perform notch and Hilbert filtering

Save each electrode as .npy under results/processed_ecog_final/


