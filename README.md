# Visual-Audio-Speech: ECoG Alignment and Preprocessing Pipeline

This repository contains the complete preprocessing and alignment pipeline for intracranial ECoG recordings collected during movie-watching experiments (Brain Treebank dataset).  
The goal is to clean, preprocess, and align high-gamma ECoG activity with sentence-level annotations from audiovisual movie stimuli.

---

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

---

## Data Description

All input data comes from the Brain Treebank dataset (MIT CSAIL, 2024): https://braintreebank.dev

Directory structure (not uploaded to GitHub):

