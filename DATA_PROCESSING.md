# Data Preprocessing and Alignment Workflow for Sentence-Level ECoG Analysis

## 1. Overview

We developed a complete preprocessing and alignment pipeline for ECoG data recorded while subjects watched audiovisual movies, with the goal of extracting sentence-aligned high-gamma neural activity. The pipeline includes electrode selection and cleaning, median common average referencing (CAR), notch filtering, Hilbert-based high-gamma feature extraction, despiking, and sentence-level alignment.  

This work uses the Brain Treebank dataset (Wang et al., 2024), which contains 43 hours of intracranial ECoG recordings from ten participants watching Hollywood movies. Each movie was manually transcribed, aligned to the audio at the word level, and parsed into Universal Dependencies (UD), resulting in more than 223,000 words and 38,000 sentences with rich linguistic and neural annotations.  


## 2. ECoG Data Preprocessing Pipeline

### 2.1 Electrode Selection

Electrode names were cleaned by removing special characters such as `*`, `#`, and `_` from the original metadata. A predefined list of corrupted electrodes (provided in `corrupted_elec.json` from Wang et al., 2024) was used to exclude noisy channels with excessive artifacts or hardware failures.  

This selection ensures that subsequent referencing and filtering are based on physiologically reliable electrodes, reducing contamination from malfunctioning contacts.


### 2.2 Common Average Referencing (CAR)

To remove global artifacts and improve signal interpretability, we applied median-based common average referencing (CAR) across all clean electrodes within each trial.  
For computational efficiency, CAR vectors were precomputed in chunks of one million samples and stored in cache to avoid redundant computation.  

The median (rather than the mean) was chosen to reduce the influence of outlier electrodes, which often exhibit abnormally high amplitudes. This referencing step centers each electrode’s potential relative to the overall cortical field, highlighting local neural activity while attenuating shared noise components such as motion or amplifier drift.


### 2.3 Notch Filtering and High-Gamma Extraction

To eliminate electrical interference, we applied IIR notch filters at 60, 120, 180, and 240 Hz, corresponding to the powerline frequency and its harmonics in the United States.  

High-gamma activity was then extracted using a Hilbert-based analytic envelope method. Specifically:

1. The signal was filtered into multiple Gaussian-shaped sub-bands between 70 and 150 Hz, logarithmically spaced using the `auto_bands()` function from the Hamilton iEEG Preprocessing toolkit.  
2. For each band, the analytic signal was computed via the Hilbert transform.  
3. The amplitude envelope (absolute value of the analytic signal) was obtained and averaged across all bands to represent broadband high-gamma power.  

This approach captures high-frequency population activity that correlates closely with local neuronal firing rates and perceptual processes during speech and movie viewing.


### 2.4 Despiking

High-gamma signals occasionally exhibit transient high-amplitude artifacts caused by electrode saturation or mechanical disturbance.  
These spikes can bias amplitude-based metrics if not properly controlled. To mitigate this issue, we applied a continuous amplitude compression method using a scaled hyperbolic tangent function:

\[
x_{\text{clean}} = n \times \tanh(x / n)
\]

where \( n = 6.0 \) defines the soft saturation threshold in standard deviation units.  
This approach smoothly limits extreme values while preserving the fine temporal structure of genuine neural fluctuations.

Unlike hard-thresholding methods that replace outliers, this nonlinear compression maintains continuity and avoids abrupt changes in signal slope, which is particularly important for maintaining the interpretability of time-frequency and ERP analyses.  

This `tanh`-based soft clipping has been widely used in neural signal denoising for its simplicity, differentiability, and effectiveness in suppressing transient artifacts without distorting physiological dynamics.  
In practice, this method substantially reduced isolated spikes while maintaining comparable spectral characteristics across electrodes.



## 3. Sentence-Level Alignment

### 3.1 Sentence Extraction

Sentence boundaries were defined using the `sentence_idx` column from the token-level transcript files.  
For each sentence, we determined its onset and offset as the earliest and latest word times within the group.  
Each sentence record also includes its text and speaker identity (when available).

This step converts fine-grained word-level annotations into broader sentence-level units that are more suitable for cognitive and linguistic analyses.


### 3.2 Playback Discontinuity Detection

To ensure alignment integrity, playback interruptions were identified from the trigger signal recorded during movie presentation.  
Under normal conditions, trigger indices increase linearly. We computed the difference between consecutive indices and marked discontinuities where the change exceeded mean + 10× standard deviation, excluding extreme outliers.

These discontinuities correspond to pauses or dropped frames in the stimulus, and were stored as “gap” intervals for exclusion.  
This method effectively captures both long pauses and short dropped segments while maintaining temporal precision.


### 3.3 Sentence Filtering

Any sentence whose time span overlapped with a playback gap was excluded.  
Specifically, if the sentence onset or offset fell within any discontinuity range, that sentence was removed.  
This ensures that all retained sentences correspond to uninterrupted stimulus playback and continuous ECoG recordings.

The resulting dataset was saved as `cleaned_sentences.csv` for reproducibility and downstream alignment.


### 3.4 Sentence-Aligned ECoG Extraction and Baseline Correction

For each clean sentence, we extracted the corresponding neural segment from the preprocessed high-gamma signal.  
The extraction window extended from −1.0 seconds before sentence onset to the sentence offset, providing sufficient context for potential anticipatory activity.  

Baseline normalization was applied by z-scoring each segment using the −0.6 to −0.1 second pre-onset window:

\[
z = \frac{x - \mu_{\text{baseline}}}{\sigma_{\text{baseline}} + 10^{-6}}
\]

This produced a set of sentence-aligned, baseline-normalized ECoG clips across electrodes and trials.  
These aligned clips were later used for event-related potential (ERP) computation and sentence-level encoding analysis.

Each aligned dataset was saved as a trial-specific `.pkl` file under `results/sentence_aligned_final/`.

| Column | Description |
|--------|-------------|
| subject | The experiment subject |
| trial | The experiment trial |
| movie | Which movie the subject in this trial watched |
| speaker | Who spoke the sentence |
| sentence | Text of the spoken sentence |
| start, end | Start and end time in the movie |
| start_idx, end_idx | Sample indices in ECoG |
| ecog_clip | Z-scored neural segment array |


## 4. Parameter Selection and Rationale

| Parameter | Value | Rationale |
|------------|--------|-----------|
| Sampling frequency (fs) | 2048 Hz | Preserves high-gamma activity while ensuring temporal resolution. |
| Notch frequencies | 60, 120, 180, 240 Hz | Correspond to powerline noise and harmonics. |
| CAR type | Median | More robust to outlier channels than mean. |
| Baseline window | −0.6 to −0.1 s | Captures pre-onset activity while avoiding sentence-related responses. |
| Discontinuity threshold | mean + 10×SD | Empirically identifies playback gaps without false positives. |
| Despike function | 6 × tanh(x / 6) | Smoothly limits amplitude outliers without hard clipping. |


## 5. Discussion and Justification

The preprocessing and alignment procedures were designed to maximize both the physiological validity of the extracted neural signals and the temporal precision of their alignment with audiovisual stimuli. The focus on high-gamma activity (70–150 Hz) stems from extensive evidence that this frequency range reliably reflects local cortical population firing, particularly in auditory and speech-related regions. By emphasizing this band, the processed signals retain direct interpretability in terms of cortical information processing during movie comprehension.

A key consideration in the design of this pipeline was the suppression of non-neural artifacts while preserving the fine temporal and spectral structure of genuine neural responses. Median-based common average referencing (CAR) was chosen over the conventional mean approach to ensure robustness against outlier electrodes, which often exhibit excessive noise or unstable baselines. Similarly, the despiking procedure used a smooth hyperbolic tangent function to compress transient high-amplitude artifacts without abrupt truncation or loss of continuity. Together, these steps enhance the stability of the high-gamma envelope and reduce contamination from motion, amplifier drift, and electrical interference.

Temporal alignment between ECoG recordings and movie stimuli was achieved through precise trigger-based synchronization, which corrects for playback pauses or timing drifts inherent in long audiovisual presentations. This ensures that each extracted ECoG segment faithfully corresponds to its intended sentence in the stimulus, maintaining sub-frame temporal accuracy essential for cognitive and linguistic analyses.

The overall pipeline is modular, deterministic, and parameter-transparent, allowing reproducibility and easy adaptation to other datasets. Empirical validation using power spectral density (PSD) and event-related potential (ERP) analyses confirmed that the preprocessing sequence effectively suppresses noise while preserving meaningful neural dynamics. The resulting high-gamma signals exhibit the expected 1/f spectral characteristics and clear temporal alignment with linguistic events, demonstrating both the efficacy and physiological plausibility of the implemented methods.


## 6. Summary

In summary, this preprocessing and alignment workflow converts raw intracranial ECoG recordings into clean, sentence-aligned high-gamma representations that are ready for subsequent neural encoding, decoding, and linguistic modeling analyses. Each component—from electrode cleaning and CAR referencing to Hilbert-based feature extraction, soft despiking, and sentence-level alignment—was carefully designed to balance noise suppression with signal preservation. The resulting dataset provides a temporally precise and physiologically interpretable basis for studying the neural dynamics of speech and audiovisual comprehension. The methodology follows and extends established best practices in recent ECoG research (Mesgarani et al., 2014; Akbari et al., 2019; Tang et al., 2024), emphasizing reproducibility, interpretability, and alignment accuracy as core principles of analysis.

