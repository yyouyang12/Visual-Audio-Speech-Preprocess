import os, json, gc, h5py
import numpy as np
from tqdm import tqdm
from scipy import signal
from numpy.lib.format import open_memmap
from ecog_preproc_utils import transformData, auto_bands, applyHilbertTransform
import cupy as cp

SFREQ = 2048
NOTCHES = [60, 120, 180, 240]
CHUNK = 1_000_000

NOTCH_FILTERS = [(signal.iirnotch(f0 / (SFREQ / 2), Q=30)) for f0 in NOTCHES]

# ========== GPU Hilbert ==========
def applyHilbertTransform_gpu(X, rate, center, sd):
    X_gpu = cp.asarray(X)
    T = X_gpu.shape[-1]
    freq = cp.fft.fftfreq(T, 1/rate)
    h = cp.zeros(len(freq), dtype=cp.float32)
    h[freq > 0] = 2.; h[0] = 1.
    k = cp.exp((-(cp.abs(freq)-center)**2)/(2*(sd**2)))
    Xc = cp.fft.ifft(cp.fft.fft(X_gpu) * h * k)
    return cp.asnumpy(Xc)

# ========== Electrode utils ==========
def get_all_electrodes(subject, data_root):
    path = os.path.join(data_root, "electrode_labels", "electrode_labels", subject, "electrode_labels.json")
    with open(path, 'r') as f: raw = json.load(f)
    return [e.replace("*", "").replace("#", "").replace("_", "") for e in raw]

def get_clean_electrodes(subject, data_root):
    all_elecs = get_all_electrodes(subject, data_root)
    with open(os.path.join(data_root, "corrupted_elec.json"), 'r') as f:
        corrupted = json.load(f)
    return list(set(all_elecs) - set(corrupted.get(subject, [])))

# ========== CAR median ==========
def compute_car_median(subject, trial, data_root, car_cache_dir):
    car_path = os.path.join(car_cache_dir, f"{subject}_{trial}_CAR_median.npy")
    if os.path.exists(car_path): return car_path
    h5_path = os.path.join(data_root, f"{subject}_{trial}.h5", f"{subject}_{trial}.h5")
    clean_labels = get_clean_electrodes(subject, data_root)
    all_labels = get_all_electrodes(subject, data_root)
    idx_map = {name: i for i, name in enumerate(all_labels)}

    with h5py.File(h5_path, "r") as f:
        root = f["data"]
        n_time = root["electrode_0"].shape[0]
        os.makedirs(car_cache_dir, exist_ok=True)
        car = open_memmap(car_path, mode="w+", dtype=np.float32, shape=(n_time,))
        for pos in range(0, n_time, CHUNK):
            end = min(pos + CHUNK, n_time)
            block = np.stack([root[f"electrode_{idx_map[e]}"][pos:end] for e in clean_labels])
            car[pos:end] = np.median(block, axis=0)
        del car; gc.collect()
    return car_path

# ========== Electrode preprocessing ==========
def preprocess_electrode(subject, trial, electrode_idx, data_root, car_path):
    h5_path = os.path.join(data_root, f"{subject}_{trial}.h5", f"{subject}_{trial}.h5")
    with h5py.File(h5_path, "r") as f:
        raw = f["data"][f"electrode_{electrode_idx}"][:].astype(np.float32)

    car = np.load(car_path, mmap_mode="r")
    raw -= car
    for b, a in NOTCH_FILTERS:
        raw = signal.lfilter(b, a, raw)

    cts, sds = auto_bands()
    envelopes = [np.abs(applyHilbertTransform_gpu(raw, SFREQ, c, sd)) for c, sd in zip(cts, sds)]
    hg = np.mean(envelopes, axis=0)
    cleaned = 6.0 * np.tanh(hg / 6.0)
    return cleaned

# ========== Trial-level processing ==========
def process_trial(subject, trial, data_root, car_cache_dir, save_dir):
    print(f"Processing {subject}_{trial}")
    electrodes = get_clean_electrodes(subject, data_root)
    if not electrodes: return False

    car_path = compute_car_median(subject, trial, data_root, car_cache_dir)
    all_labels = get_all_electrodes(subject, data_root)
    idx_map = {name: i for i, name in enumerate(all_labels)}

    trial_dir = os.path.join(save_dir, f"{subject}_{trial}")
    os.makedirs(trial_dir, exist_ok=True)
    for e in tqdm(electrodes, desc=f"{subject}_{trial}"):
        out = os.path.join(trial_dir, f"{subject}_{trial}_{e}_median.npy")
        if os.path.exists(out): continue
        idx = idx_map[e]
        np.save(out, preprocess_electrode(subject, trial, idx, data_root, car_path))
    return True

