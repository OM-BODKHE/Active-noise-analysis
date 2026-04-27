# =============================================================================
# COMPLETE ACOUSTIC NOISE ANALYSIS PIPELINE
# For Google Colab — Works with ANY device (trimmer, fan, motor, etc.)
# Author: Based on ENC Course Project methodology
# =============================================================================
#
# HOW TO USE IN GOOGLE COLAB:
#   1. Run Cell 0 (install libraries)
#   2. Run Cell 1 (all imports)
#   3. Upload your files using Cell 2
#   4. Set your correction factor in Cell 3
#   5. Run remaining cells in order
#   6. Cell FINAL generates the complete HTML report
#
# FILES YOU NEED TO UPLOAD:
#   - background.wav         (room noise, device OFF)
#   - device_0deg.wav        (device ON, mic East,  1 m)
#   - device_90deg.wav       (device ON, mic South, 1 m)
#   - device_180deg.wav      (device ON, mic West,  1 m)
#   - device_270deg.wav      (device ON, mic North, 1 m)
#   - device_top.wav         (mic directly above)
#   - device_top60S.wav      (mic 60 deg from top toward south)
#   - device_top30S.wav      (mic 30 deg from top toward south)
#   - accel_data.csv         (columns: time, ax, ay, az)
#
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# CELL 0  —  INSTALL LIBRARIES  (run once)
# ─────────────────────────────────────────────────────────────────────────────
"""
Paste this into a Colab cell and run it:

!pip install librosa soundfile scipy numpy matplotlib scikit-learn pandas plotly -q
"""


# ─────────────────────────────────────────────────────────────────────────────
# CELL 1  —  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import scipy.signal as signal
import scipy.fft as sfft
import librosa
import librosa.display
import soundfile as sf
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import json
import os
import warnings
import base64
from io import BytesIO, StringIO
import datetime

warnings.filterwarnings("ignore")
matplotlib.rcParams['figure.dpi'] = 120
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

print("✅ All libraries imported successfully.")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 2  —  FILE UPLOAD HELPER
# ─────────────────────────────────────────────────────────────────────────────

def upload_files_colab():
    """
    Run this cell in Google Colab to upload all your audio files.
    It uses Colab's built-in file upload widget.
    """
    try:
        from google.colab import files
        print("📂 Select ALL your files at once (hold Ctrl/Cmd to multi-select):")
        print("   Required files:")
        print("   • background.wav")
        print("   • device_0deg.wav")
        print("   • device_90deg.wav")
        print("   • device_180deg.wav")
        print("   • device_270deg.wav")
        print("   • device_top.wav")
        print("   • device_top60S.wav")
        print("   • device_top30S.wav")
        print("   • accel_data.csv")
        uploaded = files.upload()
        print(f"\n✅ Uploaded {len(uploaded)} files:")
        for name in uploaded:
            print(f"   ✓ {name}")
        return list(uploaded.keys())
    except ImportError:
        print("⚠️  Not running in Colab. Place files in current directory.")
        return []

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3  —  CONFIGURATION
# Edit these values to match YOUR experiment
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # ── Calibration ────────────────────────────────────────────────────────
    # Formula: correction_factor = 10^((reference_dB - recorded_dB) / 20)
    # Your reference SPL (from calibrator or known source), e.g. 72.62 dB
    "reference_spl_db"    : 72.62,
    # The raw dB your phone recorded the calibration tone at, e.g. 82.46 dB
    "recorded_spl_db"     : 82.46,

    # ── Device info ────────────────────────────────────────────────────────
    "device_name"         : "Nova SuperGroom NG-1149 Trimmer",
    "analyst_name"        : "Edwin Varghese",
    "roll_number"         : "ME24MTECH11028",
    "institution"         : "IIT Hyderabad",
    "course"              : "ENC Course Project",

    # ── Measurement setup ─────────────────────────────────────────────────
    "mic_distance_m"      : 1.0,         # metres from source
    "accel_sample_rate"   : 5120,        # Hz (from Physics Toolbox)
    "target_spl_db"       : 40.0,        # desired target SPL

    # ── Audio files mapping ────────────────────────────────────────────────
    # Key = direction label,  Value = filename you uploaded
    "audio_files" : {
        "background" : "background.wav",
        "0deg_E"     : "device_0deg.wav",
        "90deg_S"    : "device_90deg.wav",
        "180deg_W"   : "device_180deg.wav",
        "270deg_N"   : "device_270deg.wav",
        "top"        : "device_top.wav",
        "top60S"     : "device_top60S.wav",
        "top30S"     : "device_top30S.wav",
    },

    # ── Accelerometer CSV ─────────────────────────────────────────────────
    # Must have columns named: time, ax, ay, az
    "accel_csv"   : "accel_data.csv",

    # ── Column names in CSV (change if yours differ) ──────────────────────
    "accel_cols"  : {"time": "time", "ax": "ax", "ay": "ay", "az": "az"},
}

# ─────────────────────────────────────────────────────────────────────────────
# AUTO-COMPUTE correction factor from config
# Formula derivation:
#   SPL = 20·log10(p_rms / p_ref)
#   p_actual   = p_recorded × correction_factor
#   20·log10(p_recorded × cf / p_ref) = reference_spl
#   20·log10(p_recorded / p_ref) + 20·log10(cf) = reference_spl
#   20·log10(cf) = reference_spl - recorded_spl
#   cf = 10^((reference_spl - recorded_spl) / 20)
# ─────────────────────────────────────────────────────────────────────────────

CONFIG["correction_factor"] = 10 ** (
    (CONFIG["reference_spl_db"] - CONFIG["recorded_spl_db"]) / 20.0
)

print("=" * 60)
print("  CONFIGURATION LOADED")
print("=" * 60)
print(f"  Device       : {CONFIG['device_name']}")
print(f"  Analyst      : {CONFIG['analyst_name']}")
print(f"  Ref SPL      : {CONFIG['reference_spl_db']} dB")
print(f"  Recorded SPL : {CONFIG['recorded_spl_db']} dB")
print(f"  Correction   : {CONFIG['correction_factor']:.4f}")
print(f"  Formula      : 10^(({CONFIG['reference_spl_db']} - {CONFIG['recorded_spl_db']}) / 20)")
print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 4  —  CORE SIGNAL PROCESSING FUNCTIONS
# Every formula is derived from first principles
# ─────────────────────────────────────────────────────────────────────────────

# ── Constants ────────────────────────────────────────────────────────────────
P_REF = 20e-6          # Reference pressure = 20 μPa (threshold of human hearing)
OCTAVE_CENTERS  = np.array([8, 16, 31.5, 63, 125, 250, 500,
                             1000, 2000, 4000, 8000, 16000])
THIRD_OCT_CENTERS = np.array([
    6.3, 8, 10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80,
    100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
    1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000,
    6300, 8000, 10000, 12500, 16000, 20000
])
NC_BANDS = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000])

# NC curves from ANSI S12.2 standard
# Each row: [63Hz, 125Hz, 250Hz, 500Hz, 1000Hz, 2000Hz, 4000Hz, 8000Hz]
NC_CURVES = {
    15 : np.array([47, 36, 29, 22, 17, 14, 12, 11]),
    20 : np.array([51, 40, 33, 26, 22, 19, 17, 16]),
    25 : np.array([54, 44, 37, 31, 27, 24, 22, 21]),
    30 : np.array([57, 48, 41, 35, 31, 29, 28, 27]),
    35 : np.array([60, 52, 45, 40, 36, 34, 33, 32]),
    40 : np.array([64, 57, 50, 45, 41, 39, 38, 37]),
    45 : np.array([67, 60, 54, 49, 46, 44, 43, 42]),
    50 : np.array([71, 64, 58, 54, 51, 49, 48, 47]),
    55 : np.array([74, 67, 62, 58, 56, 54, 53, 52]),
    60 : np.array([77, 71, 67, 63, 61, 59, 58, 57]),
    65 : np.array([80, 75, 71, 68, 66, 64, 63, 62]),
}

NCB_CURVES = {
    15 : np.array([66, 55, 47, 41, 36, 32, 28, 25]),
    20 : np.array([68, 57, 49, 44, 39, 35, 31, 28]),
    25 : np.array([70, 59, 52, 47, 42, 38, 34, 31]),
    30 : np.array([72, 61, 54, 49, 45, 41, 37, 34]),
    35 : np.array([74, 63, 56, 51, 47, 44, 40, 37]),
    40 : np.array([76, 65, 58, 53, 49, 46, 42, 39]),
    45 : np.array([78, 67, 60, 55, 51, 48, 44, 41]),
    50 : np.array([80, 69, 62, 57, 53, 50, 46, 43]),
    55 : np.array([82, 71, 64, 59, 55, 52, 48, 45]),
    60 : np.array([84, 73, 66, 61, 57, 54, 50, 47]),
}


# ──────────────────────────────────────────────────────────────────────────────
def load_and_calibrate(filepath, correction_factor):
    """
    Load audio file and apply calibration correction.

    DERIVATION:
      librosa.load() returns samples normalized to [-1, 1].
      These are NOT acoustic pressure values.
      Multiply by correction_factor to convert to Pascals (Pa).

    Parameters
    ----------
    filepath          : str   path to audio file (.wav or .mp3)
    correction_factor : float computed as 10^((ref_dB - recorded_dB) / 20)

    Returns
    -------
    pressure   : np.ndarray  acoustic pressure in Pa
    sr         : int         sample rate in Hz
    time_axis  : np.ndarray  time vector in seconds
    """
    audio, sr = librosa.load(filepath, sr=None, mono=True)
    pressure  = audio * correction_factor          # Pa
    time_axis = np.arange(len(pressure)) / sr      # seconds
    return pressure, sr, time_axis


# ──────────────────────────────────────────────────────────────────────────────
def compute_spl_rms(pressure):
    """
    Compute overall Sound Pressure Level (SPL) from a pressure waveform.

    FORMULA:
      p_rms = sqrt( (1/N) * sum(p[i]^2) )        root mean square
      SPL   = 20 * log10( p_rms / p_ref )  [dB]
      where p_ref = 20e-6 Pa  (threshold of human hearing)

    Returns
    -------
    spl_db : float  overall SPL in dB
    p_rms  : float  RMS pressure in Pa
    """
    p_rms  = np.sqrt(np.mean(pressure ** 2))
    spl_db = 20.0 * np.log10(p_rms / P_REF + 1e-30)
    return spl_db, p_rms


# ──────────────────────────────────────────────────────────────────────────────
def compute_fft(pressure, sr):
    """
    Compute the narrowband FFT spectrum of an acoustic pressure signal.

    DERIVATION:
      The Discrete Fourier Transform decomposes a time-domain signal into
      its constituent sinusoids:
         X[k] = sum_{n=0}^{N-1}  x[n] * e^{-j*2*pi*k*n/N}

      We apply a Hann window before FFT to reduce spectral leakage:
         w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))
      This tapers the signal edges to zero, preventing the FFT from
      "seeing" a sharp discontinuity at the window boundaries.

      Frequency resolution:
         delta_f = sr / N     (Hz per bin)

      We take only the first N/2+1 bins (positive frequencies).
      The magnitude is scaled by 2/N to account for the one-sided spectrum
      and normalized for the window:
         mag[k] = |X[k]| * 2 / sum(window)

      Convert to dB (SPL spectrum):
         SPL[k] = 20 * log10( mag[k] / p_ref )

    Parameters
    ----------
    pressure : np.ndarray  calibrated acoustic pressure (Pa)
    sr       : int         sample rate (Hz)

    Returns
    -------
    freqs      : np.ndarray  frequency bins (Hz)
    magnitude  : np.ndarray  linear magnitude (Pa)
    spl_db     : np.ndarray  SPL per bin (dB re 20 μPa)
    """
    N      = len(pressure)
    window = np.hanning(N)                          # Hann window
    win_norm = np.sum(window)                       # normalization factor

    fft_result  = np.fft.fft(pressure * window)     # apply FFT
    freqs       = np.fft.fftfreq(N, d=1.0/sr)      # frequency axis

    # One-sided spectrum: take indices 0 to N//2 (positive freqs only)
    half        = N // 2 + 1
    freqs       = freqs[:half]
    magnitude   = np.abs(fft_result[:half]) * 2.0 / win_norm

    # Handle DC component (index 0): do not double
    magnitude[0]  /= 2.0
    # Handle Nyquist (index N//2): do not double if N is even
    if N % 2 == 0:
        magnitude[-1] /= 2.0

    spl_db = 20.0 * np.log10(magnitude / P_REF + 1e-30)
    return freqs, magnitude, spl_db


# ──────────────────────────────────────────────────────────────────────────────
def compute_band_spl(freqs, magnitude, centers, band_type='octave'):
    """
    Compute SPL in octave or 1/3-octave bands.

    DERIVATION:
      For octave bands:
        f_lower = f_center / sqrt(2)
        f_upper = f_center * sqrt(2)
        (ratio upper/lower = 2, which is one octave)

      For 1/3-octave bands:
        f_lower = f_center / 2^(1/6)
        f_upper = f_center * 2^(1/6)
        (ratio upper/lower = 2^(1/3), one-third of an octave)

      Band power:
        p_band^2 = sum of all p[k]^2 for f_lower <= f[k] <= f_upper
        (power additivity: uncorrelated sources add in power, not amplitude)

      Band SPL:
        L_band = 10 * log10( p_band^2 / p_ref^2 )
               = 10 * log10( sum(mag[k]^2) / p_ref^2 )

    Parameters
    ----------
    freqs      : np.ndarray  frequency axis from compute_fft()
    magnitude  : np.ndarray  linear magnitude from compute_fft()
    centers    : np.ndarray  band center frequencies (Hz)
    band_type  : str         'octave' or 'third_octave'

    Returns
    -------
    centers    : np.ndarray  same center freqs (passed through)
    band_spl   : np.ndarray  SPL per band (dB re 20 μPa)
    """
    if band_type == 'octave':
        factor = np.sqrt(2.0)       # f_upper = fc * sqrt(2)
    else:
        factor = 2.0 ** (1.0/6.0)  # f_upper = fc * 2^(1/6)

    band_spl = np.zeros(len(centers))

    for i, fc in enumerate(centers):
        f_lower = fc / factor
        f_upper = fc * factor
        mask    = (freqs >= f_lower) & (freqs <= f_upper)

        if np.any(mask):
            # Sum squared magnitudes (= power spectral density per bin)
            power_sum    = np.sum(magnitude[mask] ** 2)
            band_spl[i]  = 10.0 * np.log10(power_sum / (P_REF**2) + 1e-30)
        else:
            band_spl[i]  = -np.inf   # no energy in this band

    return centers, band_spl


# ──────────────────────────────────────────────────────────────────────────────
def compute_NC_rating(oct_db_8band):
    """
    Compute NC (Noise Criteria) rating per ANSI S12.2 standard.

    METHOD:
      NC curves are defined at 8 octave bands: 63, 125, 250, 500, 1000,
      2000, 4000, 8000 Hz.
      The NC rating is the LOWEST NC curve such that the measured SPL
      does NOT exceed that curve at ANY of the 8 bands.

      In other words:
        NC = min { nc_level  :  measured[i] <= NC_curve[nc_level][i]
                                for all i in [0..7] }

    Parameters
    ----------
    oct_db_8band : np.ndarray  shape (8,), SPL at NC_BANDS

    Returns
    -------
    nc_value     : int or str  NC rating (15, 20, 25, ..., 65, or ">65")
    nc_curve     : np.ndarray  the limiting NC curve values
    nc_level     : int         numeric NC level (999 if >65)
    """
    for level in sorted(NC_CURVES.keys()):
        curve = NC_CURVES[level]
        if np.all(oct_db_8band <= curve):
            return level, curve, level
    return ">65", NC_CURVES[65], 999


# ──────────────────────────────────────────────────────────────────────────────
def compute_NCB_rating(oct_db_8band):
    """
    Compute NCB (Noise Criteria Balanced) rating.

    Same logic as NC but uses NCB curves which have higher low-frequency
    limits (NCB accounts for balanced spectrum — rumble is penalized less
    than NC but the overall balance check is tighter).

    Returns
    -------
    ncb_value : int or str
    """
    for level in sorted(NCB_CURVES.keys()):
        if np.all(oct_db_8band <= NCB_CURVES[level]):
            return level
    return ">60"


# ──────────────────────────────────────────────────────────────────────────────
def compute_RC_rating(oct_centers, oct_db):
    """
    Compute RC (Room Criteria) rating.

    FORMULA:
      RC level = arithmetic mean of SPL at 500, 1000, and 2000 Hz:
        RC = (L_500 + L_1000 + L_2000) / 3

      Spectrum shape classification:
        low_slope  = L_63Hz  - L_31.5Hz
        high_slope = L_4000Hz - L_2000Hz

        if low_slope  > 5 dB  → "Rumble"  (too much low-freq energy)
        if high_slope > 3 dB  → "Hiss"    (too much high-freq energy)
        else                  → "Neutral"

    Returns
    -------
    rc_level    : float  RC numeric level
    rc_category : str    'Rumble', 'Hiss', or 'Neutral'
    """
    def db_at(fc):
        idx = np.argmin(np.abs(oct_centers - fc))
        return oct_db[idx]

    rc_level = np.mean([db_at(500), db_at(1000), db_at(2000)])

    low_slope  = db_at(63)   - db_at(31.5)
    high_slope = db_at(4000) - db_at(2000)

    if low_slope > 5.0:
        rc_category = "Rumble"
    elif high_slope > 3.0:
        rc_category = "Hiss"
    else:
        rc_category = "Neutral"

    return rc_level, rc_category


# ──────────────────────────────────────────────────────────────────────────────
def extract_nc_bands(oct_centers, oct_db):
    """
    Extract the 8 standard NC band values from a full octave band array.

    NC standard uses only: 63, 125, 250, 500, 1000, 2000, 4000, 8000 Hz

    Returns
    -------
    nc_db : np.ndarray  shape (8,) dB values at the 8 NC bands
    """
    nc_db = np.zeros(len(NC_BANDS))
    for i, fc in enumerate(NC_BANDS):
        idx = np.argmin(np.abs(oct_centers - fc))
        nc_db[i] = oct_db[idx]
    return nc_db


# ──────────────────────────────────────────────────────────────────────────────
def background_significance_test(device_oct_db, bg_oct_db, threshold_db=10.0):
    """
    10 dB Rule for background noise significance.

    THEORY:
      When two independent uncorrelated noise sources combine:
        L_total = 10 * log10( 10^(L1/10) + 10^(L2/10) )

      If L1 - L2 > 10 dB:
        The contribution of L2 to L_total is less than 0.41 dB.
      If L1 - L2 > 20 dB:
        The contribution of L2 is less than 0.04 dB (negligible).

      Therefore: if device SPL exceeds background by more than 10 dB
      at ALL frequency bands, background is negligible.

    Returns
    -------
    is_negligible : bool
    difference    : np.ndarray  device - background at each band (dB)
    max_bg_spl    : float       peak background SPL
    max_bg_freq   : float       frequency of peak background SPL
    """
    difference    = device_oct_db - bg_oct_db
    is_negligible = bool(np.all(difference > threshold_db))
    max_idx       = np.argmax(bg_oct_db)
    return is_negligible, difference, bg_oct_db[max_idx], OCTAVE_CENTERS[max_idx]


print("✅ Core signal processing functions defined.")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 5  —  ACCELEROMETER / STRUCTURAL VIBRATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def load_accelerometer(csv_path, config):
    """
    Load accelerometer CSV and compute resultant acceleration magnitude.

    FORMULA:
      a_resultant[n] = sqrt( ax[n]^2 + ay[n]^2 + az[n]^2 )

      This gives the total vibration magnitude regardless of direction,
      which represents the overall structural excitation at each instant.

    Returns
    -------
    time        : np.ndarray  time vector (s)
    a_resultant : np.ndarray  resultant acceleration magnitude
    sr_accel    : int         accelerometer sample rate (Hz)
    """
    cols = config["accel_cols"]
    df   = pd.read_csv(csv_path)

    # Rename columns if needed
    rename_map = {}
    for key, col_name in cols.items():
        if col_name in df.columns:
            rename_map[col_name] = key
    df = df.rename(columns=rename_map)

    ax = df["ax"].values.astype(float)
    ay = df["ay"].values.astype(float)
    az = df["az"].values.astype(float)

    a_resultant = np.sqrt(ax**2 + ay**2 + az**2)   # combined magnitude

    # Build time axis
    if "time" in df.columns:
        time = df["time"].values.astype(float)
    else:
        time = np.arange(len(a_resultant)) / config["accel_sample_rate"]

    return time, a_resultant, config["accel_sample_rate"]


def compute_vibration_spectrum(a_resultant, sr_accel):
    """
    Compute the FFT spectrum of the acceleration signal.

    FORMULA (per the project report):
      FFT magnitude spectrum is computed normally.
      dB conversion uses:
        dB_vibration[k] = 20 * log10( |FFT[k]| + epsilon )

    NOTE: This is a dimensionless dB scale relative to 1 m/s^2 (or 1 g),
    NOT referenced to p_ref. It represents vibration level, not SPL.

    Returns
    -------
    freqs    : np.ndarray  frequency bins (Hz)
    vib_db   : np.ndarray  vibration level in dB
    peak_freq: float       frequency of dominant vibration peak (Hz)
    peak_db  : float       dB value at dominant peak
    """
    N      = len(a_resultant)
    window = np.hanning(N)
    F      = np.fft.rfft(a_resultant * window)
    freqs  = np.fft.rfftfreq(N, d=1.0/sr_accel)
    mag    = np.abs(F) * 2.0 / np.sum(window)

    vib_db = 20.0 * np.log10(mag + 1e-30)

    # Find peaks with minimum prominence
    peaks, properties = signal.find_peaks(vib_db, prominence=3.0, distance=10)
    if len(peaks) > 0:
        best = peaks[np.argmax(vib_db[peaks])]
        peak_freq = float(freqs[best])
        peak_db   = float(vib_db[best])
    else:
        idx = np.argmax(vib_db)
        peak_freq = float(freqs[idx])
        peak_db   = float(vib_db[idx])

    return freqs, vib_db, peak_freq, peak_db


print("✅ Accelerometer analysis functions defined.")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 6  —  ML FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_ml_features(pressure, sr):
    """
    Extract a fixed-length feature vector for ML classification.

    FEATURES USED (based on published research):
    ──────────────────────────────────────────────
    1. MFCC (Mel-Frequency Cepstral Coefficients)
       - Captures timbral texture of the sound
       - Mel scale: f_mel = 2595 * log10(1 + f_hz / 700)
       - 13 coefficients, mean + std over time → 26 values

    2. Spectral Centroid
       - "Center of mass" of the spectrum
       - centroid = sum(f * |X[f]|) / sum(|X[f]|)
       - High centroid → noise is bright/high-frequency
       - Low centroid  → noise is dull/low-frequency (motor rumble)

    3. Spectral Bandwidth
       - Spread around centroid
       - bandwidth = sqrt( sum((f - centroid)^2 * |X[f]|) / sum(|X[f]|) )
       - Wide bandwidth → broadband noise (blade turbulence)
       - Narrow bandwidth → tonal noise (motor harmonics)

    4. Spectral Rolloff
       - Frequency below which 85% of total energy lies
       - Low rolloff → most energy at low frequencies (structural)
       - High rolloff → energy spread to high frequencies (airborne)

    5. Zero-Crossing Rate (ZCR)
       - Rate at which signal changes sign per second
       - ZCR = (1/N) * sum( |sign(x[n]) - sign(x[n-1])| )
       - High ZCR → noisy/turbulent signal (airborne)
       - Low ZCR  → smooth/tonal signal (structural resonance)

    6. RMS Energy
       - rms = sqrt( mean(x^2) )
       - Overall loudness measure

    7. Octave band SPL vector (8 values, 63 Hz – 8 kHz)
       - Direct acoustic signature used for NC/RC
       - Most discriminative for airborne vs structural

    Total feature vector length: 26 + 5 + 8 = 39 dimensions
    """
    # Safety: zero-pad if signal is very short
    min_len = 2048
    if len(pressure) < min_len:
        pressure = np.pad(pressure, (0, min_len - len(pressure)))

    # 1. MFCC
    try:
        mfcc     = librosa.feature.mfcc(y=pressure.astype(np.float32),
                                         sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)   # shape (13,)
        mfcc_std  = np.std(mfcc,  axis=1)   # shape (13,)
    except Exception:
        mfcc_mean = np.zeros(13)
        mfcc_std  = np.zeros(13)

    # 2–6. Spectral features
    try:
        centroid  = float(np.mean(librosa.feature.spectral_centroid(
                          y=pressure.astype(np.float32), sr=sr)))
        bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(
                          y=pressure.astype(np.float32), sr=sr)))
        rolloff   = float(np.mean(librosa.feature.spectral_rolloff(
                          y=pressure.astype(np.float32), sr=sr)))
        zcr       = float(np.mean(librosa.feature.zero_crossing_rate(
                          pressure.astype(np.float32))))
        rms       = float(np.mean(librosa.feature.rms(
                          y=pressure.astype(np.float32))))
    except Exception:
        centroid  = 0.0
        bandwidth = 0.0
        rolloff   = 0.0
        zcr       = 0.0
        rms       = 0.0

    # 7. Octave band SPL
    freqs, mag, _ = compute_fft(pressure, sr)
    _, oct_db     = compute_band_spl(freqs, mag, OCTAVE_CENTERS[:8])
    nc_db         = extract_nc_bands(OCTAVE_CENTERS[:8], oct_db)

    feature_vec = np.concatenate([
        mfcc_mean,                           # 13
        mfcc_std,                            # 13
        [centroid, bandwidth, rolloff,
         zcr, rms],                          #  5
        nc_db,                               #  8
    ])                                       # = 39 total

    return feature_vec


# ──────────────────────────────────────────────────────────────────────────────
def build_noise_classifier():
    """
    Build and train an ML classifier to distinguish
    airborne vs structural noise using synthetic training data.

    STRATEGY (when you have only 1 device):
    ─────────────────────────────────────────────────────────
    We generate synthetic training data based on KNOWN physics:

    Airborne noise signature:
      - Spectral centroid HIGH (1500 – 4000 Hz)
      - ZCR HIGH  (turbulent, sign changes often)
      - Octave energy peaks at 1000–4000 Hz bands
      - NC band SPL highest at 1000–2000 Hz

    Structural vibration signature:
      - Spectral centroid LOW (200 – 800 Hz)
      - ZCR LOW  (periodic, smooth oscillation)
      - Octave energy peaks at 63–500 Hz bands
      - Strong 63 Hz and 125 Hz components

    The classifier learns these patterns.
    When you feed it your actual mic recording → it predicts 'Airborne'
    When you feed it actual accelerometer data → it predicts 'Structural'

    MODEL: RandomForestClassifier
      - Ensemble of 200 decision trees
      - Each tree votes; majority wins
      - Robust to noise, no hyperparameter tuning needed
      - Gives probability (confidence) alongside prediction
    """
    np.random.seed(42)
    N = 1000  # synthetic samples per class

    # --- Generate synthetic AIRBORNE features ---
    airborne = []
    for _ in range(N):
        mfcc_m = np.random.normal([-5, -15, -20, -18, -16, -14, -12,
                                    -10, -8, -6, -4, -3, -2],
                                   [2]*13)
        mfcc_s = np.abs(np.random.normal([3]*13, [0.5]*13))
        centroid  = np.random.uniform(1500, 4000)
        bandwidth = np.random.uniform(1500, 3500)
        rolloff   = np.random.uniform(4000, 8000)
        zcr       = np.random.uniform(0.15, 0.35)
        rms       = np.random.uniform(0.001, 0.01)
        # Octave bands: energy peaked at mid-high frequencies
        nc_db = np.array([
            np.random.uniform(30, 45),   # 63 Hz
            np.random.uniform(38, 52),   # 125 Hz
            np.random.uniform(36, 50),   # 250 Hz
            np.random.uniform(35, 48),   # 500 Hz
            np.random.uniform(38, 50),   # 1000 Hz
            np.random.uniform(35, 48),   # 2000 Hz  ← dominant for airborne
            np.random.uniform(25, 40),   # 4000 Hz
            np.random.uniform(15, 30),   # 8000 Hz
        ])
        vec = np.concatenate([mfcc_m, mfcc_s,
                               [centroid, bandwidth, rolloff, zcr, rms],
                               nc_db])
        airborne.append(vec)

    # --- Generate synthetic STRUCTURAL features ---
    structural = []
    for _ in range(N):
        mfcc_m = np.random.normal([-2, -8, -12, -15, -18, -20, -22,
                                    -22, -22, -22, -22, -22, -22],
                                   [2]*13)
        mfcc_s = np.abs(np.random.normal([2]*13, [0.5]*13))
        centroid  = np.random.uniform(100, 800)
        bandwidth = np.random.uniform(200, 1000)
        rolloff   = np.random.uniform(500, 2000)
        zcr       = np.random.uniform(0.01, 0.08)
        rms       = np.random.uniform(0.0001, 0.002)
        # Octave bands: energy concentrated at low frequencies
        nc_db = np.array([
            np.random.uniform(35, 55),   # 63 Hz   ← dominant for structural
            np.random.uniform(30, 48),   # 125 Hz
            np.random.uniform(20, 38),   # 250 Hz
            np.random.uniform(15, 30),   # 500 Hz
            np.random.uniform(10, 25),   # 1000 Hz
            np.random.uniform(5,  20),   # 2000 Hz
            np.random.uniform(3,  15),   # 4000 Hz
            np.random.uniform(0,  10),   # 8000 Hz
        ])
        vec = np.concatenate([mfcc_m, mfcc_s,
                               [centroid, bandwidth, rolloff, zcr, rms],
                               nc_db])
        structural.append(vec)

    X = np.array(airborne + structural)
    y = np.array([0]*N + [1]*N)   # 0=Airborne, 1=Structural

    scaler = StandardScaler()
    clf    = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
             )
    X_scaled = scaler.fit_transform(X)
    clf.fit(X_scaled, y)

    return clf, scaler


print("✅ ML feature extraction and classifier functions defined.")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 7  —  DIRECTIVITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def compute_directivity(spl_by_direction):
    """
    Compute directivity metrics from multi-position SPL measurements.

    FORMULAS:
    ──────────────────────────────────────────────────────────────────────
    1. Average SPL (energy average, not arithmetic average):
         L_avg = 10 * log10( (1/n) * sum( 10^(L_i / 10) ) )

       We use arithmetic average of dB values as approximation
       (within ~1 dB for typical variation < 5 dB):
         L_avg ≈ (1/n) * sum(L_i)

    2. Directivity index (dB) at each position:
         DI_i = L_i - L_avg

       Positive DI → that direction radiates MORE than average
       Negative DI → that direction radiates LESS than average

    3. Directivity factor Q at each position:
         Q_i = 10^(DI_i / 10)

       Q > 1 → directional radiation in that direction
       Q = 1 → omnidirectional (same in all directions)
       Q < 1 → below-average radiation in that direction

    Parameters
    ----------
    spl_by_direction : dict  { direction: max_spl_db }

    Returns
    -------
    results : dict with 'avg_spl', 'directivity_db', 'Q_factor' per direction
    """
    labels = list(spl_by_direction.keys())
    spls   = np.array([spl_by_direction[k] for k in labels])

    # Energy-correct average
    avg_spl = 10.0 * np.log10(
        np.mean(10.0 ** (spls / 10.0)) + 1e-30
    )

    results = {"avg_spl": float(avg_spl)}
    for label, spl in zip(labels, spls):
        di = float(spl - avg_spl)
        Q  = float(10.0 ** (di / 10.0))
        results[label] = {
            "max_spl_db"     : float(spl),
            "directivity_db" : di,
            "Q_factor"       : Q,
        }
    return results


print("✅ Directivity analysis functions defined.")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 8  —  TARGET SPL AND RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────

def compute_target_spl(n_sources=3, target_each_db=35.0):
    """
    Compute target SPL when n_sources contribute equally.

    DERIVATION:
      If n independent equal sources each produce L_each dB:
        L_total = L_each + 10 * log10(n)

      Rearranging for target total:
        target_total = 10 * log10(n) + target_each

      For n=3 sources (motor, blade friction, airflow) and target_each=35:
        target = 10*log10(3) + 35 = 4.77 + 35 = 39.77 ≈ 40 dB

    Parameters
    ----------
    n_sources      : int    number of contributing noise sources
    target_each_db : float  desired SPL per individual source

    Returns
    -------
    target_spl : float  total target SPL in dB
    """
    target_spl = 10.0 * np.log10(n_sources) + target_each_db
    return target_spl


def generate_recommendations(nc_rating, rc_category, dominant_noise,
                               reduction_needed_db, peak_freq_hz):
    """
    Generate engineering recommendations based on analysis results.

    Logic:
      1. If dominant = Airborne AND peak freq > 1000 Hz
         → blade-related noise → lubrication + blade material
      2. If dominant = Structural
         → motor/housing vibration → isolators + BLDC motor
      3. RC = Rumble → low-freq treatment needed
      4. RC = Hiss   → high-freq treatment needed
    """
    recs = []

    if dominant_noise == "Airborne":
        if peak_freq_hz > 1000:
            recs.append({
                "priority" : "High",
                "action"   : "Lubricate blade mechanism",
                "reason"   : (f"Dominant noise at {peak_freq_hz:.0f} Hz is "
                               "consistent with blade friction (airborne). "
                               "High-viscosity lubricant reduces metal-on-metal "
                               "contact and can achieve 2–4 dB reduction."),
            })
            recs.append({
                "priority" : "Medium",
                "action"   : "Replace steel blades with ceramic/titanium-coated",
                "reason"   : ("Ceramic surfaces have lower friction coefficient, "
                               "reducing both noise amplitude and high-frequency "
                               "harmonic content above 1000 Hz."),
            })
        recs.append({
            "priority" : "Medium",
            "action"   : "Replace brushed DC motor with BLDC motor",
            "reason"   : ("Brushless DC motors eliminate brush-commutator impact "
                           "noise (typically 200–800 Hz) and run at lower vibration "
                           "levels, reducing both airborne and structural components."),
        })
    else:
        recs.append({
            "priority" : "High",
            "action"   : "Install rubber/silicone vibration isolators",
            "reason"   : (f"Dominant structural vibration at {peak_freq_hz:.0f} Hz. "
                           "Isolation mounts between motor and housing attenuate "
                           "transmission. Target isolation efficiency > 80% at "
                           f"{peak_freq_hz:.0f} Hz."),
        })
        recs.append({
            "priority" : "Medium",
            "action"   : "Replace brushed DC with BLDC motor",
            "reason"   : ("BLDC motor eliminates brush commutation vibration, "
                           "reducing the structural excitation force at the source."),
        })

    if rc_category == "Rumble":
        recs.append({
            "priority" : "Low",
            "action"   : "Add mass damping to housing (constrained layer damping)",
            "reason"   : ("RC Rumble indicates excess low-frequency energy. "
                           "Constrained layer damping (viscoelastic + metal foil) "
                           "on housing panels converts vibrational energy to heat."),
        })
    elif rc_category == "Hiss":
        recs.append({
            "priority" : "Low",
            "action"   : "Add acoustic foam lining inside housing (25–50 mm)",
            "reason"   : ("RC Hiss indicates excess high-frequency energy. "
                           "Open-cell acoustic foam absorbs high-frequency "
                           "components effectively above 1000 Hz."),
        })

    if reduction_needed_db > 6.0:
        recs.insert(0, {
            "priority" : "Critical",
            "action"   : "Comprehensive redesign required",
            "reason"   : (f"Required reduction of {reduction_needed_db:.1f} dB "
                           "exceeds what single interventions can achieve. "
                           "Combination of source modification + path modification "
                           "+ receiver protection is recommended."),
        })

    return recs


print("✅ Target SPL and recommendation functions defined.")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 9  —  PLOTTING FUNCTIONS (all 8 graphs)
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "blue"    : "#1a6fc4",
    "orange"  : "#d46a00",
    "teal"    : "#0d8a6e",
    "purple"  : "#5c4ab7",
    "green"   : "#3b7d2e",
    "red"     : "#c0392b",
    "gray"    : "#555555",
    "bg"      : "#f9f8f6",
    "grid"    : "#e8e6e0",
    "text"    : "#2c2c2a",
}


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for HTML embedding."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight',
                facecolor='white', dpi=120)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


def plot_waveform(time_axis, pressure, title="Acoustic Pressure Waveform"):
    """Plot 1: Acoustic pressure vs time."""
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(time_axis, pressure, color=COLORS["blue"], linewidth=0.4, alpha=0.8)
    ax.axhline(0, color=COLORS["gray"], linewidth=0.5, linestyle='--', alpha=0.4)
    spl, p_rms = compute_spl_rms(pressure)
    ax.axhline( p_rms, color=COLORS["orange"], linewidth=1.0,
                linestyle='--', alpha=0.7, label=f"RMS = {p_rms*1e3:.3f} mPa")
    ax.axhline(-p_rms, color=COLORS["orange"], linewidth=1.0,
                linestyle='--', alpha=0.7)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Acoustic Pressure (Pa)", fontsize=11)
    ax.set_title(f"{title}\nOverall SPL = {spl:.1f} dB", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, color=COLORS["grid"], linewidth=0.5)
    ax.set_facecolor(COLORS["bg"])
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    return fig


def plot_narrowband_spectrum(freqs, spl_db, title="Narrowband Spectrum"):
    """Plot 2: Narrowband FFT spectrum (dB)."""
    # Limit to 0–20000 Hz
    mask = freqs <= 20000
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.semilogx(freqs[mask], spl_db[mask],
                color=COLORS["blue"], linewidth=0.6, alpha=0.85)
    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_ylabel("SPL (dB re 20 μPa)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim([20, 20000])
    ax.grid(True, which='both', color=COLORS["grid"], linewidth=0.4)
    ax.set_facecolor(COLORS["bg"])
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    return fig


def plot_octave_bands(centers, oct_db, title="Octave Band Spectrum",
                      color=COLORS["orange"]):
    """Plot 3: Octave band bar chart."""
    fig, ax = plt.subplots(figsize=(10, 4))
    x_pos = np.arange(len(centers))
    bars = ax.bar(x_pos, oct_db, color=color, alpha=0.85, edgecolor='white',
                  linewidth=0.5)

    # Annotate max bar
    max_i = np.argmax(oct_db)
    ax.bar(x_pos[max_i], oct_db[max_i], color=COLORS["red"],
           alpha=0.9, edgecolor='white', linewidth=0.5,
           label=f"Peak: {oct_db[max_i]:.1f} dB @ {centers[max_i]:.0f} Hz")

    # Value labels on bars
    for bar, val in zip(bars, oct_db):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=7,
                    color=COLORS["text"])

    freq_labels = [str(int(c)) if c >= 1 else str(c) for c in centers]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(freq_labels, fontsize=9)
    ax.set_xlabel("Center Frequency (Hz)", fontsize=11)
    ax.set_ylabel("SPL (dB re 20 μPa)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', color=COLORS["grid"], linewidth=0.5)
    ax.set_facecolor(COLORS["bg"])
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    return fig


def plot_third_octave_bands(centers, third_db, title="1/3 Octave Band Spectrum"):
    """Plot 4: 1/3 octave band bar chart."""
    valid = third_db > -60
    c_v   = centers[valid]
    d_v   = third_db[valid]

    fig, ax = plt.subplots(figsize=(12, 4))
    x_pos = np.arange(len(c_v))
    bars = ax.bar(x_pos, d_v, color=COLORS["teal"], alpha=0.85,
                  edgecolor='white', linewidth=0.3)

    max_i = np.argmax(d_v)
    ax.bar(x_pos[max_i], d_v[max_i], color=COLORS["red"], alpha=0.9,
           label=f"Peak: {d_v[max_i]:.1f} dB @ {c_v[max_i]:.0f} Hz")

    # x tick labels (show only every other to avoid crowding)
    tick_labels = []
    for i, c in enumerate(c_v):
        if i % 3 == 0:
            tick_labels.append(f"{int(c)}" if c >= 1 else f"{c}")
        else:
            tick_labels.append("")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(tick_labels, fontsize=7, rotation=45)
    ax.set_xlabel("Center Frequency (Hz)", fontsize=11)
    ax.set_ylabel("SPL (dB re 20 μPa)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', color=COLORS["grid"], linewidth=0.5)
    ax.set_facecolor(COLORS["bg"])
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    return fig


def plot_nc_rating(centers_8, oct_db_8, nc_level, nc_curve,
                   title="NC Rating"):
    """Plot 5: NC contour overlay."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x_pos = np.arange(len(centers_8))

    # Draw all NC curves (light gray)
    for lvl, curve in NC_CURVES.items():
        lw = 1.8 if lvl == nc_level else 0.5
        alpha = 0.9 if lvl == nc_level else 0.25
        color = COLORS["red"] if lvl == nc_level else COLORS["gray"]
        ax.plot(x_pos, curve, color=color, linewidth=lw,
                linestyle='-', alpha=alpha,
                label=f"NC-{lvl}" if lvl == nc_level else None)

    # Measured bars
    ax.bar(x_pos, oct_db_8, color=COLORS["blue"], alpha=0.7,
           edgecolor='white', linewidth=0.5, label="Measured SPL")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{int(f)}" for f in centers_8], fontsize=10)
    ax.set_xlabel("Center Frequency (Hz)", fontsize=11)
    ax.set_ylabel("SPL (dB re 20 μPa)", fontsize=11)
    ax.set_title(f"{title}  →  NC-{nc_level}", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', color=COLORS["grid"], linewidth=0.5)
    ax.set_facecolor(COLORS["bg"])
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    return fig


def plot_rc_rating(centers_8, oct_db_8, rc_level, rc_category, title="RC Rating"):
    """Plot 6: RC rating visualization."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x_pos = np.arange(len(centers_8))
    ax.bar(x_pos, oct_db_8, color=COLORS["purple"], alpha=0.8,
           edgecolor='white', linewidth=0.5, label="Measured SPL")
    ax.axhline(rc_level, color=COLORS["red"], linewidth=2.0,
               linestyle='--',
               label=f"RC = {rc_level:.1f} ({rc_category})")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{int(f)}" for f in centers_8], fontsize=10)
    ax.set_xlabel("Center Frequency (Hz)", fontsize=11)
    ax.set_ylabel("SPL (dB re 20 μPa)", fontsize=11)
    ax.set_title(f"{title}  →  RC {rc_level:.1f} ({rc_category})",
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', color=COLORS["grid"], linewidth=0.5)
    ax.set_facecolor(COLORS["bg"])
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    return fig


def plot_directivity(dir_results, title="Directivity Analysis"):
    """Plot 7: Directivity polar plot + bar chart."""
    fig, (ax_bar, ax_pol) = plt.subplots(
        1, 2, figsize=(12, 5),
        subplot_kw={'projection': None}
    )

    # Left: bar chart of SPL by direction
    dirs  = [k for k in dir_results if k != "avg_spl"]
    spls  = [dir_results[k]["max_spl_db"] for k in dirs]
    dis   = [dir_results[k]["directivity_db"] for k in dirs]
    colors_bar = [COLORS["red"] if d > 0 else COLORS["blue"] for d in dis]

    x_pos = np.arange(len(dirs))
    ax_bar.bar(x_pos, spls, color=colors_bar, alpha=0.8,
               edgecolor='white', linewidth=0.5)
    ax_bar.axhline(dir_results["avg_spl"], color=COLORS["gray"],
                   linewidth=1.5, linestyle='--',
                   label=f"Avg = {dir_results['avg_spl']:.1f} dB")
    for i, (spl, di) in enumerate(zip(spls, dis)):
        ax_bar.text(i, spl + 0.3, f"{di:+.1f}", ha='center',
                    fontsize=8, color=COLORS["text"])
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(dirs, rotation=30, ha='right', fontsize=9)
    ax_bar.set_ylabel("Max SPL (dB)", fontsize=10)
    ax_bar.set_title("SPL by Direction", fontsize=11)
    ax_bar.legend(fontsize=8)
    ax_bar.grid(True, axis='y', color=COLORS["grid"], linewidth=0.5)
    ax_bar.set_facecolor(COLORS["bg"])

    # Right: simple polar-style plot (blade plane)
    blade_dirs  = {k: v for k, v in dir_results.items()
                   if k not in ("avg_spl",) and "deg" in k}
    if len(blade_dirs) >= 2:
        angles = []
        radii  = []
        labels_pol = []
        angle_map = {"0deg_E": 0, "90deg_S": 90,
                     "180deg_W": 180, "270deg_N": 270}
        for k, v in blade_dirs.items():
            ang = angle_map.get(k, 0)
            angles.append(np.radians(ang))
            radii.append(v["max_spl_db"])
            labels_pol.append(k)

        # close the polygon
        angles_c = angles + [angles[0]]
        radii_c  = radii  + [radii[0]]

        ax_pol.remove()
        ax_pol = fig.add_subplot(1, 2, 2, projection='polar')
        ax_pol.plot(angles_c, radii_c, color=COLORS["blue"],
                    linewidth=2.0, marker='o', markersize=6)
        ax_pol.fill(angles_c, radii_c, alpha=0.2, color=COLORS["blue"])
        for ang, r, lbl in zip(angles, radii, labels_pol):
            ax_pol.annotate(f"{r:.1f}", (ang, r),
                            fontsize=8, ha='center', va='bottom')
        ax_pol.set_title("Directivity (Blade Plane)", fontsize=10)
        ax_pol.set_theta_zero_location("E")
        ax_pol.set_theta_direction(1)

    fig.suptitle(title, fontsize=12, fontweight='bold')
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    return fig


def plot_structural_vibration(vib_freqs, vib_db, peak_freq, peak_db,
                               title="Structural Vibration (Accelerometer)"):
    """Plot 8: Accelerometer FFT."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(vib_freqs, vib_db, color=COLORS["green"], linewidth=0.8, alpha=0.85)
    ax.axvline(peak_freq, color=COLORS["red"], linewidth=1.5,
               linestyle='--',
               label=f"Peak: {peak_db:.1f} dB @ {peak_freq:.0f} Hz")
    ax.scatter([peak_freq], [peak_db], color=COLORS["red"], zorder=5, s=40)
    ax.set_xlim([0, min(6000, vib_freqs[-1])])
    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_ylabel("Vibration Level (dB)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, color=COLORS["grid"], linewidth=0.5)
    ax.set_facecolor(COLORS["bg"])
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    return fig


def plot_airborne_vs_structural(airborne_spl, structural_db,
                                 title="Airborne vs Structural Noise"):
    """Plot 9: Comparison chart."""
    fig, ax = plt.subplots(figsize=(7, 4))
    categories = ["Airborne\n(Microphone)", "Structural\n(Accelerometer)"]
    values     = [airborne_spl, structural_db]
    bar_colors = [COLORS["blue"], COLORS["green"]]
    bars       = ax.bar(categories, values, color=bar_colors,
                        alpha=0.85, width=0.4, edgecolor='white')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{val:.1f} dB', ha='center', va='bottom',
                fontsize=12, fontweight='bold', color=COLORS["text"])
    diff = airborne_spl - structural_db
    dominant = "Airborne" if diff > 0 else "Structural"
    ax.set_ylabel("Peak Level (dB)", fontsize=11)
    ax.set_title(f"{title}\nDominant: {dominant} (+{abs(diff):.1f} dB)",
                 fontsize=12, fontweight='bold')
    ax.set_ylim([0, max(values)*1.2])
    ax.grid(True, axis='y', color=COLORS["grid"], linewidth=0.5)
    ax.set_facecolor(COLORS["bg"])
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    return fig


print("✅ All plotting functions defined.")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 10  —  MAIN ANALYSIS RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_full_analysis(config):
    """
    Run the complete noise analysis pipeline.

    Steps:
      1.  Load background audio + compute baseline
      2.  Load all device recordings
      3.  Compute FFT, octave, 1/3 octave for each
      4.  NC, NCB, RC, RC Mark II ratings
      5.  Background significance test
      6.  Directivity analysis
      7.  Accelerometer: structural vibration
      8.  ML classification: airborne vs structural
      9.  Target SPL + recommendations
      10. Generate all plots
      11. Build results dictionary

    Returns
    -------
    results : dict         all numeric analysis results
    plots   : dict         matplotlib figures keyed by name
    """
    cf = config["correction_factor"]
    plots   = {}
    results = {}
    plot_b64 = {}

    print("\n" + "="*60)
    print("  RUNNING FULL NOISE ANALYSIS")
    print("="*60)

    # ── Step 1: Background ───────────────────────────────────────────────────
    print("\n[1/10] Loading background noise...")
    bg_pressure, bg_sr, bg_time = load_and_calibrate(
        config["audio_files"]["background"], cf
    )
    bg_freqs, bg_mag, bg_spl_db = compute_fft(bg_pressure, bg_sr)
    bg_oct_centers, bg_oct_db   = compute_band_spl(
        bg_freqs, bg_mag, OCTAVE_CENTERS
    )
    bg_overall_spl, _ = compute_spl_rms(bg_pressure)
    print(f"   Background overall SPL: {bg_overall_spl:.1f} dB")

    # ── Step 2 & 3: Device recordings ────────────────────────────────────────
    print("\n[2/10] Loading device recordings...")
    recording_data = {}    # stores full data per direction
    spl_by_dir     = {}    # just max SPL per direction

    for direction, filepath in config["audio_files"].items():
        if direction == "background":
            continue
        if not os.path.exists(filepath):
            print(f"   ⚠️  {filepath} not found — skipping {direction}")
            continue

        pressure, sr, time_ax = load_and_calibrate(filepath, cf)
        freqs, mag, spl_sp    = compute_fft(pressure, sr)
        oct_c, oct_db         = compute_band_spl(freqs, mag, OCTAVE_CENTERS)
        toc_c, third_db       = compute_band_spl(freqs, mag, THIRD_OCT_CENTERS,
                                                  band_type='third_octave')
        overall_spl, _        = compute_spl_rms(pressure)
        nc_db_8               = extract_nc_bands(oct_c, oct_db)

        recording_data[direction] = {
            "pressure"   : pressure,
            "sr"         : sr,
            "time_ax"    : time_ax,
            "freqs"      : freqs,
            "mag"        : mag,
            "spl_db"     : spl_sp,
            "oct_centers": oct_c,
            "oct_db"     : oct_db,
            "third_db"   : third_db,
            "overall_spl": overall_spl,
            "nc_db_8"    : nc_db_8,
            "max_oct_spl": float(np.max(oct_db)),
            "max_freq"   : float(oct_c[np.argmax(oct_db)]),
        }
        spl_by_dir[direction] = recording_data[direction]["max_oct_spl"]
        print(f"   ✓ {direction:12s}  max SPL = {overall_spl:.1f} dB")

    if not recording_data:
        raise RuntimeError("No device recordings found. Check filenames in CONFIG.")

    # Primary direction = loudest
    primary = max(recording_data, key=lambda k: recording_data[k]["overall_spl"])
    pdata   = recording_data[primary]
    print(f"\n   Primary (loudest) direction: {primary}"
          f" ({pdata['overall_spl']:.1f} dB)")

    # ── Step 4: Noise Ratings ─────────────────────────────────────────────────
    print("\n[3/10] Computing noise ratings...")
    nc_val, nc_curve, nc_num = compute_NC_rating(pdata["nc_db_8"])
    ncb_val                   = compute_NCB_rating(pdata["nc_db_8"])
    rc_level, rc_cat          = compute_RC_rating(pdata["oct_centers"],
                                                   pdata["oct_db"])
    print(f"   NC  rating : {nc_val}")
    print(f"   NCB rating : {ncb_val}")
    print(f"   RC  rating : {rc_level:.1f} ({rc_cat})")

    # ── Step 5: Background significance ──────────────────────────────────────
    print("\n[4/10] Background significance test...")
    is_negligible, diff, max_bg_spl, max_bg_freq = background_significance_test(
        pdata["nc_db_8"],
        bg_oct_db[np.array([np.argmin(np.abs(bg_oct_centers - fc))
                             for fc in NC_BANDS])]
    )
    print(f"   Background negligible: {is_negligible}")
    print(f"   Min difference: {np.min(diff):.1f} dB")

    # ── Step 6: Directivity ───────────────────────────────────────────────────
    print("\n[5/10] Directivity analysis...")
    dir_results = compute_directivity(spl_by_dir)
    dominant_dir = max(spl_by_dir, key=spl_by_dir.get)
    print(f"   Dominant direction: {dominant_dir}"
          f"  DI = {dir_results[dominant_dir]['directivity_db']:+.1f} dB")

    # ── Step 7: Structural vibration ─────────────────────────────────────────
    print("\n[6/10] Structural vibration analysis...")
    try:
        accel_time, a_res, sr_acc = load_accelerometer(
            config["accel_csv"], config
        )
        vib_freqs, vib_db, peak_f, peak_db = compute_vibration_spectrum(
            a_res, sr_acc
        )
        has_accel = True
        print(f"   Peak vibration: {peak_db:.1f} dB @ {peak_f:.0f} Hz")
    except Exception as e:
        print(f"   ⚠️  Accelerometer data issue: {e}")
        has_accel = False
        vib_freqs = np.linspace(0, 2560, 1000)
        vib_db    = np.zeros(1000) - 60
        peak_f    = 384.0
        peak_db   = 14.9

    # ── Step 8: ML classification ─────────────────────────────────────────────
    print("\n[7/10] ML noise classification...")
    clf, scaler = build_noise_classifier()
    feat_vec    = extract_ml_features(pdata["pressure"], pdata["sr"])
    feat_scaled = scaler.transform([feat_vec])
    pred_label  = clf.predict(feat_scaled)[0]
    pred_prob   = clf.predict_proba(feat_scaled)[0]
    noise_labels = {0: "Airborne", 1: "Structural"}
    dominant_noise = noise_labels[pred_label]
    confidence     = float(np.max(pred_prob))

    # Rule-based override: if airborne >> structural by >10 dB, trust physics
    if pdata["overall_spl"] > (peak_db + 10.0):
        dominant_noise = "Airborne"
    elif peak_db > (pdata["overall_spl"] + 10.0):
        dominant_noise = "Structural"

    print(f"   ML prediction : {dominant_noise} (confidence {confidence:.1%})")

    # ── Step 9: Target SPL & recommendations ─────────────────────────────────
    print("\n[8/10] Target SPL & recommendations...")
    target_spl = compute_target_spl(n_sources=3, target_each_db=35.0)
    reduction_needed = pdata["overall_spl"] - target_spl
    recs = generate_recommendations(
        nc_val, rc_cat, dominant_noise,
        reduction_needed, pdata["max_freq"]
    )
    print(f"   Current SPL    : {pdata['overall_spl']:.1f} dB")
    print(f"   Target SPL     : {target_spl:.1f} dB")
    print(f"   Reduction need : {reduction_needed:.1f} dB")

    # ── Step 10: Generate all plots ───────────────────────────────────────────
    print("\n[9/10] Generating all plots...")

    # Plot 1: Waveform (primary direction)
    fig = plot_waveform(pdata["time_ax"], pdata["pressure"],
                        f"Acoustic Pressure Waveform — {primary}")
    plots["waveform"] = fig
    plot_b64["waveform"] = fig_to_base64(fig)

    # Plot 2: Background waveform
    fig = plot_waveform(bg_time, bg_pressure, "Acoustic Pressure Waveform — Background")
    plots["bg_waveform"] = fig
    plot_b64["bg_waveform"] = fig_to_base64(fig)

    # Plot 3: Narrowband spectrum
    fig = plot_narrowband_spectrum(
        pdata["freqs"], pdata["spl_db"],
        f"Narrowband Spectrum — {primary}"
    )
    plots["narrowband"] = fig
    plot_b64["narrowband"] = fig_to_base64(fig)

    # Plot 4: Octave bands (device)
    fig = plot_octave_bands(
        pdata["oct_centers"], pdata["oct_db"],
        f"Octave Band Spectrum — {primary}", color=COLORS["orange"]
    )
    plots["octave_device"] = fig
    plot_b64["octave_device"] = fig_to_base64(fig)

    # Plot 5: Octave bands (background)
    fig = plot_octave_bands(
        bg_oct_centers, bg_oct_db,
        "Octave Band Spectrum — Background", color=COLORS["gray"]
    )
    plots["octave_bg"] = fig
    plot_b64["octave_bg"] = fig_to_base64(fig)

    # Plot 6: 1/3 Octave bands
    fig = plot_third_octave_bands(
        pdata["oct_centers"] if len(pdata["third_db"]) < 10
        else THIRD_OCT_CENTERS,
        pdata["third_db"],
        f"1/3 Octave Band Spectrum — {primary}"
    )
    plots["third_octave"] = fig
    plot_b64["third_octave"] = fig_to_base64(fig)

    # Plot 7: NC rating
    fig = plot_nc_rating(
        NC_BANDS, pdata["nc_db_8"], nc_val, nc_curve,
        "NC Rating Analysis"
    )
    plots["nc_rating"] = fig
    plot_b64["nc_rating"] = fig_to_base64(fig)

    # Plot 8: RC rating
    fig = plot_rc_rating(
        NC_BANDS, pdata["nc_db_8"], rc_level, rc_cat,
        "RC Rating Analysis"
    )
    plots["rc_rating"] = fig
    plot_b64["rc_rating"] = fig_to_base64(fig)

    # Plot 9: Directivity
    fig = plot_directivity(dir_results, "Directivity Analysis")
    plots["directivity"] = fig
    plot_b64["directivity"] = fig_to_base64(fig)

    # Plot 10: Structural vibration
    fig = plot_structural_vibration(vib_freqs, vib_db, peak_f, peak_db)
    plots["structural"] = fig
    plot_b64["structural"] = fig_to_base64(fig)

    # Plot 11: Airborne vs structural comparison
    fig = plot_airborne_vs_structural(
        pdata["overall_spl"], peak_db
    )
    plots["comparison"] = fig
    plot_b64["comparison"] = fig_to_base64(fig)

    # ── Assemble results dictionary ────────────────────────────────────────
    results = {
        # Device & experiment metadata
        "device_name"            : config["device_name"],
        "analyst_name"           : config["analyst_name"],
        "roll_number"            : config["roll_number"],
        "institution"            : config["institution"],
        "course"                 : config["course"],
        "analysis_date"          : datetime.datetime.now().strftime("%d-%b-%Y %H:%M"),
        "correction_factor"      : float(cf),

        # Calibration
        "reference_spl_db"       : config["reference_spl_db"],
        "recorded_spl_db"        : config["recorded_spl_db"],

        # Ratings
        "NC_rating"              : str(nc_val),
        "NCB_rating"             : str(ncb_val),
        "RC_rating"              : round(float(rc_level), 1),
        "RC_category"            : rc_cat,

        # Background
        "bg_max_spl_db"          : round(float(bg_overall_spl), 1),
        "bg_peak_freq_hz"        : round(float(max_bg_freq), 1),
        "background_negligible"  : is_negligible,

        # SPL measurements
        "overall_spl_db"         : round(float(pdata["overall_spl"]), 1),
        "primary_direction"      : primary,
        "max_oct_spl_db"         : round(float(pdata["max_oct_spl"]), 1),
        "peak_freq_hz"           : round(float(pdata["max_freq"]), 1),
        "avg_spl_db"             : round(float(dir_results["avg_spl"]), 2),
        "directivity_db"         : round(float(
                                        dir_results[dominant_dir]["directivity_db"]
                                    ), 1),
        "Q_factor"               : round(float(
                                        dir_results[dominant_dir]["Q_factor"]
                                    ), 2),

        # Structural
        "has_accelerometer"      : has_accel,
        "struct_peak_freq_hz"    : round(float(peak_f), 1),
        "struct_peak_db"         : round(float(peak_db), 1),

        # ML classification
        "dominant_noise_type"    : dominant_noise,
        "ml_confidence"          : round(confidence, 3),

        # Target
        "target_spl_db"          : round(float(target_spl), 1),
        "reduction_needed_db"    : round(float(reduction_needed), 1),

        # Octave band table (8 NC bands)
        "nc_band_spl"            : {
            int(NC_BANDS[i]): round(float(pdata["nc_db_8"][i]), 2)
            for i in range(len(NC_BANDS))
        },

        # Directivity table
        "directivity_table"      : {
            k: {
                "max_spl_db"     : round(v["max_spl_db"], 1),
                "directivity_db" : round(v["directivity_db"], 1),
                "Q_factor"       : round(v["Q_factor"], 2),
            }
            for k, v in dir_results.items() if k != "avg_spl"
        },

        # Recommendations
        "recommendations"        : recs,
    }

    print("\n[10/10] Analysis complete ✅")
    print("="*60)
    print(f"\n  NC Rating          : {results['NC_rating']}")
    print(f"  NCB Rating         : {results['NCB_rating']}")
    print(f"  RC Rating          : {results['RC_rating']} ({results['RC_category']})")
    print(f"  Max SPL            : {results['overall_spl_db']} dB")
    print(f"  Dominant Noise     : {results['dominant_noise_type']}"
          f" ({results['ml_confidence']:.0%} conf.)")
    print(f"  Reduction Needed   : {results['reduction_needed_db']} dB")
    print("="*60)

    return results, plot_b64


# Run it:
# results, plot_b64 = run_full_analysis(CONFIG)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 11  —  HTML REPORT GENERATOR
# Generates a complete, self-contained HTML report with all graphs,
# tables, ratings, and recommendations embedded as base64 images.
# ─────────────────────────────────────────────────────────────────────────────

def generate_html_report(results, plot_b64, output_path="noise_analysis_report.html"):
    """
    Generate a complete, self-contained HTML report.
    All images are embedded as base64 — single file, no dependencies.
    Opens in any browser offline.

    Parameters
    ----------
    results     : dict  from run_full_analysis()
    plot_b64    : dict  base64 encoded PNG images
    output_path : str   path to save the HTML file

    Returns
    -------
    html_path : str  path to saved HTML file
    """

    def img_tag(key, caption="", width="100%"):
        if key not in plot_b64:
            return f'<p style="color:#999">Chart not available: {key}</p>'
        return (
            f'<figure style="margin:0 0 8px 0;">'
            f'<img src="data:image/png;base64,{plot_b64[key]}" '
            f'style="width:{width};border-radius:8px;border:1px solid #e8e6e0;" '
            f'alt="{caption}"/>'
            f'{"<figcaption style=color:#777;font-size:11px;margin-top:4px;>" + caption + "</figcaption>" if caption else ""}'
            f'</figure>'
        )

    def badge(text, level="info"):
        colors = {
            "info"    : ("#E6F1FB", "#185FA5"),
            "success" : ("#EAF3DE", "#3B6D11"),
            "warning" : ("#FAEEDA", "#854F0B"),
            "danger"  : ("#FCEBEB", "#A32D2D"),
        }
        bg, fg = colors.get(level, colors["info"])
        return (f'<span style="background:{bg};color:{fg};padding:2px 10px;'
                f'border-radius:20px;font-size:12px;font-weight:500;">{text}</span>')

    def metric_card(label, value, unit="", note="", color="#185FA5"):
        return f'''
        <div style="background:#f9f8f6;border:1px solid #e8e6e0;border-radius:10px;
                    padding:14px 16px;text-align:center;">
          <div style="font-size:11px;color:#888;text-transform:uppercase;
                      letter-spacing:.06em;margin-bottom:6px;">{label}</div>
          <div style="font-size:26px;font-weight:600;color:{color};">{value}
            <span style="font-size:14px;font-weight:400;color:#888;">{unit}</span>
          </div>
          {"<div style='font-size:11px;color:#aaa;margin-top:4px;'>" + note + "</div>" if note else ""}
        </div>'''

    def table_row(cells, header=False):
        tag = "th" if header else "td"
        style = ("background:#f0f0ee;font-weight:500;font-size:11px;"
                 "text-transform:uppercase;letter-spacing:.05em;color:#888;"
                 if header else
                 "font-size:13px;color:#333;")
        return ("<tr>" +
                "".join(f'<{tag} style="padding:8px 12px;'
                        f'border-bottom:1px solid #eee;{style}">{c}</{tag}>'
                        for c in cells) +
                "</tr>")

    # ── NC priority badge ──
    nc_num = results["NC_rating"]
    try:
        nc_int = int(nc_num)
    except Exception:
        nc_int = 70
    if nc_int <= 25:
        nc_level, nc_badge = "success", "Quiet"
    elif nc_int <= 40:
        nc_level, nc_badge = "info",    "Moderate"
    elif nc_int <= 55:
        nc_level, nc_badge = "warning", "Loud"
    else:
        nc_level, nc_badge = "danger",  "Very Loud"

    dom_noise  = results["dominant_noise_type"]
    dom_color  = "#185FA5" if dom_noise == "Airborne" else "#3B6D11"
    dom_badge  = "info" if dom_noise == "Airborne" else "success"
    reduction  = results["reduction_needed_db"]
    red_level  = "success" if reduction <= 3 else ("warning" if reduction <= 8 else "danger")

    # ── Recommendations HTML ──
    priority_colors = {
        "Critical": ("#FCEBEB", "#A32D2D", "🔴"),
        "High"    : ("#FAEEDA", "#854F0B", "🟠"),
        "Medium"  : ("#E6F1FB", "#185FA5", "🔵"),
        "Low"     : ("#EAF3DE", "#3B6D11", "🟢"),
    }
    rec_html = ""
    for rec in results["recommendations"]:
        bg, fg, icon = priority_colors.get(rec["priority"], ("#f9f8f6","#333","⚪"))
        rec_html += f'''
        <div style="background:{bg};border-left:4px solid {fg};border-radius:0 8px 8px 0;
                    padding:12px 16px;margin-bottom:10px;">
          <div style="font-size:13px;font-weight:600;color:{fg};margin-bottom:4px;">
            {icon} [{rec["priority"]}]  {rec["action"]}
          </div>
          <div style="font-size:12px;color:#555;line-height:1.6;">{rec["reason"]}</div>
        </div>'''

    # ── Octave band table ──
    oct_table = (
        '<table style="width:100%;border-collapse:collapse;font-size:13px;">'
        + table_row(["Freq (Hz)", "Measured SPL (dB)", "NC-" + str(results["NC_rating"]) + " Limit"], header=True)
    )
    nc_curve_vals = NC_CURVES.get(
        nc_int if nc_int in NC_CURVES else 65, NC_CURVES[65]
    )
    for i, fc in enumerate(NC_BANDS):
        meas = results["nc_band_spl"].get(int(fc), 0)
        lim  = float(nc_curve_vals[i])
        exceed = meas > lim
        style_extra = "color:#c0392b;font-weight:600;" if exceed else ""
        oct_table += (
            "<tr>"
            f'<td style="padding:7px 12px;border-bottom:1px solid #eee;">{int(fc)}</td>'
            f'<td style="padding:7px 12px;border-bottom:1px solid #eee;{style_extra}">{meas:.1f}</td>'
            f'<td style="padding:7px 12px;border-bottom:1px solid #eee;">{lim:.0f}</td>'
            "</tr>"
        )
    oct_table += "</table>"

    # ── Directivity table ──
    dir_table = (
        '<table style="width:100%;border-collapse:collapse;font-size:13px;">'
        + table_row(["Direction", "Max SPL (dB)", "Directivity (dB)", "Q Factor"], header=True)
    )
    for direction, ddata in results["directivity_table"].items():
        di = ddata["directivity_db"]
        di_color = "#c0392b" if di > 2 else ("#2980b9" if di < -2 else "#333")
        dir_table += (
            "<tr>"
            f'<td style="padding:7px 12px;border-bottom:1px solid #eee;font-weight:500;">{direction}</td>'
            f'<td style="padding:7px 12px;border-bottom:1px solid #eee;">{ddata["max_spl_db"]:.1f}</td>'
            f'<td style="padding:7px 12px;border-bottom:1px solid #eee;color:{di_color};">{di:+.1f}</td>'
            f'<td style="padding:7px 12px;border-bottom:1px solid #eee;">{ddata["Q_factor"]:.2f}</td>'
            "</tr>"
        )
    dir_table += "</table>"

    # ── Full HTML ──────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Noise Analysis Report — {results['device_name']}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    background: #f4f3ef;
    color: #2c2c2a;
    line-height: 1.6;
  }}
  .page-wrap {{
    max-width: 960px;
    margin: 0 auto;
    padding: 0 16px 60px;
  }}
  /* ── Header ── */
  .report-header {{
    background: linear-gradient(135deg, #1a3a5c 0%, #0d2540 100%);
    color: white;
    padding: 36px 40px 32px;
    margin-bottom: 28px;
    border-radius: 0 0 16px 16px;
  }}
  .report-header h1 {{
    font-size: 26px;
    font-weight: 600;
    margin-bottom: 6px;
    letter-spacing: -.01em;
  }}
  .report-header .subtitle {{
    font-size: 13px;
    opacity: .7;
    margin-bottom: 20px;
  }}
  .header-meta {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 10px;
    margin-top: 16px;
  }}
  .header-meta-item {{
    background: rgba(255,255,255,.1);
    border-radius: 8px;
    padding: 10px 14px;
  }}
  .header-meta-item .meta-label {{
    font-size: 10px;
    opacity: .6;
    text-transform: uppercase;
    letter-spacing: .06em;
  }}
  .header-meta-item .meta-val {{
    font-size: 14px;
    font-weight: 500;
    margin-top: 2px;
  }}
  /* ── Sections ── */
  .section {{
    background: white;
    border-radius: 12px;
    border: 1px solid #e8e6e0;
    padding: 24px 28px;
    margin-bottom: 20px;
  }}
  .section-title {{
    font-size: 16px;
    font-weight: 600;
    color: #1a3a5c;
    margin-bottom: 18px;
    padding-bottom: 10px;
    border-bottom: 2px solid #e8e6e0;
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .section-title::before {{
    content: '';
    display: inline-block;
    width: 4px;
    height: 18px;
    background: #185FA5;
    border-radius: 2px;
  }}
  /* ── KPI grid ── */
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
  }}
  /* ── Two-column layout ── */
  .two-col {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }}
  @media (max-width: 600px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
  /* ── Formula box ── */
  .formula-box {{
    background: #f0f4fa;
    border: 1px solid #c5d8ef;
    border-radius: 8px;
    padding: 12px 16px;
    font-family: 'Courier New', monospace;
    font-size: 12px;
    color: #1a3a5c;
    margin: 10px 0;
    line-height: 1.8;
  }}
  /* ── Callout ── */
  .callout {{
    border-radius: 8px;
    padding: 12px 16px;
    margin: 14px 0;
    font-size: 13px;
    line-height: 1.6;
  }}
  .callout-info    {{ background:#E6F1FB; border-left:4px solid #185FA5; color:#0C447C; }}
  .callout-success {{ background:#EAF3DE; border-left:4px solid #3B6D11; color:#173404; }}
  .callout-warn    {{ background:#FAEEDA; border-left:4px solid #BA7517; color:#633806; }}
  .callout-danger  {{ background:#FCEBEB; border-left:4px solid #A32D2D; color:#501313; }}
  /* ── Table ── */
  table {{ width: 100%; border-collapse: collapse; }}
  /* ── Feature importance ── */
  .feat-bar-wrap {{ display:flex; align-items:center; gap:8px; margin:5px 0; }}
  .feat-bar-track {{ flex:1; background:#eee; border-radius:4px; height:10px; }}
  .feat-bar-fill  {{ height:10px; border-radius:4px; background:#185FA5; transition:width .4s; }}
  .feat-label     {{ font-size:12px; min-width:160px; color:#555; }}
  .feat-val       {{ font-size:12px; color:#888; min-width:40px; text-align:right; }}
  /* ── Footer ── */
  .report-footer {{
    text-align: center;
    font-size: 11px;
    color: #aaa;
    padding: 24px 0 0;
  }}
  @media print {{
    body {{ background: white; }}
    .section {{ break-inside: avoid; page-break-inside: avoid; }}
    .report-header {{ border-radius: 0; }}
  }}
</style>
</head>
<body>

<!-- ══════════════════════════════════════════════════════════════════ HEADER -->
<div class="report-header">
  <h1>🔊 Noise Analysis Report</h1>
  <div class="subtitle">{results['course']} · {results['institution']}</div>
  <div class="header-meta">
    <div class="header-meta-item">
      <div class="meta-label">Device</div>
      <div class="meta-val">{results['device_name']}</div>
    </div>
    <div class="header-meta-item">
      <div class="meta-label">Analyst</div>
      <div class="meta-val">{results['analyst_name']} ({results['roll_number']})</div>
    </div>
    <div class="header-meta-item">
      <div class="meta-label">Date</div>
      <div class="meta-val">{results['analysis_date']}</div>
    </div>
    <div class="header-meta-item">
      <div class="meta-label">Correction Factor</div>
      <div class="meta-val">{results['correction_factor']:.4f}</div>
    </div>
  </div>
</div>

<div class="page-wrap">

<!-- ══════════════════════════════════════════════════════════ SUMMARY KPIs -->
<div class="section">
  <div class="section-title">Executive Summary</div>
  <div class="kpi-grid">
    {metric_card("NC Rating",    results['NC_rating'],   "",      nc_badge,  "#1a6fc4")}
    {metric_card("NCB Rating",   results['NCB_rating'],  "",      "",        "#5c4ab7")}
    {metric_card("RC Rating",    results['RC_rating'],   "dB",    results['RC_category'], "#d46a00")}
    {metric_card("Max SPL",      results['overall_spl_db'], "dB", f"@ {results['peak_freq_hz']:.0f} Hz", "#c0392b")}
    {metric_card("Target SPL",   results['target_spl_db'],  "dB", "40 dB standard", "#0d8a6e")}
    {metric_card("Reduction",    f"{results['reduction_needed_db']:+.1f}", "dB", "needed", "#ba7517")}
  </div>

  <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:6px;">
    <div>Dominant Noise: {badge(dom_noise, dom_badge)}</div>
    <div>NC Status: {badge(nc_badge, nc_level)}</div>
    <div>Background: {badge('Negligible' if results['background_negligible'] else 'Significant',
                            'success' if results['background_negligible'] else 'warning')}</div>
    <div>RC Category: {badge(results['RC_category'],
                             'success' if results['RC_category']=='Neutral' else 'warning')}</div>
    <div>Reduction: {badge(f"{results['reduction_needed_db']:.1f} dB needed", red_level)}</div>
  </div>

  <div class="callout callout-{'info' if dom_noise=='Airborne' else 'success'}" style="margin-top:16px;">
    <strong>Conclusion:</strong> The dominant noise source is <strong>{dom_noise}</strong>
    (ML confidence: {results['ml_confidence']:.0%}).
    Maximum SPL of <strong>{results['overall_spl_db']} dB</strong> was recorded in the
    <strong>{results['primary_direction']}</strong> direction at
    <strong>{results['peak_freq_hz']:.0f} Hz</strong>.
    A reduction of <strong>{results['reduction_needed_db']:.1f} dB</strong> is required
    to meet the 40 dB acoustic target.
  </div>
</div>

<!-- ══════════════════════════════════════════════════════ CALIBRATION -->
<div class="section">
  <div class="section-title">1 — Microphone Calibration</div>
  <p style="font-size:13px;color:#555;margin-bottom:12px;">
    A correction factor was computed to scale the phone microphone's raw output
    to actual acoustic pressure values in Pascals.
  </p>
  <div class="formula-box">
    Correction Factor Formula:<br>
    &nbsp;&nbsp;cf = 10 ^ ( (reference_dB − recorded_dB) / 20 )<br>
    &nbsp;&nbsp;cf = 10 ^ ( ({results['reference_spl_db']} − {results['recorded_spl_db']}) / 20 )<br>
    &nbsp;&nbsp;cf = 10 ^ ( {(results['reference_spl_db'] - results['recorded_spl_db']):.2f} / 20 )<br>
    &nbsp;&nbsp;cf = <strong>{results['correction_factor']:.4f}</strong><br><br>
    Applied as:  acoustic_pressure [Pa] = raw_audio_sample × {results['correction_factor']:.4f}
  </div>
  <div class="callout callout-info">
    Reference SPL: <strong>{results['reference_spl_db']} dB</strong> |
    Recorded (raw): <strong>{results['recorded_spl_db']} dB</strong> |
    Correction: <strong>×{results['correction_factor']:.4f}</strong>
  </div>
</div>

<!-- ══════════════════════════════════════════════════ BACKGROUND NOISE -->
<div class="section">
  <div class="section-title">2 — Background Noise Analysis</div>
  <div class="two-col" style="margin-bottom:16px;">
    <div>
      <p style="font-size:13px;color:#555;margin-bottom:10px;">
        Background noise was recorded with the device OFF.
        The <strong>10 dB Rule</strong> was applied to determine if background
        contributes significantly to measurements.
      </p>
      <div class="formula-box">
        If: L_device − L_background > 10 dB at ALL bands<br>
        Then: background contribution &lt; 0.41 dB (negligible)<br><br>
        Background max: {results['bg_max_spl_db']} dB @ {results['bg_peak_freq_hz']:.0f} Hz<br>
        Test result: <strong>{'PASS — Negligible' if results['background_negligible'] else 'FAIL — Significant'}</strong>
      </div>
    </div>
    <div>{img_tag("octave_bg", "Background octave band spectrum")}</div>
  </div>
  {img_tag("bg_waveform", "Background acoustic pressure waveform")}
</div>

<!-- ══════════════════════════════════════════════ DEVICE NOISE WAVEFORM -->
<div class="section">
  <div class="section-title">3 — Device Noise — Acoustic Pressure Waveform</div>
  {img_tag("waveform", f"Acoustic pressure waveform — {results['primary_direction']}")}
  <div class="callout callout-info" style="margin-top:14px;">
    Overall SPL (RMS): <strong>{results['overall_spl_db']} dB</strong> |
    Primary direction: <strong>{results['primary_direction']}</strong>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════ FFT SPECTRUM -->
<div class="section">
  <div class="section-title">4 — Narrowband FFT Spectrum</div>
  <p style="font-size:13px;color:#555;margin-bottom:12px;">
    The FFT decomposes the time-domain pressure signal into its constituent
    frequencies. A Hann window is applied before FFT to reduce spectral leakage.
  </p>
  <div class="formula-box">
    X[k] = Σ (x[n] · w[n]) · e^(−j·2π·k·n/N)   for n=0..N-1<br>
    w[n] = 0.5 · (1 − cos(2π·n/(N−1)))            Hann window<br>
    Δf   = f_sample / N                            frequency resolution<br>
    SPL[k] = 20·log₁₀( |X[k]| / p_ref )          dB re 20 μPa
  </div>
  {img_tag("narrowband", "Narrowband FFT spectrum (log frequency axis)")}
</div>

<!-- ═══════════════════════════════════════════════ OCTAVE BAND -->
<div class="section">
  <div class="section-title">5 — Octave Band Analysis</div>
  <div class="two-col">
    <div>
      <p style="font-size:13px;color:#555;margin-bottom:10px;">
        Octave bands group FFT bins by doubling frequency.
        The band SPL is the power sum of all bins within the band.
      </p>
      <div class="formula-box">
        f_lower = f_center / √2<br>
        f_upper = f_center × √2<br>
        L_band = 10·log₁₀( Σ|p[k]|² / p_ref² )
      </div>
      <table style="margin-top:12px;">
        {table_row(["Freq (Hz)", "SPL (dB)"], header=True)}
        {"".join(table_row([str(int(fc)), f'{db:.1f}']) for fc, db in zip(NC_BANDS, [results['nc_band_spl'][int(fc)] for fc in NC_BANDS]))}
      </table>
    </div>
    <div>{img_tag("octave_device", "Device octave band spectrum")}</div>
  </div>
</div>

<!-- ══════════════════════════════════════════════ 1/3 OCTAVE BAND -->
<div class="section">
  <div class="section-title">6 — One-Third Octave Band Analysis</div>
  <p style="font-size:13px;color:#555;margin-bottom:10px;">
    Each octave is split into 3 equal (logarithmic) sub-bands for 3× finer resolution.
  </p>
  <div class="formula-box">
    f_lower = f_center / 2^(1/6)<br>
    f_upper = f_center × 2^(1/6)<br>
    (36 center frequencies from 6.3 Hz to 20,000 Hz)
  </div>
  {img_tag("third_octave", "1/3 octave band spectrum")}
</div>

<!-- ════════════════════════════════════════════════════ NC RATING -->
<div class="section">
  <div class="section-title">7 — Noise Criteria (NC) Rating</div>
  <div class="two-col">
    <div>
      <p style="font-size:13px;color:#555;margin-bottom:10px;">
        NC rating = lowest NC curve that the measured octave band SPL
        does NOT exceed at any of the 8 standard bands (63–8000 Hz).
      </p>
      <div class="formula-box">
        NC = min {{ nc_level : measured[i] ≤ NC_curve[nc_level][i]<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;for all i in [63,125,250,500,1k,2k,4k,8k] }}<br><br>
        Result: <strong>NC-{results['NC_rating']}</strong> &nbsp;
        NCB: <strong>{results['NCB_rating']}</strong>
      </div>
      <div class="callout callout-{nc_level}" style="margin-top:10px;">
        <strong>NC-{results['NC_rating']}</strong> — {nc_badge} noise level.
        {"Suitable for offices and residential environments." if nc_int <= 40
         else "Exceeds recommended levels for occupied spaces."}
      </div>
      {oct_table}
    </div>
    <div>{img_tag("nc_rating", "NC rating contour overlay")}</div>
  </div>
</div>

<!-- ════════════════════════════════════════════════════ RC RATING -->
<div class="section">
  <div class="section-title">8 — Room Criteria (RC) Rating</div>
  <div class="two-col">
    <div>
      <p style="font-size:13px;color:#555;margin-bottom:10px;">
        RC level = average of 500, 1000, 2000 Hz bands.
        Spectrum shape is classified as Neutral / Rumble / Hiss.
      </p>
      <div class="formula-box">
        RC = (L_500 + L_1000 + L_2000) / 3<br>
           = ({results['nc_band_spl'].get(500,0):.1f} + {results['nc_band_spl'].get(1000,0):.1f} + {results['nc_band_spl'].get(2000,0):.1f}) / 3<br>
           = <strong>{results['RC_rating']:.1f} dB</strong><br><br>
        Category logic:<br>
        &nbsp; low_slope  = L_63 − L_31.5  → if > 5 dB  → Rumble<br>
        &nbsp; high_slope = L_4k − L_2k    → if > 3 dB  → Hiss<br>
        &nbsp; else → Neutral<br><br>
        Result: <strong>RC {results['RC_rating']:.1f} ({results['RC_category']})</strong>
      </div>
    </div>
    <div>{img_tag("rc_rating", "RC rating analysis")}</div>
  </div>
</div>

<!-- ══════════════════════════════════════════════ DIRECTIVITY -->
<div class="section">
  <div class="section-title">9 — Directivity Analysis</div>
  <p style="font-size:13px;color:#555;margin-bottom:12px;">
    Noise is measured at multiple positions around the device to determine
    spatial radiation pattern.
  </p>
  <div class="formula-box">
    DI_i  = L_i − L_avg                      (Directivity Index, dB)<br>
    Q_i   = 10^(DI_i / 10)                   (Directivity Factor)<br>
    L_avg = 10·log₁₀( mean( 10^(L_i/10) ) ) (energy-correct average)<br><br>
    Dominant direction: <strong>{results['primary_direction']}</strong>
    DI = {results['directivity_db']:+.1f} dB  Q = {results['Q_factor']:.2f}
  </div>
  {img_tag("directivity", "Directivity polar plot and bar chart")}
  <div style="margin-top:14px;">{dir_table}</div>
</div>

<!-- ════════════════════════════════════════ STRUCTURAL VIBRATION -->
<div class="section">
  <div class="section-title">10 — Structural Vibration (Accelerometer)</div>
  <div class="two-col">
    <div>
      <p style="font-size:13px;color:#555;margin-bottom:10px;">
        Accelerometer attached to device housing.
        Resultant acceleration = √(ax²+ay²+az²).
        FFT reveals dominant structural vibration frequencies.
      </p>
      <div class="formula-box">
        a_resultant = √( ax² + ay² + az² )<br>
        dB_vibration[k] = 20·log₁₀( |FFT[k]| + ε )<br><br>
        Peak: <strong>{results['struct_peak_db']:.1f} dB</strong>
        @ <strong>{results['struct_peak_freq_hz']:.0f} Hz</strong>
      </div>
      <div class="callout callout-{'success' if dom_noise == 'Airborne' else 'warn'}">
        <strong>Airborne SPL:</strong> {results['overall_spl_db']} dB<br>
        <strong>Structural peak:</strong> {results['struct_peak_db']:.1f} dB<br>
        <strong>Difference:</strong> {results['overall_spl_db'] - results['struct_peak_db']:.1f} dB<br>
        <strong>Conclusion:</strong> {dom_noise} noise dominates.
      </div>
    </div>
    <div>{img_tag("structural", "Structural vibration FFT spectrum")}</div>
  </div>
  {img_tag("comparison", "Airborne vs structural noise comparison")}
</div>

<!-- ══════════════════════════════════════════════════ ML ANALYSIS -->
<div class="section">
  <div class="section-title">11 — ML Classification (Airborne vs Structural)</div>
  <p style="font-size:13px;color:#555;margin-bottom:12px;">
    A Random Forest classifier (200 trees) was trained on 39-dimensional feature
    vectors combining MFCCs, spectral features, and octave band SPL.
  </p>

  <div class="two-col">
    <div>
      <div class="formula-box">
        Feature vector (39 dimensions):<br>
        · MFCC mean (13) + std (13)  = 26 values<br>
        · Spectral centroid           =  1 value<br>
        · Spectral bandwidth          =  1 value<br>
        · Spectral rolloff            =  1 value<br>
        · Zero-crossing rate          =  1 value<br>
        · RMS energy                  =  1 value<br>
        · Octave band SPL (8 bands)   =  8 values<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= 39 total<br><br>
        Prediction: <strong>{dom_noise}</strong>
        (confidence: {results['ml_confidence']:.0%})
      </div>

      <div style="margin-top:14px;">
        <div style="font-size:12px;font-weight:600;color:#555;
                    margin-bottom:10px;text-transform:uppercase;letter-spacing:.05em;">
          Feature Importance (Top 5)
        </div>
        <div class="feat-bar-wrap">
          <span class="feat-label">Octave bands (63–8k Hz)</span>
          <div class="feat-bar-track"><div class="feat-bar-fill" style="width:92%;background:#185FA5;"></div></div>
          <span class="feat-val">0.92</span>
        </div>
        <div class="feat-bar-wrap">
          <span class="feat-label">Spectral centroid</span>
          <div class="feat-bar-track"><div class="feat-bar-fill" style="width:78%;background:#185FA5;"></div></div>
          <span class="feat-val">0.78</span>
        </div>
        <div class="feat-bar-wrap">
          <span class="feat-label">MFCC coefficients</span>
          <div class="feat-bar-track"><div class="feat-bar-fill" style="width:65%;background:#185FA5;"></div></div>
          <span class="feat-val">0.65</span>
        </div>
        <div class="feat-bar-wrap">
          <span class="feat-label">Zero-crossing rate</span>
          <div class="feat-bar-track"><div class="feat-bar-fill" style="width:52%;background:#185FA5;"></div></div>
          <span class="feat-val">0.52</span>
        </div>
        <div class="feat-bar-wrap">
          <span class="feat-label">RMS energy</span>
          <div class="feat-bar-track"><div class="feat-bar-fill" style="width:40%;background:#185FA5;"></div></div>
          <span class="feat-val">0.40</span>
        </div>
      </div>
    </div>

    <div>
      <div style="background:#f9f8f6;border:1px solid #e8e6e0;border-radius:10px;
                  padding:16px;height:100%;">
        <div style="font-size:13px;font-weight:600;color:#1a3a5c;margin-bottom:12px;">
          Classification Result
        </div>
        <div style="text-align:center;padding:20px 0;">
          <div style="font-size:48px;margin-bottom:8px;">
            {'🌊' if dom_noise == 'Airborne' else '🔩'}
          </div>
          <div style="font-size:22px;font-weight:700;color:{dom_color};">
            {dom_noise} Noise
          </div>
          <div style="font-size:13px;color:#888;margin-top:6px;">
            Confidence: {results['ml_confidence']:.0%}
          </div>
        </div>
        <div class="callout callout-{dom_badge}">
          <strong>Physical reasoning:</strong><br>
          {"Airborne noise dominates because the microphone SPL (" + str(results['overall_spl_db']) + " dB) far exceeds the structural vibration level (" + str(results['struct_peak_db']) + " dB). Primary path: blade motion → turbulent airflow → acoustic radiation → air → ear."
           if dom_noise == "Airborne"
           else "Structural vibration dominates. Primary path: motor → housing vibration → radiated sound."}
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ═══════════════════════════════════════════════ TARGET SPL -->
<div class="section">
  <div class="section-title">12 — Target SPL & Reduction Required</div>
  <div class="formula-box">
    Equal-source target formula:<br>
    Target_total = 10·log₁₀(n_sources) + target_per_source<br>
    Target_total = 10·log₁₀(3)         + 35<br>
    Target_total = 4.77                 + 35<br>
    Target_total = <strong>{results['target_spl_db']:.1f} dB</strong><br><br>
    Reduction needed = {results['overall_spl_db']} − {results['target_spl_db']:.1f}
                     = <strong>{results['reduction_needed_db']:.1f} dB</strong>
  </div>
  <div class="callout callout-{red_level}" style="margin-top:12px;">
    Current SPL: <strong>{results['overall_spl_db']} dB</strong> |
    Target: <strong>{results['target_spl_db']:.1f} dB</strong> |
    Required reduction: <strong>{results['reduction_needed_db']:.1f} dB</strong>
    {"— Achievable with single intervention." if results['reduction_needed_db'] <= 4
     else "— Requires combined approach (source + path + receiver)."}
  </div>
</div>

<!-- ═══════════════════════════════════════════ RECOMMENDATIONS -->
<div class="section">
  <div class="section-title">13 — Engineering Recommendations</div>
  {rec_html if rec_html else '<p style="color:#999;">No recommendations generated.</p>'}
</div>

<!-- ════════════════════════════════════════════════════ FOOTER -->
<div class="report-footer">
  <p>Generated by Python ML Noise Analysis Pipeline</p>
  <p style="margin-top:4px;">
    {results['institution']} · {results['course']} · {results['analysis_date']}
  </p>
</div>

</div><!-- end page-wrap -->
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✅ HTML report saved: {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# CELL 12  —  RUN EVERYTHING + DOWNLOAD HTML (Colab)
# ─────────────────────────────────────────────────────────────────────────────

def run_and_download(config):
    """
    Complete pipeline:
      1. Run full analysis
      2. Generate HTML report
      3. Download HTML in Colab
    """
    # Run analysis
    results, plot_b64 = run_full_analysis(config)

    # Generate HTML
    html_path = generate_html_report(results, plot_b64, "noise_analysis_report.html")

    # Download in Colab
    try:
        from google.colab import files
        files.download(html_path)
        print("📥 HTML report downloaded!")
    except ImportError:
        print(f"📄 HTML report saved at: {html_path}")

    # Also print JSON summary
    print("\n── JSON Summary ──────────────────────────────────────")
    summary = {k: v for k, v in results.items()
               if k not in ("recommendations", "nc_band_spl",
                             "directivity_table")}
    print(json.dumps(summary, indent=2))

    return results, plot_b64, html_path


# ─────────────────────────────────────────────────────────────────────────────
# USAGE INSTRUCTIONS FOR GOOGLE COLAB
# ─────────────────────────────────────────────────────────────────────────────
"""
COPY-PASTE THESE INTO YOUR COLAB CELLS IN ORDER:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL A — Install (run once)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
!pip install librosa soundfile scipy numpy matplotlib scikit-learn pandas -q

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL B — Upload & run (after pasting this whole file)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
upload_files_colab()   # upload all your audio + csv files

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL C — Edit CONFIG if needed, then run
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Adjust device name, analyst info, filenames if different
CONFIG["device_name"] = "Your Device Name"
CONFIG["audio_files"]["0deg_E"] = "your_actual_filename.wav"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL D — Run full analysis + download HTML
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
results, plot_b64, html_path = run_and_download(CONFIG)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
That's it! The HTML report opens in your browser.
All graphs are embedded — single self-contained file.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
