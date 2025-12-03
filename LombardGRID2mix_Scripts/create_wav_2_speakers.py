"""
Внимание! Скрипт неточный, и не возвращает результат, идентичный
скрипту create_wav_2speakers.m. Но на слух достаточно похоже.
-----
Attention! The script is inaccurate and does not return a result identical to the
create_wav_2speakers.m script. But it sounds quite similar.
"""

"""
create_wav_2_speakers.py

Python port of the MATLAB script create_wav_2_speakers.m

"""

import os
import sys
import ctypes
from ctypes import c_float, c_long, c_double, POINTER
import numpy as np
import soundfile as sf
import subprocess
import shutil
import resampy
from scipy.io import savemat

# -------------------- CONFIG --------------------
LOMBARDGRID_ROOT = './data_and_mixing_instructions'
OUTPUT_DIR_8K = './lombardgrid_2_speakers/wav8k'
DATA_TYPES = ['l_tr','l_cv','l_tt', 'p_tr', 'p_cv', 'p_tt']
MIN_MAX = ['min']  # only minimal version here
FS8K = 8000
# ------------------------------------------------

# Try loading the native active-level library with OS-aware filename selection
_this_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

# Candidate names we try (feel free to adjust to your provided filenames)
if os.name == 'nt' or sys.platform.startswith('win'):
    lib_name = "libsvp56.dll"
else:
    lib_name = "libsvp56.so"

_found_path = None
p = os.path.join(_this_dir, lib_name)
if os.path.exists(p):
    _found_path = p


if _found_path is None:
    raise FileNotFoundError(
        f"Could not find a native active-level library in {_this_dir}."
        "Please place the libsvp56.dll (Windows) or libsvp56.so (Linux) in the script folder."
    )

_native = ctypes.CDLL(_found_path)
# define function prototype
# double p56_active_level_from_buffer(const float *buffer_in, long smpno, double fs, double *activity_out)
_native.p56_active_level_from_buffer.restype = c_double
_native.p56_active_level_from_buffer.argtypes = [POINTER(c_float), c_long, c_double, POINTER(c_double)]


def p56_active_level_from_buffer_numpy(buf: np.ndarray, fs: float):
    """Call the native p56_active_level_from_buffer on a numpy float32 1D array.
    Returns (level_returned_by_func, activity_out_double).
    """
    assert buf.dtype == np.float32, "buffer must be float32"
    assert buf.ndim == 1, "buffer must be 1D"
    n = buf.size
    buf_ctypes = buf.ctypes.data_as(POINTER(c_float))
    activity_out = c_double(0.0)
    level_db = _native.p56_active_level_from_buffer(buf_ctypes, c_long(n), c_double(fs), ctypes.byref(activity_out))
    lev_linear = 10.0 ** (float(level_db) / 10.0)

    lev_linear = max(lev_linear, 1e-12)
    # normalize as in activlev: y_norm = y / sqrt(level)
    #y_norm = buf.astype(np.float64) / np.sqrt(lev_linear)

    return buf, float(lev_linear)


def ensure_mono(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y
    # If multi-channel, average channels (MATLAB wavread often gave NxC)
    return np.mean(y, axis=1)


def read_and_resample_ffmpeg(path: str, target_fs: int) -> (np.ndarray, int):
    """Use ffmpeg (if available) to read and resample audio to `target_fs`.

    Returns (audio_float32_mono, target_fs)
    The audio is returned as float32 numpy array in range [-1.0, 1.0).
    """
    ffmpeg_exe = shutil.which('ffmpeg')
    if ffmpeg_exe is None:
        raise FileNotFoundError('ffmpeg executable not found in PATH')

    # Build ffmpeg command: output raw 32-bit float little endian PCM to stdout
    cmd = [ffmpeg_exe, '-i', path, '-vn', '-ac', '1', '-ar', str(target_fs), '-af', 'aresample=resampler=soxr', '-f', 'f32le', '-acodec', 'pcm_f32le', '-']
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed reading/resampling '{path}': {e.stderr.decode('utf-8', errors='replace')}")

    # Interpret stdout bytes as float32
    data = np.frombuffer(proc.stdout, dtype='<f4')  # little-endian 32-bit float
    if data.size == 0:
        # sometimes ffmpeg may not produce output (corrupt file) -- raise informative error
        raise RuntimeError(f"ffmpeg produced no audio data for '{path}'. stderr: {proc.stderr.decode('utf-8', errors='replace')}")
    return data.astype(np.float32), target_fs


def read_wav_mono(path: str):
    """Read audio and return (mono_float32, sample_rate). Will try ffmpeg first for speed,
    falling back to soundfile+resampy if ffmpeg isn't available or fails.
    """
    try:
        # try to read+resample with ffmpeg (fast)
        arr, fs = read_and_resample_ffmpeg(path, FS8K)
        return ensure_mono(arr).astype(np.float32), fs
    except FileNotFoundError:
        # ffmpeg not found -> fallback to soundfile + resampy
        data, fs = sf.read(path, always_2d=False)
        data = ensure_mono(np.asarray(data))
        if fs != FS8K:
            data = resampy.resample(data, fs, FS8K)
        return data.astype(np.float32), FS8K
    except Exception as e:
        # any ffmpeg error -> fallback but notify
        print(f"Warning: ffmpeg read failed for {path} ({e})."
              f" Falling back to soundfile/resampy.")
        data, fs = sf.read(path, always_2d=False)
        data = ensure_mono(np.asarray(data))
        if fs != FS8K:
            data = resampy.resample(data, fs, FS8K)
        return data.astype(np.float32), FS8K


os.makedirs(OUTPUT_DIR_8K, exist_ok=True)

for minmax in MIN_MAX:
    for dtype in DATA_TYPES:
        out_dir_for_type = os.path.join(OUTPUT_DIR_8K, minmax, dtype)
        os.makedirs(os.path.join(out_dir_for_type, 's1'), exist_ok=True)
        os.makedirs(os.path.join(out_dir_for_type, 's2'), exist_ok=True)
        os.makedirs(os.path.join(out_dir_for_type, 'mix'), exist_ok=True)

        task_file = os.path.join(LOMBARDGRID_ROOT, f'mix_{dtype}.txt')
        if not os.path.exists(task_file):
            print(f"Task file not found: {task_file} -- skipping {dtype}")
            continue

        # prepare output list files (like MATLAB script did)
        source1_list = []
        source2_list = []
        mix_list = []

        # we'll store scaling info in numpy arrays and save as .mat at the end
        with open(task_file, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]

        num_files = len(lines)
        scaling_8k = np.zeros((num_files, 2), dtype=np.float64)
        scaling16bit_8k = np.zeros((num_files,), dtype=np.float64)  # kept for compatibility, will store mix_scaling

        print(f"Processing: {minmax}_{dtype}  -- {num_files} items")

        for i, line in enumerate(lines):
            # expected format: <path1> <snr1> <path2> <snr2>
            parts = line.split()
            if len(parts) < 4:
                print(f"Skipping malformed line ({i+1}): {line}")
                continue
            relpath1, snr1_str, relpath2, snr2_str = parts[0], parts[1], parts[2], parts[3]
            snr1 = float(snr1_str)
            snr2 = float(snr2_str)

            # names
            invwav1_name = os.path.splitext(os.path.basename(relpath1))[0]
            invwav2_name = os.path.splitext(os.path.basename(relpath2))[0]
            mix_name = f"{invwav1_name}_{int(snr1)}_{invwav2_name}_{int(snr2)}"

            source1_list.append(relpath1)
            source2_list.append(relpath2)
            mix_list.append(mix_name)

            path1 = os.path.join(LOMBARDGRID_ROOT, relpath1)
            path2 = os.path.join(LOMBARDGRID_ROOT, relpath2)
            if not os.path.exists(path1) or not os.path.exists(path2):
                print(f"File missing for line {i+1}: {path1} or {path2} -- skipping")
                continue

            s1, fs1 = read_wav_mono(path1)
            s2, fs2 = read_wav_mono(path2)

            # ensure float32 (read_wav_mono already returns float32)
            s1_8k = s1.astype(np.float32)
            s2_8k = s2.astype(np.float32)

            # compute active levels using provided native function
            # NOTE: the numeric meaning of the returned 'level' should be verified with your MATLAB activlev output.
            try:
                s1_8k, lev1 = p56_active_level_from_buffer_numpy(s1_8k, FS8K)
            except Exception as e:
                raise RuntimeError(f"Error calling native active level function on file {path1}: {e}")
            try:
                s2_8k, lev2 = p56_active_level_from_buffer_numpy(s2_8k, FS8K)
            except Exception as e:
                raise RuntimeError(f"Error calling native active level function on file {path2}: {e}")

            # Protect against non-positive lev
            eps = 1e-12
            lev1 = max(lev1, eps)
            lev2 = max(lev2, eps)

            weight_1 = 10.0 ** (snr1 / 20.0)
            weight_2 = 10.0 ** (snr2 / 20.0)

            # Scale signals similar to MATLAB commented block: s = weight * s / sqrt(lev)
            s1_8k = weight_1 * s1_8k / np.sqrt(lev1)
            s2_8k = weight_2 * s2_8k / np.sqrt(lev2)

            # align lengths according to min/max policy
            if minmax == 'max':
                mix_len = max(len(s1_8k), len(s2_8k))
                if len(s1_8k) < mix_len:
                    s1_8k = np.concatenate([s1_8k, np.zeros(mix_len - len(s1_8k), dtype=np.float32)])
                if len(s2_8k) < mix_len:
                    s2_8k = np.concatenate([s2_8k, np.zeros(mix_len - len(s2_8k), dtype=np.float32)])
            else:  # 'min'
                mix_len = min(len(s1_8k), len(s2_8k))
                s1_8k = s1_8k[:mix_len]
                s2_8k = s2_8k[:mix_len]

            mix_8k = s1_8k + s2_8k

            # global normalization (avoid clipping) using 0.9 headroom
            max_amp_8k = max(np.max(np.abs(mix_8k)), np.max(np.abs(s1_8k)), np.max(np.abs(s2_8k)), eps)
            mix_scaling_8k = 0.9 / max_amp_8k
            s1_8k = mix_scaling_8k * s1_8k
            s2_8k = mix_scaling_8k * s2_8k
            mix_8k = mix_scaling_8k * mix_8k

            # save scaling factors like MATLAB did
            scaling_8k[i, 0] = weight_1 * mix_scaling_8k / np.sqrt(lev1)
            scaling_8k[i, 1] = weight_2 * mix_scaling_8k / np.sqrt(lev2)
            scaling16bit_8k[i] = mix_scaling_8k

            # write 16-bit PCM WAV files (int16) using soundfile
            # convert to int16 range -32768..32767 like MATLAB's int16(round(2^15 * x))
            def float_to_int16(arr: np.ndarray) -> np.ndarray:
                clipped = np.clip(arr, -1.0, 1.0 - 1.0 / (2**15))
                return (clipped * (2**15)).astype(np.int16)

            s1_16bit = float_to_int16(s1_8k)
            s2_16bit = float_to_int16(s2_8k)
            mix_16bit = float_to_int16(mix_8k)

            sf.write(os.path.join(out_dir_for_type, 's1', mix_name + '.wav'), s1_16bit, FS8K, subtype='PCM_16')
            sf.write(os.path.join(out_dir_for_type, 's2', mix_name + '.wav'), s2_16bit, FS8K, subtype='PCM_16')
            sf.write(os.path.join(out_dir_for_type, 'mix', mix_name + '.wav'), mix_16bit, FS8K, subtype='PCM_16')

            if (i + 1) % 10 == 0:
                print('.', end='', flush=True)
                if (i + 1) % 200 == 0:
                    print()

        # save scaling.mat (MATLAB format)
        out_mat = os.path.join(out_dir_for_type, 'scaling.mat')
        savemat(out_mat, {'scaling_8k': scaling_8k, 'scaling16bit_8k': scaling16bit_8k})

        # write the list files similarly to MATLAB's mix files
        path1 = os.path.join(LOMBARDGRID_ROOT, f'mix_2_spk_{minmax}_{dtype}_1')
        path2 = os.path.join(LOMBARDGRID_ROOT, f'mix_2_spk_{minmax}_{dtype}_2')
        path3 = os.path.join(LOMBARDGRID_ROOT, f'mix_2_spk_{minmax}_{dtype}_mix')

        with open(path1, 'w', encoding='utf-8') as f1, \
             open(path2, 'w', encoding='utf-8') as f2, \
             open(path3, 'w', encoding='utf-8') as fm:
            for a, b, c in zip(source1_list, source2_list, mix_list):
                f1.write(a)
                f2.write(b)
                fm.write(c)

print('Done.')
