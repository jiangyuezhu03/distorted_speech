import numpy as np
import sys
import os
import subprocess
from io import StringIO
from tqdm import tqdm

import librosa
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, hilbert, spectrogram
import scipy.signal as signal
from scipy.signal.windows import triang, kaiser
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline

from gammatone.gtgram import gtgram
from gammatone.filters import centre_freqs, make_erb_filters, erb_filterbank

from datasets import load_dataset

SR = 16000
ROOT = "C:\\Users\\28177\\ASR_test"
IN_ROOT = os.path.join(ROOT, "TEDLIUM_release1")


# fast
def fast(audio, sr, speedup=2.5):
    res = librosa.effects.time_stretch(audio, rate=speedup)
    return res


# time-reversed
def time_reverse(audio, sr, segment_ms=62):
    segment_len = int(sr * (segment_ms / 1000))  # e.g. 62 ms → 992 samples
    reversed_audio = []

    for start in range(0, len(audio), segment_len):
        end = start + segment_len
        chunk = audio[start:end]
        reversed_audio.append(chunk[::-1])  # Reverse segment

    return np.concatenate(reversed_audio)


# narrowband
def narrowband(y, sr, center_freq=2000):
    # Compute lower and upper cutoff frequencies for 1/3-octave band
    factor = 2 ** (1 / 6)  # 1/3 octave = 2^(1/3) => ±1/6 on log scale
    low = center_freq / factor
    high = center_freq * factor

    nyquist = sr / 2
    low_norm = low / nyquist
    high_norm = high / nyquist

    # 5th-order Butterworth bandpass filter
    sos = butter(N=5, Wn=[low_norm, high_norm], btype='band', output='sos')

    # Apply zero-phase filtering
    y_filtered = sosfiltfilt(sos, y)
    return y_filtered


# for noise/tone vocoded
def create_noise(length, target_rms=0.1):
    noise = np.random.normal(0, 1, length).astype(np.float32)

    # Normalize to target RMS (optional, for consistent loudness)
    rms = np.sqrt(np.mean(noise ** 2))
    noise = noise * (target_rms / rms)

    # Optional: clip to [-1, 1] to avoid overflow when saving
    noise = np.clip(noise, -1.0, 1.0)
    return noise


def extract_envelope(signal, window):
    power = signal ** 2
    smoothed = np.convolve(power, window, mode='same')
    return np.sqrt(smoothed)


def bandpass_filter(signal, low, high, sr):
    sos = butter(5, [low, high], btype='bandpass', fs=sr, output='sos')
    return sosfiltfilt(sos, signal)


def generate_sine_carrier(freq, length, sr):
    t = np.arange(length) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def noise_vocode(signal, sr):
    band_edges = [100, 328, 713, 1365, 2469, 4338, 7500]
    window_len = int(0.064 * sr)
    kaiser_window = kaiser(window_len, beta=20)

    vocoded = np.zeros_like(signal)

    for i in range(len(band_edges) - 1):
        low, high = band_edges[i], band_edges[i + 1]
        band = bandpass_filter(signal, low, high, sr)
        envelope = extract_envelope(band, kaiser_window)

        noise = np.random.normal(0, 1, len(signal)).astype(np.float32)
        vocoder = bandpass_filter(noise, low, high, sr)
        modulated = envelope * vocoder
        vocoded += modulated

    return vocoded


def tone_vocode(signal, sr):
    band_edges = [100, 328, 713, 1365, 2469, 4338, 7500]
    window_len = int(0.064 * sr)
    kaiser_window = kaiser(window_len, beta=20)

    vocoded = np.zeros_like(signal)

    for i in range(len(band_edges) - 1):
        low, high = band_edges[i], band_edges[i + 1]
        band = bandpass_filter(signal, low, high, sr)
        envelope = extract_envelope(band, kaiser_window)

        f_carrier = np.sqrt(low * high)
        vocoder = generate_sine_carrier(f_carrier, len(signal), sr)
        modulated = envelope * vocoder
        vocoded += modulated

    return vocoded


# sinewave
def obtain_formants_praat(speech_file, formant_output_path):
    with open(formant_output_path, 'wb') as out_f:
        subprocess.run([praat_path, praat_script, speech_file], stdout=out_f)
    with open(formant_output_path, "rb") as f:
        raw = f.read()
    cleaned = raw.replace(b"\x00", b"").decode("utf-8", errors="ignore")
    params = np.loadtxt(StringIO(cleaned)).T
    f1 = params[2]
    f2 = params[3]
    return f1, f2


def get_amplitudes_nearest_bin(formant_freqs, freq_bins, Sxx):
    amps = []
    for i, freq in enumerate(formant_freqs):
        idx = np.argmin(np.abs(freq_bins - freq))  # nearest frequency bin
        amps.append(Sxx[idx, i])  # i now correctly indexes time bin
    return np.array(amps)


def sinwave_speech_nearestbin(filepath):
    waveform, sr = librosa.load(filepath, sr=16000)
    praat_out = "praat_output.txt"
    f1, f2 = obtain_formants_praat(filepath, praat_out)

    # Spectrogram using 16ms window (to match the paper and formant steps)
    f, t, Sxx = spectrogram(
        waveform,
        fs=sr,
        window='hann',
        nperseg=256,  # 16 ms at 16 kHz
        noverlap=128  # 50% overlap
    )

    # Align formant values to spectrogram times
    formant_times = np.arange(len(f1)) * 0.01  # 10ms intervals
    f1_interp = interp1d(formant_times, f1, kind='linear', fill_value='extrapolate')(t)
    f2_interp = interp1d(formant_times, f2, kind='linear', fill_value='extrapolate')(t)

    # Get amplitudes via nearest-bin lookup
    a1 = get_amplitudes_nearest_bin(f1_interp, f, Sxx)
    a2 = get_amplitudes_nearest_bin(f2_interp, f, Sxx)

    # Interpolate frequencies and amplitudes to sample level (16kHz)
    total_duration = len(waveform) / sr
    times_interp = np.arange(0, total_duration, 1 / sr)

    f1_freq_interp = interp1d(t, f1_interp, kind='linear', fill_value='extrapolate')(times_interp)
    f1_amp_interp = interp1d(t, a1, kind='linear', fill_value='extrapolate')(times_interp)
    f2_freq_interp = interp1d(t, f2_interp, kind='linear', fill_value='extrapolate')(times_interp)
    f2_amp_interp = interp1d(t, a2, kind='linear', fill_value='extrapolate')(times_interp)

    # Generate sine waves
    f1_phase = 2 * np.pi * np.cumsum(f1_freq_interp) / sr
    f2_phase = 2 * np.pi * np.cumsum(f2_freq_interp) / sr

    sine_wave1 = np.sin(f1_phase) * f1_amp_interp
    sine_wave2 = np.sin(f2_phase) * f2_amp_interp

    sine_wave_speech = sine_wave1 + sine_wave2
    sine_wave_speech /= np.max(np.abs(sine_wave_speech))

    return sine_wave_speech, sr


def get_amplitudes_bilinear(formant_freqs, time_bins, freq_bins, Sxx):
    interp_func = RectBivariateSpline(freq_bins, time_bins, Sxx)
    amps = np.array([interp_func(freq, t)[0, 0] for freq, t in zip(formant_freqs, time_bins)])
    return amps


def sinwave_speech(filepath):
    waveform, sr = librosa.load(filepath, sr=16000)
    praat_out = "praat_output.txt"
    f1, f2 = obtain_formants_praat(filepath, praat_out)

    # Spectrogram with ~10ms frame (to match Praat formant times)
    f, t, Sxx = spectrogram(waveform, fs=sr, window='hann', nperseg=160, noverlap=80)

    # Interpolate formants to match spectrogram time axis
    formant_times = np.arange(len(f1)) * 0.01  # 10 ms intervals
    f1_interp = interp1d(formant_times, f1, kind='linear', fill_value='extrapolate')(t)
    f2_interp = interp1d(formant_times, f2, kind='linear', fill_value='extrapolate')(t)

    # Get amplitudes via bilinear interpolation
    a1 = get_amplitudes_bilinear(f1_interp, t, f, Sxx)
    a2 = get_amplitudes_bilinear(f2_interp, t, f, Sxx)

    # Interpolate to sample-level (16kHz) time axis
    total_duration = len(waveform) / sr
    times_interp = np.arange(0, total_duration, 1 / sr)

    f1_freq_interp = interp1d(t, f1_interp, kind='linear', fill_value='extrapolate')
    f1_amp_interp = interp1d(t, a1, kind='linear', fill_value='extrapolate')
    f2_freq_interp = interp1d(t, f2_interp, kind='linear', fill_value='extrapolate')
    f2_amp_interp = interp1d(t, a2, kind='linear', fill_value='extrapolate')

    f1_inst_freq = f1_freq_interp(times_interp)
    f1_inst_amp = f1_amp_interp(times_interp)
    f2_inst_freq = f2_freq_interp(times_interp)
    f2_inst_amp = f2_amp_interp(times_interp)

    f1_phase = 2 * np.pi * np.cumsum(f1_inst_freq) / sr
    f2_phase = 2 * np.pi * np.cumsum(f2_inst_freq) / sr

    sine_wave1 = np.sin(f1_phase) * f1_inst_amp
    sine_wave2 = np.sin(f2_phase) * f2_inst_amp
    sine_wave_speech = sine_wave1 + sine_wave2

    # Normalize
    sine_wave_speech /= np.max(np.abs(sine_wave_speech))

    return sine_wave_speech, sr


# glimpsed
def genenerate_centre_freqs(speech, sr, channels=55, f_min=50, f_max=7500):
    center_freqs = centre_freqs(sr, channels, f_min, f_max)
    return center_freqs


def generate_speech_shaped_noise(speech, sr):
    noise_len = len(speech)

    # Step 1: Compute spectral envelope of the speech
    S = np.abs(librosa.stft(speech, n_fft=1024))  # (freq_bins, frames)
    mean_spectrum = np.mean(S, axis=1)  # average over time
    mean_spectrum /= np.max(mean_spectrum)  # normalize

    # Step 2: Generate white noise
    white_noise = np.random.randn(noise_len)

    # Step 3: Design FIR filter with desired magnitude response
    n_fft = len(mean_spectrum) * 2
    freq = np.linspace(0, 1, len(mean_spectrum))
    filt = signal.firwin2(numtaps=1025, freq=freq, gain=mean_spectrum)

    # Step 4: Filter white noise to shape its spectrum
    shaped_noise = signal.lfilter(filt, 1.0, white_noise)

    # Step 5: Trim to match original speech length
    shaped_noise = shaped_noise[:noise_len]

    return shaped_noise


def match_snr(speech, noise, target_snr_db=0):
    speech_rms = rms(speech)
    noise_rms = rms(noise)

    desired_noise_rms = speech_rms / (10 ** (target_snr_db / 20))
    scaled_noise = noise * (desired_noise_rms / noise_rms)
    return scaled_noise


def extract_smoothed_envelope(signal, sr, tau_ms=8):
    env = np.abs(hilbert(signal))
    return smooth_exponential(env, sr, tau_ms)


def smooth_exponential(envelope, sr, tau_ms):
    tau = tau_ms / 1000
    alpha = np.exp(-1 / (tau * sr))
    smoothed = np.zeros_like(envelope)
    for i in range(1, len(envelope)):
        smoothed[i] = alpha * smoothed[i - 1] + (1 - alpha) * envelope[i]
    return smoothed


def generate_glimpse_mask(speech, noise, sr, filters, threshold=3, channels=55, f_min=50, f_max=7500):
    # 2. Apply filterbank to speech and noise
    speech_bands = erb_filterbank(speech, filters)
    noise_bands = erb_filterbank(noise, filters)

    # 3. Envelope extraction and smoothing
    speech_env = np.array([extract_smoothed_envelope(b, sr) for b in speech_bands])
    noise_env = np.array([extract_smoothed_envelope(b, sr) for b in noise_bands])

    # 4. Downsample to 10ms (100 Hz frame rate)
    step = int(sr / 100)
    speech_env_ds = speech_env[:, ::step]
    noise_env_ds = noise_env[:, ::step]

    eps = 1e-15
    mask = (20 * np.log10(speech_env_ds + eps) > 20 * np.log10(noise_env_ds + eps) + threshold)

    return mask


def apply_glimpse_mask_and_resynthesize(mixture, mask, sr, filters, win_len_ms=20):
    channels, time_frames = mask.shape
    samples_per_frame = int(sr / 100)  # 10 ms frame step
    win_len = int((win_len_ms / 1000) * sr)
    half_win = win_len // 2
    total_length = len(mixture)

    # Triangular window
    tri_win = triang(win_len)

    # ERB filterbank
    filtered = erb_filterbank(mixture, filters)

    # Initialize output bands
    band_resynth = np.zeros((channels, total_length))

    for ch in range(channels):
        band = filtered[ch]
        acc = np.zeros_like(band)

        for t in range(time_frames):
            if not mask[ch, t]:
                continue
            center = t * samples_per_frame
            start = max(0, center - half_win)
            end = min(total_length, center + half_win)
            w_start = half_win - (center - start)
            w_end = w_start + (end - start)

            acc[start:end] += band[start:end] * tri_win[w_start:w_end]

        band_resynth[ch] = acc

    # Forward pass
    summed = np.sum(band_resynth, axis=0)

    # Reverse, filter again, re-reverse
    reversed_signal = summed[::-1]
    filtered_again = erb_filterbank(reversed_signal, filters)
    summed_again = np.sum(filtered_again, axis=0)[::-1]

    return summed_again.astype(np.float32)


def glimpsed_speech(speech, sr):
    shaped_noise = generate_speech_shaped_noise(speech, sr)
    shaped_noise = match_snr(speech, shaped_noise, target_snr_db=0)
    mixture = speech + shaped_noise
    centre_freqs = genenerate_centre_freqs(speech, sr)
    filters = make_erb_filters(sr, centre_freqs)
    mask = generate_glimpse_mask(speech, shaped_noise, sr, filters)  # threshold = 2

    glimpsed = apply_glimpse_mask_and_resynthesize(mixture, mask, sr, filters)
    return glimpsed


# sculpted
# suggested fix from chatgpt
def upsample_mask(mask, sr, frame_rate=100):
    num_channels, num_frames = mask.shape
    full_length = int(num_frames * sr / frame_rate)
    upsampled = np.zeros((num_channels, full_length))
    x_old = np.linspace(0, full_length, num=num_frames)
    x_new = np.arange(full_length)
    for ch in range(num_channels):
        f = interp1d(x_old, mask[ch], kind='linear', fill_value="extrapolate")
        upsampled[ch] = f(x_new)
    return upsampled


# ------------#
def segment_music(music, speech, sr):
    speech_len = len(speech) / sr
    start_sample = np.random.randint(0, len(music) - int(sr * speech_len))
    music_segment = music[start_sample: start_sample + int(sr * speech_len)]
    return music_segment


def generate_sculpted_mask(speech, masker, sr, filters, threshold=-6):
    # 2. Apply filterbank to speech and noise
    speech_bands = erb_filterbank(speech, filters)
    masker_bands = erb_filterbank(masker, filters)

    # 3. Envelope extraction and smoothing
    speech_env = np.array([extract_smoothed_envelope(b, sr) for b in speech_bands])
    masker_env = np.array([extract_smoothed_envelope(b, sr) for b in masker_bands])

    # 4. Downsample to 10ms (100 Hz frame rate)
    step = int(sr / 100)
    speech_env_ds = speech_env[:, ::step]
    masker_env_ds = masker_env[:, ::step]

    # 5. Compute mask: speech > music - 6 dB
    eps = 1e-10
    mask = 20 * np.log10(speech_env_ds + eps) > 20 * np.log10(masker_env_ds + eps) + threshold
    return mask


def sculpted_speech(speech, sr):
    # generate music carrier
    music_wave, _ = librosa.load("music_bach.mp3", sr=16000)
    music_fragment = segment_music(music_wave, speech, sr)
    ssn = generate_speech_shaped_noise(speech, sr)

    centre_freqs = genenerate_centre_freqs(speech, sr)
    filters = make_erb_filters(sr, centre_freqs)

    mask = generate_sculpted_mask(speech, ssn, sr, filters)

    music_bands = erb_filterbank(music_fragment, filters)

    # Sample rate for gain mask (100 Hz)
    upsampled = upsample_mask(mask, sr)  # shape: (channels, n_samples)

    masked_bands = []
    for ch, band in enumerate(music_bands):
        gain = upsampled[ch][:len(band)]
        masked_band = band * gain
        masked_bands.append(masked_band)

    # Sum and normalize output
    sculpted = np.sum(masked_bands, axis=0)

    return sculpted


# for scaling
def rms(signal):
    return np.sqrt(np.mean(signal ** 2))


def get_ref_waveform(filepath):
    wave, _ = librosa.load(filepath, sr=16000)
    return wave


def scale_toclean(reference_waveform, my_waveform):
    target_rms = rms(reference_waveform)
    your_rms = rms(my_waveform)

    scaled_wave = my_waveform * (target_rms / your_rms)
    return scaled_wave


def scale_tofactor(distortion_type, waveform):
    rms_targets = {
        'glimpsed': .055,
        'noise_vocoded': .045,
        'sculpted': .053,
        'sinewave': .045,
        'narrowband': .03,
        'fast': .07,
        'reversed': .06,
        'tone_vocoded': .055,
        'practice': .062
    }
    factor = rms_targets.get(distortion_type)
    return waveform * factor / rms(waveform)


def process_and_save(sample, split_name, distortion_name, transform_fn, scale, output_root):
    filename = os.path.basename(sample['audio']['path'])
    in_path = os.path.join(IN_ROOT, split_name, "sph", filename)
    if distortion_name == "sinewave":
        # pass in filepath instead of waveform
        y, sr = sinwave_speech(os.path.join("test_sample", in_path))
    else:
        y, sr = librosa.load(in_path, sr=16000)
        y = transform_fn(y, sr)
    # prescale the audio to make it louder
    print("prescaling")
    y_ref = get_ref_waveform("hvd_481.wav")
    y_out = scale_toclean(y_ref, y)

    if scale == "different":
        y_scaled = scale_tofactor(distortion_name, y_out)
    if scale == "same":
        y_scaled = scale_toclean(get_ref_waveform("hvd_481.wav"), y_out)

    fname = os.path.splitext(os.path.basename(in_path))[0] + ".flac"
    out_path = os.path.join(output_root, split_name, distortion_name, fname)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, y_scaled, sr)


def process_folder(directory, distortion_name, transform_fn, scale, output_root):
    for fname in os.listdir(directory):
        if not fname.lower().endswith((".wav", ".flac", ".mp3")):
            continue  # skip non-audio files

        in_path = os.path.join(directory, fname)
        if distortion_name == "sinewave":
            # pass in filepath instead of waveform
            y_out, sr = sinwave_speech(os.path.join("test_sample", fname))
        else:
            y, sr = librosa.load(in_path, sr=16000)
            # prescale the audio to make it louder
            print("prescaling")
            y_ref = get_ref_waveform("hvd_481.wav")
            y_out = scale_toclean(y_ref, y)

            # Apply transformation
            y_out = transform_fn(y, sr)

        # Apply scaling
        if scale == "different":
            y_scaled = scale_tofactor(distortion_name, y_out)
        elif scale == "same":
            # You may want to parameterize the reference file too
            y_ref = get_ref_waveform("hvd_481.wav")
            y_scaled = scale_toclean(y_ref, y_out)
        else:
            y_scaled = y_out  # no scaling

        # Save output
        out_fname = os.path.splitext(fname)[0] + ".flac"
        out_path = os.path.join(output_root, distortion_name, out_fname)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sf.write(out_path, y_scaled, sr)

        print(f"Processed {fname} -> {out_path}")


if __name__ == "__main__":

    distortion_type = sys.argv[1].lower()
    split = sys.argv[2]  # train, validation, test
    scaling = "different"
    out_folder = "/work/tc068/tc068/jiangyue_zhu/.cache/"

    ds = load_from_disk(os.path.join(ROOT, "ted_cache"))[split]

    if distortion_type == "fast":
        distortion = "fast"
        distortion_func = fast
    if distortion_type == "reversed":
        distortion = "reversed"
        distortion_func = time_reverse
    if distortion_type == "narrowband":
        distortion = "narrowband"
        distortion_func = narrowband
    if distortion_type == "noisevoc":
        distortion = "noise_vocoded"
        distortion_func = noise_vocode
    if distortion_type == "tonevoc":
        distortion = "tone_vocoded"
        distortion_func = tone_vocode
    if distortion_type == "sinewave":
        distortion = "sinewave"
        praat_path = "C:/Users/28177/Praat.exe"
        praat_script = os.path.join(ROOT, 'acoustic_analysis.praat')
        distortion_func = sinwave_speech_nearestbin
    if distortion_type == "glimpsed":
        distortion = "glimpsed"
        distortion_func = glimpsed_speech
    if distortion_type == "sculpted":
        music_wave, _ = librosa.load("music_bach.mp3", sr=16000)
        distortion = "sculpted"
        distortion_func = sculpted_speech

    # process_folder("test_sample",distortion, distortion_func, scaling, "test_sample/distorted")
    print(f'Running {distortion_type} distortion, rescaling with {scaling}, saving to {out_folder}')
    for sample in tqdm(ds):
        process_and_save(sample, split, distortion_name=distortion, transform_fn=distortion_func, scale=scaling,
                         output_root=out_folder)