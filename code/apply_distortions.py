# having import problem, not in use
import os
import sys
from tqdm import tqdm
import numpy as np
import librosa
import datasets
from datasets import load_from_disk, Dataset

# from scipy.signal import butter, sosfiltfilt, hilbert, spectrogram
# import scipy.signal as signal
# from scipy.signal.windows import triang, kaiser
# import scipy.io.wavfile as wav
# from scipy.interpolate import interp1d, interp2d, RectBivariateSpline
#
# from gammatone.gtgram import gtgram
# from gammatone.filters import centre_freqs, make_erb_filters, erb_filterbank

# import parselmouth

def fast(audio, sr, speedup=2.5):
    res = librosa.effects.time_stretch(audio, rate=speedup)
    return res

def time_reverse(audio, sr, segment_ms=62):
    segment_len = int(sr * (segment_ms / 1000))  # e.g. 62 ms → 992 samples
    reversed_audio = []
    for start in range(0, len(audio), segment_len):
        end = start + segment_len
        chunk = audio[start:end]
        reversed_audio.append(chunk[::-1])  # Reverse segment

    return np.concatenate(reversed_audio)

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


def multi_band_filter(y, sr, center_freqs_list, bandwidth_octaves, order=5):
    nyquist = sr / 2
    y_combined = np.zeros_like(y)

    for cf in center_freqs_list:
        factor = 2 ** (bandwidth_octaves / 2)
        low = cf / factor
        high = cf * factor
        low_norm = low / nyquist
        high_norm = high / nyquist

        if low_norm <= 0 or high_norm >= 1:
            continue  # skip invalid band

        sos = butter(order, [low_norm, high_norm], btype='band', output='sos')
        y_band = sosfiltfilt(sos, y)
        y_combined += y_band  # Sum each band

    return y_combined


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


def noise_vocode(signal, sr=16000):
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


def tone_vocode(signal, sr=16000):
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


def extract_f1_f2(waveform, sr=16000, step=0.01):
    sound = parselmouth.Sound(values=waveform, sampling_frequency=sr)
    tmin = sound.xmin
    tmax = sound.xmax

    formant_obj = sound.to_formant_burg(time_step=step, max_number_of_formants=5,
                                        maximum_formant=5500, window_length=0.025, pre_emphasis_from=50)

    times = np.arange(tmin, tmax, step)
    f1_list = []
    f2_list = []

    for time in times:
        f1 = formant_obj.get_value_at_time(1, time)
        f2 = formant_obj.get_value_at_time(2, time)

        f1 = -1 if f1 is None or np.isnan(f1) else round(f1)
        f2 = -1 if f2 is None or np.isnan(f2) else round(f2)

        f1_list.append(f1)
        f2_list.append(f2)

    return np.array(f1_list), np.array(f2_list)


def get_amplitudes_nearest_bin(formant_freqs, freq_bins, Sxx):
    amps = []
    for i, freq in enumerate(formant_freqs):
        idx = np.argmin(np.abs(freq_bins - freq))  # nearest frequency bin
        amps.append(Sxx[idx, i])  # i now correctly indexes time bin
    return np.array(amps)


def sinwave_speech(waveform, sr):
    waveform = np.array(waveform)
    f1, f2 = extract_f1_f2(waveform)
    sr = 16000
    # Spectrogram with ~10ms frame (to match Praat formant times)
    f, t, Sxx = spectrogram(waveform, fs=sr, window='hann', nperseg=160, noverlap=80)

    # Interpolate formants to match spectrogram time axis
    formant_times = np.arange(len(f1)) * 0.01  # 10 ms intervals
    f1_interp = interp1d(formant_times, f1, kind='linear', fill_value='extrapolate')(t)
    f2_interp = interp1d(formant_times, f2, kind='linear', fill_value='extrapolate')(t)

    # Get amplitudes via bilinear interpolation
    a1 = get_amplitudes_nearest_bin(f1_interp, f, Sxx)
    a2 = get_amplitudes_nearest_bin(f2_interp, f, Sxx)

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

    return sine_wave_speech


# glimpsed
def genenerate_centre_freqs(sr, channels=55, f_min=100, f_max=7500):  # 100-7500
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


def generate_glimpse_mask(speech, noise, sr, filters, threshold=3):
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


def apply_glimpse_mask_and_resynthesize(mixture, mask, sr, filters, tri_win):
    channels, time_frames = mask.shape
    samples_per_frame = int(sr / 100)  # 10 ms frame step
    win_len = len(tri_win)
    half_win = win_len // 2
    total_length = len(mixture)

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


# changed: added filters, tri_win
def glimpsed_speech(speech, sr, filters, tri_win):
    shaped_noise = generate_speech_shaped_noise(speech, sr)
    shaped_noise = match_snr(speech, shaped_noise, target_snr_db=0)
    mixture = speech + shaped_noise
    mask = generate_glimpse_mask(speech, shaped_noise, sr, filters)
    glimpsed = apply_glimpse_mask_and_resynthesize(mixture, mask, sr, filters, tri_win)
    return glimpsed


# added this function
def prepare_glimpsed_config(sr=16000, win_len_ms=20, channels=55, f_min=100, f_max=7500):
    centre_freqs = genenerate_centre_freqs(sr, channels, f_min, f_max)
    filters = make_erb_filters(sr, centre_freqs)
    win_len = int((win_len_ms / 1000) * sr)
    tri_win = triang(win_len)
    return filters, tri_win


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


# changed: to accept reusable music and filters
def sculpted_speech(speech, sr, music_wave, filters):
    music_fragment = segment_music(music_wave, speech, sr)
    ssn = generate_speech_shaped_noise(speech, sr)

    mask = generate_sculpted_mask(speech, ssn, sr, filters)

    music_bands = erb_filterbank(music_fragment, filters)
    upsampled = upsample_mask(mask, sr)

    masked_bands = []
    for ch, band in enumerate(music_bands):
        gain = upsampled[ch][:len(band)]
        masked_band = band * gain
        masked_bands.append(masked_band)

    sculpted = np.sum(masked_bands, axis=0)
    return sculpted


# added this configurer
def prepare_sculpted_config(sr=16000, channels=55, f_min=50, f_max=7500):
    music_wave, _ = librosa.load("music_bach.mp3", sr=sr)
    centre_freqs = genenerate_centre_freqs(sr, channels, f_min, f_max)
    filters = make_erb_filters(sr, centre_freqs)
    return music_wave, filters


def rms(signal):
    return np.sqrt(np.mean(signal ** 2))


def scale_tofactor(distortion_type, waveform):
    base_name = distortion_type.split("_")[0]
    rms_targets = {
        'glimpsed': .055,
        'noise_vocoded': .045,
        'sculpted': .053,
        'sinewave': .045,
        'narrowband': .03,
        'fast': .07,
        'reversed': .06,
        'tone_vocoded': .055,
        'practice': .062,
        "clean": 1.0
    }
    factor = rms_targets.get(base_name)
    if factor is None:
        raise ValueError(f"No scaling factor defined for base distortion type '{base_name}'")

    return waveform * factor / rms(waveform)


def same_return(waveform, sr):
    return waveform


def run_narrowband_sweep(ds):
    print('running sweep')
    sweep_configs = [
        {"name": "low_mid_1_3", "bands": [1100, 2100], "bandwidth": 1 / 3},
        {"name": "high_mid_1_3", "bands": [2100, 4200], "bandwidth": 1 / 3},
        {"name": "mid_only_1_3", "bands": [2100], "bandwidth": 1 / 3},
        {"name": "mid_only_2_3", "bands": [2100], "bandwidth": 2 / 3},
        {"name": "mid_only_1.0", "bands": [2100], "bandwidth": 1.0},
        {"name": "low_high_1_3", "bands": [1100, 4200], "bandwidth": 1 / 3}
    ]
    new_sweep_configs = [
        {"name": "all_bands_1_3", "bands": [1100, 2100, 4200], "bandwidth": 1 / 3},
        {"name": "all_bands_2_3", "bands": [1100, 2100, 4200], "bandwidth": 2 /3 },
        {"name": "low_mid_2_3", "bands": [1100, 2100], "bandwidth": 2 / 3},
        {"name": "high_mid_2_3", "bands": [2100, 4200], "bandwidth": 2 / 3},
    ]
    for config in new_sweep_configs:
        def apply_func(y, sr):
            return multi_band_filter(y, sr, config["bands"], config["bandwidth"], order=5)

        distortion_type = "narrowband"
        distortion_condition = f"{distortion_type}_{config['name']}"
        gen_new_dataset(ds, distortion_type, distortion_condition, apply_func)

def run_reversed_sweep(ds):
    print('running sweep')
    sweep_configs = [
        {"name": "40ms", "win_size": 40},
        {"name": "50ms", "win_size": 50},
        {"name": "80ms", "win_size": 80},
        {"name": "100ms", "win_size": 100},
    ]
    for config in sweep_configs:
        def apply_func(y, sr):
            return time_reverse(y, sr, config["win_size"])

        distortion_type = "reversed"
        distortion_condition = f"{distortion_type}_{config['name']}"
        gen_new_dataset(ds, distortion_type, distortion_condition, apply_func)

# changed apply_distortion
def apply_distortion(ds, distortion_type, condition, sweep=False):
    # sr must be 16000
    distortion_func = None
    distortion = distortion_type

    if distortion_type == "fast":
        distortion_func = fast
    elif distortion_type == "reversed":
        distortion_func = time_reverse
        if sweep:
            run_reversed_sweep(ds)
            return
    elif distortion_type == "narrowband":
        distortion_func = narrowband
        if sweep:
            run_narrowband_sweep(ds)
            return
    elif distortion_type == "noisevoc":
        distortion_func = noise_vocode
    elif distortion_type == "tonevoc":
        distortion_func = tone_vocode
    elif distortion_type == "sinewave":
        distortion_func = sinwave_speech
    elif distortion_type == "sculpted":
        music_wave, filters = prepare_sculpted_config(sr=16000)
        distortion_func = lambda waveform, sr: sculpted_speech(waveform, sr, music_wave, filters)
    elif distortion_type == "glimpsed":
        filters, tri_win = prepare_glimpsed_config(sr=16000)
        distortion_func = lambda waveform, sr: glimpsed_speech(waveform, sr, filters, tri_win)
    elif distortion_type == "clean":
        distortion_func = same_return

    if distortion_func is None:
        raise ValueError(f"Unknown distortion type: {distortion_type}")

    gen_new_dataset(ds, distortion,condition, distortion_func)


# tried with a different music
def gen_new_dataset(original_ds, distortion_type, distortion_condition,apply_function):
    outpath = f"../ted3test_distorted_adjusted/{distortion_type}/{distortion_condition}"
    # outpath = f"../ted1train_distorted/{distortion_type}"
    os.makedirs(outpath, exist_ok=True)

    distort_ds = []
    pbar = tqdm(original_ds, desc="Starting")
    for sample in pbar:
        transcript = sample["text"].strip().lower()

        if "ignore_time_segment_in_scoring" in transcript or "inter_segment_gap" in transcript:
            pbar.set_description("skipping")
            continue

        audio = sample["audio"]  # a dictionary with keys: array and sampling rate
        waveform = audio["array"]
        sr = audio["sampling_rate"]
        pbar.set_description("Distorting")
        new_waveform = apply_function(waveform, sr=16000)

        pbar.set_description("Scaling")
        scaled_new_waveform = scale_tofactor(distortion_type, new_waveform)

        distort_sample = dict(sample)
        distort_sample["audio"]["array"] = scaled_new_waveform
        distort_ds.append(distort_sample)

    new_dataset = Dataset.from_list(distort_ds)
    new_dataset.save_to_disk(outpath)
    print(f"new dataset saved to {outpath}")


if __name__ == "__main__":
    distortion_type = sys.argv[1].lower()
    # my_ds = load_from_disk(f"../ted3test_distorted/clean/") # basis clean data to apply distortions to
    my_ds = load_from_disk("../ted3test_distorted/clean")
    my_ds = my_ds.cast_column("audio", datasets.Audio(decode=False))
    sweep = sys.argv[2].lower() == "true" if len(sys.argv) > 2 else False
    apply_distortion(my_ds, distortion_type, sweep)