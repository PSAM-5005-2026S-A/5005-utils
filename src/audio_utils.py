import numpy as np
import soundfile as sf
import wave

## Audio I/O

def get_samples_and_rate(wav_filename):
  with wave.open(wav_filename, mode="rb") as wav_in:
    if wav_in.getsampwidth() != 2:
      raise Exception("Input not 16-bit")

    nchannels = wav_in.getnchannels()
    nframes = wav_in.getnframes()
    nsamples = nchannels * nframes
    xb = wav_in.readframes(nframes)
    b_np = np.frombuffer(xb, dtype=np.int16) / nchannels
    samples = [int(sum(b_np[b0 : b0 + nchannels])) for b0 in range(0, nsamples, nchannels)]

    return samples, wav_in.getframerate()

def wav_to_list(wav_filename):
  s, _ = get_samples_and_rate(wav_filename)
  return s

def list_to_wav(samples, sr, filename):
  sf.write(filename, samples, sr, subtype="PCM_16")


# Audio Analysis Functions

def logFilter(x, factor=3):
  if factor < 1:
    return x
  else:
    return np.exp(factor * np.log(x)) // np.power(10, factor*5)

def fft(samples, rate=44100, filter_factor=0):
  _fft = logFilter(np.abs(np.fft.fft(samples * np.hanning(len(samples))))[ :len(samples) // 2], filter_factor).tolist()
  num_samples = len(_fft)
  hps = (rate//2) / num_samples
  _freqs = [s * hps for s in range(num_samples)]
  return _fft, _freqs

def stft(samples, rate=44100, window_len=1024):
  _times = list(range(0, len(samples), window_len))

  hps = (rate//2) / (window_len//2)
  _freqs = [s * hps for s in range(window_len//2)]

  sample_windows = [samples[s : s + window_len] for s in _times]
  sample_windows[-1] = (sample_windows[-1] + len(sample_windows[0]) * [0])[:len(sample_windows[0])]

  hammed_windows = sample_windows * np.hamming(window_len)
  _ffts = np.log(np.clip(np.abs(np.fft.rfft(hammed_windows))[:, :window_len//2], a_min=1e-3, a_max=None))

  return (_ffts.T).tolist(), _freqs, _times

def ifft(fs):
  return np.fft.fftshift(np.fft.irfft(fs)).tolist()

# Generate Audio

def tone(freq, length_seconds, amp=1, sr=44100, fade=False):
  length_samples = length_seconds * sr
  t = range(0, length_samples)
  ham = np.ones(length_samples)
  if fade:
    ham = np.hamming(length_samples)
  two_pi = 2.0 * np.pi
  return np.array([amp * np.sin(two_pi * freq * x / sr) for x in t] * ham).astype(np.float16).tolist(), sr

def multi_tone(freqs, length_seconds, amp=1, sr=44100, fade=False):
  tones = np.array([tone(f, length_seconds, amp, sr, fade)[0] for f in freqs])
  return tones.mean(axis=0), sr
