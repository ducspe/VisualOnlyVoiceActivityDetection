import numpy as np
import math
from librosa import util


def clean_speech_VAD(speech_t,
                     fs=16e3,
                     wlen_sec=50e-3,
                     hop_percent=0.25,
                     center=True,
                     pad_mode='reflect',
                     pad_at_end=True,
                     vad_threshold=1.70):
    """ Computes VAD based on threshold in the time domain

    Args:
        speech_t ([type]): [description]
        fs ([type]): [description]
        wlen_sec ([type]): [description]
        hop_percent ([type]): [description]
        center ([type]): [description]
        pad_mode ([type]): [description]
        pad_at_end ([type]): [description]
        eps ([type], optional): [description]. Defaults to 1e-8.

    Returns:
        ndarray: [description]
    """
    nfft = int(wlen_sec * fs)  # STFT window length in samples
    hopsamp = int(hop_percent * nfft)  # hop size in samples
    # Sometimes stft / istft shortens the output due to window size
    # so you need to pad the end with hopsamp zeros
    if pad_at_end:
        utt_len = len(speech_t) / fs
        if math.ceil(utt_len / wlen_sec / hop_percent) != int(utt_len / wlen_sec / hop_percent):
            y = np.pad(speech_t, (0, hopsamp), mode='constant')
        else:
            y = speech_t.copy()
    else:
        y = speech_t.copy()

    if center:
        y = np.pad(y, int(nfft // 2), mode=pad_mode)

    y_frames = util.frame(y, frame_length=nfft, hop_length=hopsamp)
    
    power = np.power(y_frames,2).sum(axis=0)
    vad = power > np.power(10, vad_threshold) * np.min(power)
    vad = np.float32(vad)
    vad = vad[None]
    return vad


def clean_speech_IBM(speech_tf,
                     eps=1e-8,
                     ibm_threshold=50):
    """ Calculate softened mask
    """
    mag = abs(speech_tf)
    power_db = 20 * np.log10(mag + eps)  # Smoother mask with log
    mask = power_db > np.max(power_db) - ibm_threshold
    mask = np.float32(mask)
    return mask


def noise_robust_clean_speech_IBM(speech_t,
                                  speech_tf,
                                  fs=16e3,
                                  wlen_sec=50e-3,
                                  hop_percent=0.25,
                                  center=True,
                                  pad_mode='reflect',
                                  pad_at_end=True,
                                  vad_threshold=1.70,
                                  eps=1e-8,
                                  ibm_threshold=50):
    """
    Create IBM labels robust to noisy speech recordings using noise-robst VAD.
    In particular, the labels are robust to noise occuring before / after speech.
    """
    # Compute vad
    vad = clean_speech_VAD(speech_t,
                           fs=fs,
                           wlen_sec=wlen_sec,
                           hop_percent=hop_percent,
                           center=center,
                           pad_mode=pad_mode,
                           pad_at_end=pad_at_end,
                           vad_threshold=vad_threshold)

    # Binary mask
    ibm = clean_speech_IBM(speech_tf,
                           eps=eps,
                           ibm_threshold=ibm_threshold)
    
    # Noise-robust binary mask
    ibm = ibm * vad
    return ibm
