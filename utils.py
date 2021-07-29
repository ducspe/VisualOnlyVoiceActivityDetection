import librosa
from processing.target import clean_speech_VAD
import os
import torch
import torch.nn.functional as F
from datetime import datetime


# Define parameters:
# global_frame_rate = 29.970030  # frames per second
wlen_sec = 0.064  # window length in seconds
hop_percent = 0.25  # math.floor((1 / (wlen_sec * global_frame_rate)) * 1e4) / 1e4  # hop size as a percentage of the window length
win = 'hann'  # type of window function (to perform filtering in the time domain)
center = False  # see https://librosa.org/doc/0.7.2/_modules/librosa/core/spectrum.html#stft
pad_mode = 'reflect'  # This argument is ignored if center = False
pad_at_end = True  # pad audio file at end to match same size after stft + istft

# Noise robust VAD
vad_quantile_fraction_begin = 0.5  # 0.93
vad_quantile_fraction_end = 0.55  # 0.99
vad_quantile_weight = 1.0  # 0.999
vad_threshold = 1.7

# Noise robust IBM
ibm_quantile_fraction = 0.25  # 0.999
ibm_quantile_weight = 1.0  # 0.999
ibm_threshold = 50


# Other parameters:
sampling_rate = 16000
dtype = 'complex64'
eps = 1e-8


def create_ground_truth_labels_from_path(audio_path):
    raw_clean_audio, Fs = librosa.load(audio_path, sr=sampling_rate)

    mask_labels = clean_speech_VAD(raw_clean_audio,
                           fs=sampling_rate,
                           wlen_sec=wlen_sec,
                           hop_percent=hop_percent,
                           center=center,
                           pad_mode=pad_mode,
                           pad_at_end=pad_at_end,
                           vad_threshold=vad_threshold)

    return mask_labels.T


def create_ground_truth_labels(raw_clean_audio):

    mask_labels = clean_speech_VAD(raw_clean_audio,
                           fs=sampling_rate,
                           wlen_sec=wlen_sec,
                           hop_percent=hop_percent,
                           center=center,
                           pad_mode=pad_mode,
                           pad_at_end=pad_at_end,
                           vad_threshold=vad_threshold)

    return mask_labels.T


def create_video_paths_list(base_path):
    video_paths_list = []
    speaker_folders = sorted([x for x in os.listdir(base_path)])
    for speaker in speaker_folders:
            speaker_path = os.path.join(base_path, speaker)
            speaker_mat_files = sorted([y for y in os.listdir(speaker_path)])

            for sentence_mat_file in speaker_mat_files:
                sentence_video_path = os.path.join(speaker_path, sentence_mat_file)
                video_paths_list.append(sentence_video_path)

    return video_paths_list


def create_audio_paths_list(base_path):
    audio_paths_list = []
    speaker_folders = sorted([x for x in os.listdir(base_path)])
    for speaker in speaker_folders:
            speaker_path = os.path.join(base_path, speaker, "straightcam")
            speaker_wav_files = sorted([y for y in os.listdir(speaker_path)])

            for sentence_wav_file in speaker_wav_files:
                sentence_audio_path = os.path.join(speaker_path, sentence_wav_file)
                audio_paths_list.append(sentence_audio_path)

    return audio_paths_list


# A custom collate function for many-to-many training
# The function creates the batch by padding the inputs and labels
# with respect to the longest sequence in the batch
# Note that this collate_many2many gets called after the getitem function.
def collate_many2many(batch):
    # print("The batch is: ", batch)
    lengths = [i[0].shape[0] for i in batch]  # get the length of each sequence in the batch. Remember that one sample in the batch
                                              # is actually a video file
    batch_size = len(batch)

    max_framesseq_length = max(lengths)

    _, channels, height, width = batch[0][0].size()  # _ stands for sequnce_length/nr_of_frames in this particular batch

    padded_data = torch.zeros((batch_size, max_framesseq_length, channels, height, width))
    padded_target = torch.zeros((batch_size, max_framesseq_length, 1))

    for idx, (sample, length) in enumerate(zip(batch, lengths)):
        # Pad sequence at the END of nr_of_frames/sequence_length dimension
        # Note that below it makes a difference if we say: (0, max_frames...) or (max_frames..., 0)
        padded_data[idx] = F.pad(input=sample[0], pad=(0,0 , 0,0 , 0,0 , 0,max_framesseq_length-length), mode='constant', value=0.)  # pad last dimension
        padded_target[idx] = F.pad(input=sample[1], pad=(0,0 , 0,max_framesseq_length-length), mode='constant', value=0.)

    lengths = torch.LongTensor(lengths)

    return lengths, padded_data, padded_target


class MyLogger:
    def __init__(self, prefix):
        os.makedirs("logs_folder", exist_ok=True)
        logs = open(f"logs_folder/{prefix}loggerfile_{datetime.now()}.txt", "w")
        self.name = logs.name
        logs.write(f"The logging started at: {datetime.now()}\n\n")
        logs.close()

    def log(self, my_message, extra_newline=False):
        logs = open(self.name, "a")
        logs.write(f"{my_message}\n")
        if extra_newline:
            logs.write("\n")
        logs.close()
