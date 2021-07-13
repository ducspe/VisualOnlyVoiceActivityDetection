import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms
import utils
import librosa
import skvideo.io
import os
#import cv2

# params:
epsilon = 1e-8  # for numerical stability
sampling_rate = 16000


class VideoDataset(Dataset):
    def __init__(self, video_paths, label_paths, train_statistics, streamtype):
        self.video_paths = video_paths
        self.label_paths = label_paths
        self.video_train_info = np.load(train_statistics, allow_pickle=True)
        self.video_train_normalization_mean = self.video_train_info.item()["all_videos_mean_before_normalization"]
        self.video_train_normalization_std = self.video_train_info.item()["all_videos_std_before_normalization"]

        self.video_ram = []
        self.clean_label_ram = []

        for video_pa in video_paths:
            preprocessed_file = np.load(video_pa)
            loaded = np.nan_to_num(preprocessed_file) 
            if streamtype == "of":
                self.video_ram.append((loaded + np.ones_like(loaded))/(loaded.flatten().max()+epsilon))
            else:
                self.video_ram.append(preprocessed_file/255.0)
        for clean_audio_pa in self.label_paths:
            clean_audio, Fs = librosa.load(clean_audio_pa, sr=sampling_rate)
            clean_audio_label = utils.create_ground_truth_labels(clean_audio)
            self.clean_label_ram.append(clean_audio_label)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, id):
        # Video:
        normalize = torchvision.transforms.Normalize(
            # mean=[self.video_train_normalization_mean, self.video_train_normalization_mean,
            #       self.video_train_normalization_mean],
            # std=[self.video_train_normalization_std, self.video_train_normalization_std,
            #      self.video_train_normalization_std]
            mean=[0, 0, 0],
            std=[1, 1, 1]
            )
        transform_forward = torchvision.transforms.Compose([
            torchvision.transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            normalize,
        ])

        ram_frames = self.video_ram[id]

        video_sample = torch.from_numpy(np.stack((ram_frames,) * 3, axis=1))

        augmented_video_sample = transform_forward(video_sample)

        # For test purposes, to see how the augmentation looks like:
        # print("aug: ", augmented_video_sample.shape)
        #
        # augmented_video_sample_numpy = augmented_video_sample.cpu().numpy()
        #
        # os.makedirs(f'mp4_dataout/', exist_ok=True)
        # dda_mp4_file = f'mp4_dataout/check.mp4'
        # upsampling_video_writer = skvideo.io.FFmpegWriter(dda_mp4_file,
        #                                                   inputdict={'-r': str(62.5),
        #                                                              '-s': '{}x{}'.format(67, 67)},
        #                                                   outputdict={'-filter:v': 'fps=fps={}'.format(
        #                                                       62.5),
        #                                                       '-c:v': 'libx264',
        #                                                       '-crf': str(17),
        #                                                       '-preset': 'veryslow'}
        #                                                   )
        #
        # for up_frame in augmented_video_sample_numpy:
        #     upsampling_video_writer.writeFrame(up_frame*255.0)
        #
        # upsampling_video_writer.close()

        # Label:
        label = torch.from_numpy(self.clean_label_ram[id]).type(torch.FloatTensor)
        sync_len = min(video_sample.shape[0], label.shape[0])

        return augmented_video_sample[:sync_len], label[:sync_len]
