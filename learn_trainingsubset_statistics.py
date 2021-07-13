import utils
import numpy as np
#import cv2

epsilon = 1e-8


def analyze_dataset(video_paths):
    video_ram = []

    for video_pa in video_paths:
        preprocessed_file = np.load(video_pa)
        video_ram.append(preprocessed_file/255.0)

    all_videos = np.concatenate(video_ram, axis=0)
    max_before_normalization = all_videos.flatten().max()
    mean_before_normalization = all_videos.flatten().mean()
    std_before_normalization = all_videos.flatten().std()

    dataset_info_dict = {}
    dataset_info_dict['all_videos_normalization_max_before_normalization'] = max_before_normalization
    # Videos where divided by 255 so this max should be 1. Additionally in the preprocessing phase, only one channel was stored
    dataset_info_dict['all_videos_mean_before_normalization'] = mean_before_normalization
    dataset_info_dict['all_videos_std_before_normalization'] = std_before_normalization

    # all_videos /= max_before_normalization
    #
    # max_after_normalization = all_videos.flatten().max()
    # mean_after_normalization = all_videos.flatten().mean()
    # std_after_normalization = all_videos.flatten().std()
    # dataset_info_dict['all_videos_mean_after_normalization'] = mean_after_normalization
    # dataset_info_dict['all_videos_std_after_normalization'] = std_after_normalization

    with open('video_train_statistics.npy', 'wb') as f:
        np.save(f, dataset_info_dict)

    loaded_data = np.load("video_train_statistics.npy", allow_pickle=True)
    print(loaded_data.item())

    # Show one image to see if it looks ok:
    # cv2.imshow('image', all_videos[10])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    base_dir = "data_dda"
    video_train_path = "{}/video/train".format(base_dir)
    video_train_paths_list = utils.create_video_paths_list(video_train_path)
    print("Video train paths: ", video_train_paths_list)

    analyze_dataset(video_paths=video_train_paths_list)
