from validation_dataset import TestValDataset
from torch.utils.data import DataLoader
from networks.video_network import VideoNet
import torch
from losses import binary_cross_entropy, f1_accuracy_metrics_oncuda
import os
from utils import collate_many2many
from utils import MyLogger

# Parameters:
test_gpu_list = [0]
model_to_evaluate = "saved_models/best_model.pt"
TEST_STREAM_TYPE = "rgb"  # for the optical flow this has to be changed to "of"

base_test_datadir = "data_dda"
norm_statistics = "video_train_statistics.npy"
test_video_path = "{}/video/test/".format(base_test_datadir)
test_clean_audio_path = "{}/clean_audio/test".format(base_test_datadir)

test_batch_size = 1
num_workers = 0

lstm_layers = 2
lstm_hidden_size = 1024
epsilon = 1e-8

############################################################
# End of configuration section
############################################################
my_test_logger = MyLogger("Test_")
speaker_folders = sorted([x for x in os.listdir(test_video_path)])
print("Test users list: ", speaker_folders)

test_model = VideoNet(lstm_layers=lstm_layers, lstm_hidden_size=lstm_hidden_size).cuda()
test_model = torch.nn.DataParallel(test_model, device_ids=test_gpu_list)

if model_to_evaluate.__contains__("_checkpoint"):
    print("Loading the checkpoint")
    test_model.load_state_dict(torch.load(model_to_evaluate)["model_state_dict"])
else:
    print("Loading the best model")
    test_model = torch.load(model_to_evaluate)

test_model.eval()  # turn on inference mode

with torch.no_grad():
    agregated_results_dict = {}
    for speaker in speaker_folders:
        video_paths_list_perspeaker = []
        audio_paths_list_perspeaker = []

        video_speaker_path = os.path.join(test_video_path, speaker)
        video_speaker_sentence_files = sorted([y for y in os.listdir(video_speaker_path)])

        audio_speaker_path = os.path.join(test_clean_audio_path, speaker, "straightcam")
        audio_speaker_sentence_files = sorted([y for y in os.listdir(audio_speaker_path)])

        for video_sentence_file in video_speaker_sentence_files:
            sentence_video_path = os.path.join(video_speaker_path, video_sentence_file)
            video_paths_list_perspeaker.append(sentence_video_path)

        for audio_sentence_file in audio_speaker_sentence_files:
            sentence_audio_path = os.path.join(audio_speaker_path, audio_sentence_file)
            audio_paths_list_perspeaker.append(sentence_audio_path)

        print(f"Speaker {speaker} has video list {video_paths_list_perspeaker}")
        print(f"Speaker {speaker} has audio list {audio_paths_list_perspeaker}")

        video_test_dataset = TestValDataset(video_paths=video_paths_list_perspeaker, label_paths=audio_paths_list_perspeaker, train_statistics=norm_statistics, streamtype=TEST_STREAM_TYPE)

        test_loader = DataLoader(video_test_dataset,
                                 batch_size=test_batch_size, shuffle=False,
                                 collate_fn=collate_many2many,
                                 num_workers=num_workers, pin_memory=False,
                                 )
        test_len = len(test_loader)

        total_test_f1 = 0
        total_test_acc = 0
        total_test_precision = 0
        total_test_recall = 0
        total_test_tnr = 0
        total_test_loss = 0
        for test_batch_count, test_batch_data in enumerate(test_loader):
            test_lengths, test_video_sequence, test_target_label_vad = test_batch_data
            test_lengths = test_lengths.cuda()
            test_video_sequence = test_video_sequence.cuda()
            test_target_label_vad = test_target_label_vad.cuda()
            test_y_hat_soft = test_model(test_video_sequence, test_lengths)

            test_loss = 0
            for (test_length, test_soft_prob, test_target) in zip(test_lengths, test_y_hat_soft, test_target_label_vad):
                test_loss += binary_cross_entropy(test_soft_prob[:test_length], test_target[:test_length], epsilon)

            total_test_loss += test_loss

            test_y_hat_hard = (torch.sigmoid(test_y_hat_soft) > 0.5).type(torch.CharTensor)

            test_batch_f1, test_batch_accuracy, test_batch_precision, test_batch_recall, test_batch_tnr = 0., 0., 0., 0., 0.
            for (test_length, test_pred, test_target) in zip(test_lengths, test_y_hat_hard, test_target_label_vad):
                test_f1, test_accuracy, test_precision, test_recall, test_tnr = f1_accuracy_metrics_oncuda(
                    y_hat_hard=torch.flatten(test_pred[:test_length]), y=torch.flatten(test_target[:test_length]))
                test_batch_f1 += test_f1
                test_batch_accuracy += test_accuracy
                test_batch_precision += test_precision
                test_batch_recall += test_recall
                test_batch_tnr += test_tnr

            test_batch_f1 /= len(test_lengths)
            test_batch_accuracy /= len(test_lengths)
            test_batch_precision /= len(test_lengths)
            test_batch_recall /= len(test_lengths)
            test_batch_tnr /= len(test_lengths)

            total_test_f1 += test_batch_f1
            total_test_acc += test_batch_accuracy
            total_test_precision += test_batch_precision
            total_test_recall += test_batch_recall
            total_test_tnr += test_batch_tnr

        print("Speaker {}: Avg test F1={:.5f}, avg test Acc={:.5f}, avg test Prec={:.5f}, avg test Rec={:.5f}, avg test TNR={:.5f}".format(speaker,
               total_test_f1 / test_len,
               total_test_acc / test_len,
               total_test_precision / test_len,
               total_test_recall / test_len,
               total_test_tnr / test_len
               ))

        agregated_results_dict[speaker] = [total_test_f1 / test_len,
                                           total_test_acc / test_len,
                                           total_test_precision / test_len,
                                           total_test_recall / test_len,
                                           total_test_tnr / test_len,
                                           1 - total_test_tnr / test_len]

        print("############################################################################")

    print("Final summary of results on the test split:")

    avg_F1_all_speakers = 0
    avg_acc_all_speakers = 0
    avg_prec_all_speakers = 0
    avg_rec_all_speakers = 0
    avg_tnr_all_speakers = 0
    avg_fpr_all_speakers = 0
    for key, val in agregated_results_dict.items():
        per_speaker_info = "Speaker {} has F1={:.5f}, Acc={:.5f}, Prec={:.5f}, Recall={:.5f}, TNR={:.5f}, FPR={:.5f}".format(key, val[0], val[1], val[2], val[3], val[4], val[5])
        print(per_speaker_info)
        my_test_logger.log(per_speaker_info)
        avg_F1_all_speakers += val[0]
        avg_acc_all_speakers += val[1]
        avg_prec_all_speakers += val[2]
        avg_rec_all_speakers += val[3]
        avg_tnr_all_speakers += val[4]
        avg_fpr_all_speakers += val[5]

    info_line1 = "\n################# Average results over all speakers #################\n"
    print(info_line1)
    my_test_logger.log(info_line1)

    info_line2 = "Avg F1  all speakers: {:.5f}".format(avg_F1_all_speakers.item()/len(agregated_results_dict.keys()))
    print(info_line2)
    my_test_logger.log(info_line2)

    info_line3 = "Avg Acc all speakers: {:.5f}".format(avg_acc_all_speakers.item()/len(agregated_results_dict.keys()))
    print(info_line3)
    my_test_logger.log(info_line3)

    info_line4 = "Avg Pre all speakers: {:.5f}".format(avg_prec_all_speakers.item()/len(agregated_results_dict.keys()))
    print(info_line4)
    my_test_logger.log(info_line4)

    info_line5 = "Avg Rec all speakers: {:.5f}".format(avg_rec_all_speakers.item()/len(agregated_results_dict.keys()))
    print(info_line5)
    my_test_logger.log(info_line5)

    info_line6 = "Avg TNR all speakers: {:.5f}".format(avg_tnr_all_speakers.item()/len(agregated_results_dict.keys()))
    print(info_line6)
    my_test_logger.log(info_line6)

    info_line7 = "Avg FPR all speakers: {:.5f}".format(avg_fpr_all_speakers.item()/len(agregated_results_dict.keys()))
    print(info_line7)
    my_test_logger.log(info_line7)

