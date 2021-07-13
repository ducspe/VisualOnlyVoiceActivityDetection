import torch
import os
from torch.utils.data import DataLoader

from validation_dataset import TestValDataset
from networks.video_network import VideoNet
from losses import binary_cross_entropy, f1_accuracy_metrics_oncuda
from utils import collate_many2many
from utils import MyLogger

# Parameters:
overall_threshold = 0.5  # threshold after fusion

rgb_threshold = 0.5
of_threshold = 0.5


test_gpu_list = [0]  # change here
rgb_model_to_evaluate = "saved_models/best_model.pt"  # change here
of_model_to_evaluate = "saved_models/best_model.pt"  # change here

base_test_datadir = "data_dda"  # change here

rgb_norm_statistics = "video_train_statistics.npy"  # change here to for ex: rgb_video_train_statistics.npy
rgb_test_video_path = "{}/video/test/".format(base_test_datadir)  # change here

of_norm_statistics = "video_train_statistics.npy"  # change here to for ex: of_video_train_statistics.npy
of_test_video_path = "{}/video/test".format(base_test_datadir)  # change here

test_clean_audio_path = "{}/clean_audio/test".format(base_test_datadir)

test_batch_size = 1  # change here
num_workers = 0

lstm_layers = 2
lstm_hidden_size = 1024
epsilon = 1e-8


############################################################
# End of configuration section
############################################################

# Create a logger object for the test procedure:
my_test_logger = MyLogger("Test_")

speaker_folders = sorted([x for x in os.listdir(rgb_test_video_path)])
print("Test users list: ", speaker_folders)

# RGB model initialization:
rgb_test_model = VideoNet(lstm_layers=lstm_layers, lstm_hidden_size=lstm_hidden_size).cuda()
rgb_test_model = torch.nn.DataParallel(rgb_test_model, device_ids=test_gpu_list)

if rgb_model_to_evaluate.__contains__("_checkpoint"):
    print("Loading the RGB model checkpoint")
    rgb_test_model.load_state_dict(torch.load(rgb_model_to_evaluate)["model_state_dict"])
else:
    print("Loading the best RGB model")
    rgb_test_model = torch.load(rgb_model_to_evaluate)

rgb_test_model.eval()  # turn on inference mode for the RGB branch


# Optical Flow model initialization:
of_test_model = VideoNet(lstm_layers=lstm_layers, lstm_hidden_size=lstm_hidden_size).cuda()
of_test_model = torch.nn.DataParallel(of_test_model, device_ids=test_gpu_list)

if of_model_to_evaluate.__contains__("_checkpoint"):
    print("Loading the Optical Flow model checkpoint")
    of_test_model.load_state_dict(torch.load(of_model_to_evaluate)["model_state_dict"])
else:
    print("Loading the best Optical Flow model")
    of_test_model = torch.load(of_model_to_evaluate)

of_test_model.eval()  # turn on inference mode for the optical flow branch


# Start test prediction process:
with torch.no_grad():
    rgb_aggregated_results_dict = {}
    of_aggregated_results_dict = {}
    fused_aggregated_results_dict = {}
    for speaker_count, speaker in enumerate(speaker_folders):
        rgb_video_paths_list_perspeaker = []
        of_video_paths_list_perspeaker = []
        audio_paths_list_perspeaker = []

        rgb_video_speaker_path = os.path.join(rgb_test_video_path, speaker)
        of_video_speaker_path = os.path.join(of_test_video_path, speaker)
        audio_speaker_path = os.path.join(test_clean_audio_path, speaker, "straightcam")

        rgb_video_speaker_sentence_files = sorted([y for y in os.listdir(rgb_video_speaker_path)])
        of_video_speaker_sentence_files = sorted([y for y in os.listdir(of_video_speaker_path)])
        audio_speaker_sentence_files = sorted([y for y in os.listdir(audio_speaker_path)])

        for rgb_video_sentence_file in rgb_video_speaker_sentence_files:
            rgb_sentence_video_path = os.path.join(rgb_video_speaker_path, rgb_video_sentence_file)
            rgb_video_paths_list_perspeaker.append(rgb_sentence_video_path)

        for of_video_sentence_file in of_video_speaker_sentence_files:
            of_sentence_video_path = os.path.join(of_video_speaker_path, of_video_sentence_file)
            of_video_paths_list_perspeaker.append(of_sentence_video_path)

        for audio_sentence_file in audio_speaker_sentence_files:
            sentence_audio_path = os.path.join(audio_speaker_path, audio_sentence_file)
            audio_paths_list_perspeaker.append(sentence_audio_path)

        print(f"Speaker {speaker} has RGB video list with length {len(rgb_video_paths_list_perspeaker)}")
        print(f"Speaker {speaker} has audio list with length {len(audio_paths_list_perspeaker)}")
        print(f"Speaker {speaker} has Optical Flow list with length {len(of_video_paths_list_perspeaker)}")

        # rgb dataset loader
        rgb_video_test_dataset = TestValDataset(video_paths=rgb_video_paths_list_perspeaker,
                                                label_paths=audio_paths_list_perspeaker,
                                                train_statistics=rgb_norm_statistics, streamtype="rgb")

        rgb_test_loader = DataLoader(rgb_video_test_dataset,
                                     batch_size=test_batch_size, shuffle=False,
                                     collate_fn=collate_many2many,
                                     num_workers=num_workers, pin_memory=False,
                                     )

        rgb_test_len = len(rgb_test_loader)  # this will be used for the fusion related code as well

        # optical flow dataset loader
        of_video_test_dataset = TestValDataset(video_paths=of_video_paths_list_perspeaker,
                                               label_paths=audio_paths_list_perspeaker,
                                               train_statistics=of_norm_statistics, streamtype="of")

        of_test_loader = DataLoader(of_video_test_dataset,
                                    batch_size=test_batch_size, shuffle=False,
                                    collate_fn=collate_many2many,
                                    num_workers=num_workers, pin_memory=False,
                                    )

        of_test_len = len(of_test_loader)

        # rgb metrics initialization:
        rgb_total_test_f1 = 0
        rgb_total_test_acc = 0
        rgb_total_test_precision = 0
        rgb_total_test_recall = 0
        rgb_total_test_tnr = 0
        rgb_total_test_loss = 0

        # of metrics initialization:
        of_total_test_f1 = 0
        of_total_test_acc = 0
        of_total_test_precision = 0
        of_total_test_recall = 0
        of_total_test_tnr = 0
        of_total_test_loss = 0

        # fusion metrics initialization
        fused_total_test_f1 = 0
        fused_total_test_acc = 0
        fused_total_test_precision = 0
        fused_total_test_recall = 0
        fused_total_test_tnr = 0
        fused_total_test_loss = 0
        for rgb_test_batch_count, rgb_and_of_test_batch_data in enumerate(zip(rgb_test_loader, of_test_loader)):
            rgb_test_lengths, rgb_test_video_sequence, rgb_test_target_label_vad = rgb_and_of_test_batch_data[0]
            rgb_test_lengths = rgb_test_lengths.cuda()
            rgb_test_video_sequence = rgb_test_video_sequence.cuda()
            rgb_test_target_label_vad = rgb_test_target_label_vad.cuda()
            rgb_test_y_hat_soft = rgb_test_model(rgb_test_video_sequence, rgb_test_lengths)

            rgb_test_loss = 0
            for (rgb_test_length, rgb_test_soft_prob, rgb_test_target) in zip(rgb_test_lengths, rgb_test_y_hat_soft, rgb_test_target_label_vad):
                rgb_test_loss += binary_cross_entropy(rgb_test_soft_prob[:rgb_test_length], rgb_test_target[:rgb_test_length], epsilon)

            rgb_total_test_loss += rgb_test_loss

            rgb_test_y_hat_hard = (torch.sigmoid(rgb_test_y_hat_soft) > rgb_threshold).type(torch.CharTensor)

            rgb_test_batch_f1, rgb_test_batch_accuracy, rgb_test_batch_precision, rgb_test_batch_recall, rgb_test_batch_tnr = 0., 0., 0., 0., 0.
            for (rgb_test_length, rgb_test_pred, rgb_test_target) in zip(rgb_test_lengths, rgb_test_y_hat_hard, rgb_test_target_label_vad):
                rgb_test_f1, rgb_test_accuracy, rgb_test_precision, rgb_test_recall, rgb_test_tnr = f1_accuracy_metrics_oncuda(
                    y_hat_hard=torch.flatten(rgb_test_pred[:rgb_test_length]), y=torch.flatten(rgb_test_target[:rgb_test_length]))
                rgb_test_batch_f1 += rgb_test_f1
                rgb_test_batch_accuracy += rgb_test_accuracy
                rgb_test_batch_precision += rgb_test_precision
                rgb_test_batch_recall += rgb_test_recall
                rgb_test_batch_tnr += rgb_test_tnr

            rgb_test_batch_f1 /= len(rgb_test_lengths)
            rgb_test_batch_accuracy /= len(rgb_test_lengths)
            rgb_test_batch_precision /= len(rgb_test_lengths)
            rgb_test_batch_recall /= len(rgb_test_lengths)
            rgb_test_batch_tnr /= len(rgb_test_lengths)

            rgb_total_test_f1 += rgb_test_batch_f1
            rgb_total_test_acc += rgb_test_batch_accuracy
            rgb_total_test_precision += rgb_test_batch_precision
            rgb_total_test_recall += rgb_test_batch_recall
            rgb_total_test_tnr += rgb_test_batch_tnr

            ############################################################

            of_test_lengths, of_test_video_sequence, of_test_target_label_vad = rgb_and_of_test_batch_data[1]
            of_test_lengths = of_test_lengths.cuda()
            of_test_video_sequence = of_test_video_sequence.cuda()
            of_test_target_label_vad = of_test_target_label_vad.cuda()
            of_test_y_hat_soft = of_test_model(of_test_video_sequence, of_test_lengths)

            of_test_loss = 0
            for (of_test_length, of_test_soft_prob, of_test_target) in zip(of_test_lengths, of_test_y_hat_soft,
                                                                           of_test_target_label_vad):
                of_test_loss += binary_cross_entropy(of_test_soft_prob[:of_test_length],
                                                     of_test_target[:of_test_length], epsilon)

            of_total_test_loss += of_test_loss

            of_test_y_hat_hard = (torch.sigmoid(of_test_y_hat_soft) > of_threshold).type(torch.CharTensor)
            
            overall_probability = torch.sigmoid(rgb_test_y_hat_soft) + torch.sigmoid(of_test_y_hat_soft)

            of_test_batch_f1, of_test_batch_accuracy, of_test_batch_precision, of_test_batch_recall, of_test_batch_tnr = 0., 0., 0., 0., 0.
            for (of_test_length, of_test_pred, of_test_target) in zip(of_test_lengths, of_test_y_hat_hard,
                                                                      of_test_target_label_vad):
                of_test_f1, of_test_accuracy, of_test_precision, of_test_recall, of_test_tnr = f1_accuracy_metrics_oncuda(
                    y_hat_hard=torch.flatten(of_test_pred[:of_test_length]),
                    y=torch.flatten(of_test_target[:of_test_length]))
                of_test_batch_f1 += of_test_f1
                of_test_batch_accuracy += of_test_accuracy
                of_test_batch_precision += of_test_precision
                of_test_batch_recall += of_test_recall
                of_test_batch_tnr += of_test_tnr

            of_test_batch_f1 /= len(of_test_lengths)
            of_test_batch_accuracy /= len(of_test_lengths)
            of_test_batch_precision /= len(of_test_lengths)
            of_test_batch_recall /= len(of_test_lengths)
            of_test_batch_tnr /= len(of_test_lengths)

            of_total_test_f1 += of_test_batch_f1
            of_total_test_acc += of_test_batch_accuracy
            of_total_test_precision += of_test_batch_precision
            of_total_test_recall += of_test_batch_recall
            of_total_test_tnr += of_test_batch_tnr


            ##############################################################
            # logical/bitwise operations can also be used, but they result in a tradeoff between TNR and TPR, ex: one goes down by 5%, and the other one goes up by 5%
            # fused_test_y_hat_hard = torch.logical_and(rgb_test_y_hat_hard, of_test_y_hat_hard).type(torch.CharTensor)
            # better to use soft fusion/soft decisions:
            fused_test_y_hat_hard = (torch.sigmoid(overall_probability) > overall_threshold).type(torch.CharTensor)

            fused_test_batch_f1, fused_test_batch_accuracy, fused_test_batch_precision, fused_test_batch_recall, fused_test_batch_tnr = 0., 0., 0., 0., 0.

            for (fused_test_length, fused_test_pred, fused_test_target) in zip(rgb_test_lengths,
                                                                               fused_test_y_hat_hard,
                                                                               rgb_test_target_label_vad):
                fused_test_f1, fused_test_accuracy, fused_test_precision, fused_test_recall, fused_test_tnr = f1_accuracy_metrics_oncuda(
                    y_hat_hard=torch.flatten(fused_test_pred[:fused_test_length]),
                    y=torch.flatten(fused_test_target[:fused_test_length]))
                fused_test_batch_f1 += fused_test_f1
                fused_test_batch_accuracy += fused_test_accuracy
                fused_test_batch_precision += fused_test_precision
                fused_test_batch_recall += fused_test_recall
                fused_test_batch_tnr += fused_test_tnr

            fused_test_batch_f1 /= len(rgb_test_lengths)
            fused_test_batch_accuracy /= len(rgb_test_lengths)
            fused_test_batch_precision /= len(rgb_test_lengths)
            fused_test_batch_recall /= len(rgb_test_lengths)
            fused_test_batch_tnr /= len(rgb_test_lengths)

            fused_total_test_f1 += fused_test_batch_f1
            fused_total_test_acc += fused_test_batch_accuracy
            fused_total_test_precision += fused_test_batch_precision
            fused_total_test_recall += fused_test_batch_recall
            fused_total_test_tnr += fused_test_batch_tnr

        print(
            "[RGB] Speaker {}: Avg test F1={:.5f}, avg test Acc={:.5f}, avg test Prec={:.5f}, avg test Rec={:.5f}, avg test TNR={:.5f}".format(
                speaker,
                rgb_total_test_f1 / rgb_test_len,
                rgb_total_test_acc / rgb_test_len,
                rgb_total_test_precision / rgb_test_len,
                rgb_total_test_recall / rgb_test_len,
                rgb_total_test_tnr / rgb_test_len
            ))

        rgb_aggregated_results_dict[speaker] = [rgb_total_test_f1 / rgb_test_len,
                                                rgb_total_test_acc / rgb_test_len,
                                                rgb_total_test_precision / rgb_test_len,
                                                rgb_total_test_recall / rgb_test_len,
                                                rgb_total_test_tnr / rgb_test_len,
                                                1 - rgb_total_test_tnr / rgb_test_len]

        print(
            "[OF] Speaker {}: Avg test F1={:.5f}, avg test Acc={:.5f}, avg test Prec={:.5f}, avg test Rec={:.5f}, avg test TNR={:.5f}".format(
                speaker,
                of_total_test_f1 / of_test_len,
                of_total_test_acc / of_test_len,
                of_total_test_precision / of_test_len,
                of_total_test_recall / of_test_len,
                of_total_test_tnr / of_test_len
            ))

        of_aggregated_results_dict[speaker] = [of_total_test_f1 / of_test_len,
                                               of_total_test_acc / of_test_len,
                                               of_total_test_precision / of_test_len,
                                               of_total_test_recall / of_test_len,
                                               of_total_test_tnr / of_test_len,
                                               1 - of_total_test_tnr / of_test_len]

        print(
            "[FUSED] Speaker {}: Avg test F1={:.5f}, avg test Acc={:.5f}, avg test Prec={:.5f}, avg test Rec={:.5f}, avg test TNR={:.5f}".format(
                speaker,
                fused_total_test_f1 / rgb_test_len,
                fused_total_test_acc / rgb_test_len,
                fused_total_test_precision / rgb_test_len,
                fused_total_test_recall / rgb_test_len,
                fused_total_test_tnr / rgb_test_len
            ))

        fused_aggregated_results_dict[speaker] = [fused_total_test_f1 / rgb_test_len,
                                                  fused_total_test_acc / rgb_test_len,
                                                  fused_total_test_precision / rgb_test_len,
                                                  fused_total_test_recall / rgb_test_len,
                                                  fused_total_test_tnr / rgb_test_len,
                                                  1 - fused_total_test_tnr / rgb_test_len]


        print(f"####################### End of processing for speaker {speaker} ##################################")


    ####### Printing average RGB results:
    print("\nFinal summary of RGB results on the test split:")

    rgb_avg_F1_all_speakers = 0
    rgb_avg_acc_all_speakers = 0
    rgb_avg_prec_all_speakers = 0
    rgb_avg_rec_all_speakers = 0
    rgb_avg_tnr_all_speakers = 0
    rgb_avg_fpr_all_speakers = 0
    for rgb_key, rgb_val in rgb_aggregated_results_dict.items():
        rgb_per_speaker_info = "Speaker {} has F1={:.5f}, Acc={:.5f}, Prec={:.5f}, Recall={:.5f}, TNR={:.5f}, FPR={:.5f}".format(rgb_key, rgb_val[0], rgb_val[1], rgb_val[2], rgb_val[3], rgb_val[4], rgb_val[5])
        print(rgb_per_speaker_info)
        my_test_logger.log(rgb_per_speaker_info)
        rgb_avg_F1_all_speakers += rgb_val[0]
        rgb_avg_acc_all_speakers += rgb_val[1]
        rgb_avg_prec_all_speakers += rgb_val[2]
        rgb_avg_rec_all_speakers += rgb_val[3]
        rgb_avg_tnr_all_speakers += rgb_val[4]
        rgb_avg_fpr_all_speakers += rgb_val[5]

    rgb_info_line1 = "\n################# Average RGB results over all speakers #################\n"
    print(rgb_info_line1)
    my_test_logger.log(rgb_info_line1)

    rgb_info_line2 = "Avg F1  all speakers: {:.5f}".format(rgb_avg_F1_all_speakers.item() / len(rgb_aggregated_results_dict.keys()))
    print(rgb_info_line2)
    my_test_logger.log(rgb_info_line2)

    rgb_info_line3 = "Avg Acc all speakers: {:.5f}".format(rgb_avg_acc_all_speakers.item() / len(rgb_aggregated_results_dict.keys()))
    print(rgb_info_line3)
    my_test_logger.log(rgb_info_line3)

    rgb_info_line4 = "Avg Pre all speakers: {:.5f}".format(rgb_avg_prec_all_speakers.item() / len(rgb_aggregated_results_dict.keys()))
    print(rgb_info_line4)
    my_test_logger.log(rgb_info_line4)

    rgb_info_line5 = "Avg Rec all speakers: {:.5f}".format(rgb_avg_rec_all_speakers.item() / len(rgb_aggregated_results_dict.keys()))
    print(rgb_info_line5)
    my_test_logger.log(rgb_info_line5)

    rgb_info_line6 = "Avg TNR all speakers: {:.5f}".format(rgb_avg_tnr_all_speakers.item() / len(rgb_aggregated_results_dict.keys()))
    print(rgb_info_line6)
    my_test_logger.log(rgb_info_line6)

    rgb_info_line7 = "Avg FPR all speakers: {:.5f}".format(rgb_avg_fpr_all_speakers.item() / len(rgb_aggregated_results_dict.keys()))
    print(rgb_info_line7)
    my_test_logger.log(rgb_info_line7)

    ####### Printing average OF results:
    print("\nFinal summary of OF results on the test split:")

    of_avg_F1_all_speakers = 0
    of_avg_acc_all_speakers = 0
    of_avg_prec_all_speakers = 0
    of_avg_rec_all_speakers = 0
    of_avg_tnr_all_speakers = 0
    of_avg_fpr_all_speakers = 0
    for of_key, of_val in of_aggregated_results_dict.items():
        of_per_speaker_info = "Speaker {} has F1={:.5f}, Acc={:.5f}, Prec={:.5f}, Recall={:.5f}, TNR={:.5f}, FPR={:.5f}".format(
            of_key, of_val[0], of_val[1], of_val[2], of_val[3], of_val[4], of_val[5])
        print(of_per_speaker_info)
        my_test_logger.log(of_per_speaker_info)
        of_avg_F1_all_speakers += of_val[0]
        of_avg_acc_all_speakers += of_val[1]
        of_avg_prec_all_speakers += of_val[2]
        of_avg_rec_all_speakers += of_val[3]
        of_avg_tnr_all_speakers += of_val[4]
        of_avg_fpr_all_speakers += of_val[5]

    of_info_line1 = "\n################# Average OF results over all speakers #################\n"
    print(of_info_line1)
    my_test_logger.log(of_info_line1)

    of_info_line2 = "Avg F1  all speakers: {:.5f}".format(
        of_avg_F1_all_speakers.item() / len(of_aggregated_results_dict.keys()))
    print(of_info_line2)
    my_test_logger.log(of_info_line2)

    of_info_line3 = "Avg Acc all speakers: {:.5f}".format(
        of_avg_acc_all_speakers.item() / len(of_aggregated_results_dict.keys()))
    print(of_info_line3)
    my_test_logger.log(of_info_line3)

    of_info_line4 = "Avg Pre all speakers: {:.5f}".format(
        of_avg_prec_all_speakers.item() / len(of_aggregated_results_dict.keys()))
    print(of_info_line4)
    my_test_logger.log(of_info_line4)

    of_info_line5 = "Avg Rec all speakers: {:.5f}".format(
        of_avg_rec_all_speakers.item() / len(of_aggregated_results_dict.keys()))
    print(of_info_line5)
    my_test_logger.log(of_info_line5)

    of_info_line6 = "Avg TNR all speakers: {:.5f}".format(
        of_avg_tnr_all_speakers.item() / len(of_aggregated_results_dict.keys()))
    print(of_info_line6)
    my_test_logger.log(of_info_line6)

    of_info_line7 = "Avg FPR all speakers: {:.5f}".format(
        of_avg_fpr_all_speakers.item() / len(of_aggregated_results_dict.keys()))
    print(of_info_line7)
    my_test_logger.log(of_info_line7)

    ####### Printing average fused results:
    print("\nFinal summary of FUSED results on the test split:")

    fused_avg_F1_all_speakers = 0
    fused_avg_acc_all_speakers = 0
    fused_avg_prec_all_speakers = 0
    fused_avg_rec_all_speakers = 0
    fused_avg_tnr_all_speakers = 0
    fused_avg_fpr_all_speakers = 0
    for fused_key, fused_val in fused_aggregated_results_dict.items():
        fused_per_speaker_info = "Speaker {} has F1={:.5f}, Acc={:.5f}, Prec={:.5f}, Recall={:.5f}, TNR={:.5f}, FPR={:.5f}".format(
            fused_key, fused_val[0], fused_val[1], fused_val[2], fused_val[3], fused_val[4], fused_val[5])
        print(fused_per_speaker_info)
        my_test_logger.log(fused_per_speaker_info)
        fused_avg_F1_all_speakers += fused_val[0]
        fused_avg_acc_all_speakers += fused_val[1]
        fused_avg_prec_all_speakers += fused_val[2]
        fused_avg_rec_all_speakers += fused_val[3]
        fused_avg_tnr_all_speakers += fused_val[4]
        fused_avg_fpr_all_speakers += fused_val[5]

    fused_info_line1 = "\n################# Average FUSED results over all speakers #################\n"
    print(fused_info_line1)
    my_test_logger.log(fused_info_line1)

    fused_info_line2 = "Avg F1  all speakers: {:.5f}".format(
        fused_avg_F1_all_speakers.item() / len(fused_aggregated_results_dict.keys()))
    print(fused_info_line2)
    my_test_logger.log(fused_info_line2)

    fused_info_line3 = "Avg Acc all speakers: {:.5f}".format(
        fused_avg_acc_all_speakers.item() / len(fused_aggregated_results_dict.keys()))
    print(fused_info_line3)
    my_test_logger.log(fused_info_line3)

    fused_info_line4 = "Avg Pre all speakers: {:.5f}".format(
        fused_avg_prec_all_speakers.item() / len(fused_aggregated_results_dict.keys()))
    print(fused_info_line4)
    my_test_logger.log(fused_info_line4)

    fused_info_line5 = "Avg Rec all speakers: {:.5f}".format(
        fused_avg_rec_all_speakers.item() / len(fused_aggregated_results_dict.keys()))
    print(fused_info_line5)
    my_test_logger.log(fused_info_line5)

    fused_info_line6 = "Avg TNR all speakers: {:.5f}".format(
        fused_avg_tnr_all_speakers.item() / len(fused_aggregated_results_dict.keys()))
    print(fused_info_line6)
    my_test_logger.log(fused_info_line6)

    fused_info_line7 = "Avg FPR all speakers: {:.5f}".format(
        fused_avg_fpr_all_speakers.item() / len(fused_aggregated_results_dict.keys()))
    print(fused_info_line7)
    my_test_logger.log(fused_info_line7)
