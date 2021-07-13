from train_dataset import VideoDataset
from validation_dataset import TestValDataset
import torch
from networks.video_network import VideoNet
from torch.utils.data import DataLoader
import torch.nn as nn
from utils import create_video_paths_list, create_audio_paths_list, collate_many2many, MyLogger
from losses import binary_cross_entropy, f1_accuracy_metrics_oncuda
import os

# Define parameters:
continue_training_initialization_checkpoint = ""
gpu_list = [0]  # [0, 1] # IDs of GPUs to use for training
STREAM_TYPE = "rgb"  # If "of" is specified then the nan_to_num function will be called and the corresponding normalization
num_epochs = 2  # 200
batch_size = 1  # 16
val_batch_size = 1  # 16
checkpoint_save_freq = 1  # 2
num_workers = 0

lstm_layers = 2
lstm_hidden_size = 1024
learning_rate = 0.0001
epsilon = 1e-8

# Define model and dataset:
base_dir = "data_dda/"
norm_statistics = "video_train_statistics.npy"
video_train_path = "{}/video/train".format(base_dir)
video_validation_path = "{}/video/dev".format(base_dir)

clean_audio_train_path = "{}/clean_audio/train".format(base_dir)
clean_audio_validation_path = "{}/clean_audio/dev".format(base_dir)

###########################################################################################
# End of configs section
###########################################################################################

my_logger = MyLogger("Train_")
video_train_paths_list = create_video_paths_list(video_train_path)
print("Video train paths: ", video_train_paths_list)

video_validation_paths_list = create_video_paths_list(video_validation_path)
print("Video validation paths: ", video_validation_paths_list)

clean_audio_train_paths_list = create_audio_paths_list(clean_audio_train_path)
print("Audio train paths: ", clean_audio_train_paths_list)

clean_audio_validation_paths_list = create_audio_paths_list(clean_audio_validation_path)
print("Audio validation paths: ", clean_audio_validation_paths_list)

assert len(video_train_paths_list) == len(clean_audio_train_paths_list)
assert len(video_validation_paths_list) == len(clean_audio_validation_paths_list)

model = VideoNet(lstm_layers=lstm_layers, lstm_hidden_size=lstm_hidden_size).cuda()

if continue_training_initialization_checkpoint:
    model.load_state_dict(torch.load(continue_training_initialization_checkpoint)["model_state_dict"])

model = torch.nn.DataParallel(model, device_ids=gpu_list)

os.makedirs('saved_models/checkpoints', exist_ok=True)

video_train_dataset = VideoDataset(video_paths=video_train_paths_list, label_paths=clean_audio_train_paths_list, train_statistics=norm_statistics, streamtype=STREAM_TYPE)
video_validation_dataset = TestValDataset(video_paths=video_validation_paths_list, label_paths=clean_audio_validation_paths_list, train_statistics=norm_statistics, streamtype=STREAM_TYPE)

train_loader = DataLoader(
    video_train_dataset,
    batch_size=batch_size, shuffle=True,
    collate_fn=collate_many2many,
    num_workers=num_workers, pin_memory=False,
    drop_last=False
)

validation_loader = DataLoader(
    video_validation_dataset,
    batch_size=val_batch_size, shuffle=False,
    collate_fn=collate_many2many,
    num_workers=num_workers, pin_memory=False,
    drop_last=False
)

tr_len = len(train_loader)
val_len = len(validation_loader)

# criterion = nn.BCEWithLogitsLoss()
# criterion = binary_cross_entropy
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def validation_routine():
    model.eval()
    total_val_f1 = 0
    total_val_accuracy = 0
    total_val_precision = 0
    total_val_recall = 0
    total_val_tnr = 0
    total_val_loss = 0
    with torch.no_grad():
        for val_batch_count, val_batch_data in enumerate(validation_loader):
            val_lengths, val_video_sequence, val_target_label_vad = val_batch_data
            val_lengths = val_lengths.cuda()

            val_video_sequence = val_video_sequence.cuda()
            val_target_label_vad = val_target_label_vad.cuda()
            val_y_hat_soft = model(val_video_sequence, val_lengths)

            val_loss = 0
            for (val_length, val_soft_prob, val_target) in zip(val_lengths, val_y_hat_soft, val_target_label_vad):
                val_loss += binary_cross_entropy(val_soft_prob[:val_length], val_target[:val_length], epsilon)

            total_val_loss += val_loss

            val_y_hat_hard = (torch.sigmoid(val_y_hat_soft) > 0.5).type(torch.CharTensor)

            val_batch_f1, val_batch_accuracy, val_batch_precision, val_batch_recall, val_batch_tnr = 0., 0., 0., 0., 0.
            for (val_length, val_pred, val_target) in zip(val_lengths, val_y_hat_hard, val_target_label_vad):
                val_f1, val_accuracy, val_precision, val_recall, val_tnr = f1_accuracy_metrics_oncuda(y_hat_hard=torch.flatten(val_pred[:val_length]), y=torch.flatten(val_target[:val_length]))
                val_batch_f1 += val_f1
                val_batch_accuracy += val_accuracy
                val_batch_precision += val_precision
                val_batch_recall += val_recall
                val_batch_tnr += val_tnr

            val_batch_f1 /= len(val_lengths)
            val_batch_accuracy /= len(val_lengths)
            val_batch_precision /= len(val_lengths)
            val_batch_recall /= len(val_lengths)
            val_batch_tnr /= len(val_lengths)

            total_val_f1 += val_batch_f1
            total_val_accuracy += val_batch_accuracy
            total_val_precision += val_batch_precision
            total_val_recall += val_batch_recall
            total_val_tnr += val_batch_tnr

    return total_val_f1/val_len, total_val_accuracy/val_len, total_val_precision/val_len, total_val_recall/val_len, total_val_tnr/val_len, total_val_loss/val_len


val_f1_forbestval = 0
val_loss_forbestval = 1e6
epoch_forbestval = 0
val_acc_forbestval = 0
val_prec_forbestval = 0
val_rec_forbestval = 0
val_tnr_forbestval = 0
for epoch in range(num_epochs):
    model.train()
    total_f1 = 0
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_tnr = 0
    total_loss = 0
    for batch_count, batch_data in enumerate(train_loader):
        model.zero_grad()
        lengths, video_sequence, target_label_vad = batch_data
        lengths = lengths.cuda()
        # print("Lengths from the collate function: ", lengths)
        video_sequence = video_sequence.cuda()
        # print("video_mouth_roi sequence: ", video_sequence.shape)
        target_label_vad = target_label_vad.cuda()
        # print("target label: ", target_label_vad.shape)
        y_hat_soft = model(video_sequence, lengths)

        loss = 0
        for (length, soft_prob, target) in zip(lengths, y_hat_soft, target_label_vad):
            loss += binary_cross_entropy(soft_prob[:length], target[:length], epsilon)  # criterion(y_hat_soft, target_label_vad)

        total_loss += loss
        loss.backward()
        optimizer.step()

        # Calculate the evaluation metrics:
        y_hat_hard = (torch.sigmoid(y_hat_soft) > 0.5).type(torch.CharTensor)
        batch_f1, batch_accuracy, batch_precision, batch_recall, batch_tnr = 0., 0., 0., 0., 0.
        for (length, pred, target) in zip(lengths, y_hat_hard, target_label_vad):
            f1, accuracy, precision, recall, tnr = f1_accuracy_metrics_oncuda(y_hat_hard=torch.flatten(pred[:length]), y=torch.flatten(target[:length]))
            batch_f1 += f1
            batch_accuracy += accuracy
            batch_precision += precision
            batch_recall += recall
            batch_tnr += tnr

        batch_f1 /= len(lengths)
        batch_accuracy /= len(lengths)
        batch_precision /= len(lengths)
        batch_recall /= len(lengths)
        batch_tnr /= len(lengths)

        total_f1 += batch_f1
        total_accuracy += batch_accuracy
        total_precision += batch_precision
        total_recall += batch_recall
        total_tnr += batch_tnr
        info_line1 = "Epoch {}, batch {}: F1={:.5f}, Acc={:.5f}, Prec={:.5f}, Rec={:.5f}, TNR={:.5f}, Loss={:.5f}".format(epoch, batch_count, batch_f1, batch_accuracy, batch_precision, batch_recall, batch_tnr, loss)
        print(info_line1)

    info_line2 = "--------------START EVAL--------------------"
    print(info_line2)
    my_logger.log(info_line2)

    info_line3 = "Epoch {}: avg train F1={:.5f}, avg train Acc={:.5f}, avg train precision={:.5f}, avg train recall={:.5f}, avg train tnr={:.5f}, avg train loss={:.5f}".format(epoch, total_f1/tr_len, total_accuracy/tr_len, total_precision/tr_len, total_recall/tr_len, total_tnr/tr_len, total_loss/tr_len)
    print(info_line3)
    my_logger.log(info_line3, extra_newline=True)
    # Evaluate on the validation dataset:
    vr_f1, vr_acc, vr_prec, vr_rec, vr_tnr, vr_loss = validation_routine()  # vr: validation routine
    info_line4 = "Avg validation F1={:.5f}, avg validation Acc={:.5f}, avg validation Prec={:.5f}, avg validation Rec={:.5f}, TNR={:.5f}, avg validation loss={:.5f}".format(vr_f1, vr_acc, vr_prec, vr_rec, vr_tnr, vr_loss)
    print(info_line4)
    my_logger.log(info_line4, extra_newline=True)
    if vr_loss < val_loss_forbestval:
        val_f1_forbestval = vr_f1
        val_loss_forbestval = vr_loss
        val_acc_forbestval = vr_acc
        val_prec_forbestval = vr_prec
        val_rec_forbestval = vr_rec
        val_tnr_forbestval = vr_tnr
        epoch_forbestval = epoch
        torch.save(model, "saved_models/best_model.pt")

    info_line5 = "The best validation session had f1 = {:.5f} and got registered at epoch {}. Accuracy at that point was {:.5f}, Prec was {:.5f}, Rec was {:.5f} TNR was {:.5f}, and loss was {:.5f}".format(val_f1_forbestval, epoch_forbestval, val_acc_forbestval, val_prec_forbestval, val_rec_forbestval, val_tnr_forbestval, val_loss_forbestval)
    print(info_line5)
    my_logger.log(info_line5)

    info_line6 = "--------------END EVAL----------------------"
    print(info_line6)
    my_logger.log(info_line6)
    # Save checkpoints:
    if epoch % checkpoint_save_freq == 0:
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        checkpoint_name = "epoch{}_valf1_{:.4f}_valacc_{:.4f}_valloss_{:.4f}_trainloss_{:.4f}_checkpoint.pt".format(epoch, vr_f1, vr_acc, vr_loss, total_loss/tr_len)
        torch.save(state, "saved_models/checkpoints/{}".format(checkpoint_name))
