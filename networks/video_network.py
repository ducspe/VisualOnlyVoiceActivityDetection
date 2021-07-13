import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MyConvNet(nn.Module):
    def __init__(self):
        super(MyConvNet, self).__init__()  # Call the super constructor
        print("Instantiating MyConvNet")
        self.conv1 = nn.Conv2d(3, 64, 5, stride=2)  # input (channel) size is 3, output is 64, and kernel size is 5
        self.conv2 = nn.Conv2d(64, 64, 5, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 5, stride=2)
        self.fc1 = nn.Linear(64 * 5 * 5, 3200)
        self.fc2 = nn.Linear(3200, 1600)
        self.fc3 = nn.Linear(1600, 512)

    def forward(self, x):
        # print("Initial shape: ", x.shape)

        x = F.relu(self.conv1(x))  # first convolutional and pooling layer
        # print("After conv1: ", x.shape)

        x = F.relu(self.conv2(x))  # second convolutional and pooling layer
        # print("After conv2: ", x.shape)

        x = F.relu(self.conv3(x))
        # print("After conv3: ", x.shape)

        x = x.view(-1, 64 * 5 * 5)  # flatten

        x = F.relu(self.fc1(x))
        # print("After fc1: ", x.shape)

        x = F.relu(self.fc2(x))
        # print("After fc2: ", x.shape)

        x = F.sigmoid(self.fc3(x))

        return x


class My_Customized_Resnet_Model(nn.Module):
    def __init__(self):
        super(My_Customized_Resnet_Model, self).__init__()
        base_model = models.resnet18(pretrained=True)
        my_custom_model = nn.Sequential(*list(base_model.children())[:-1])

        print("Generating my custom resnet 18 model: ")

        # Freeze/Non-freeze code:
        child_counter = 0
        for child in my_custom_model.children():
            if child_counter < 7:
                print("Child ", child_counter, " was frozen")
                for param in child.parameters():
                    param.requires_grad = False
            elif child_counter == 7:
                subchildren_of_child_counter = 0
                for subchildren_of_child in child.children():
                    if subchildren_of_child_counter < 1:
                        for param in subchildren_of_child.parameters():
                            param.requires_grad = False
                        print("Sub-Child ", subchildren_of_child_counter, " of child ", child_counter, " was frozen")
                    else:
                        print("Sub-Child ", subchildren_of_child_counter, " of child ", child_counter, " was not frozen")
                    subchildren_of_child_counter += 1
            else:
                print("Child ", child_counter, " was not frozen")

            child_counter += 1

        self.visionpartlyfrozen = my_custom_model
        print("Finished generating the customized partly frozen resnet model")

    def forward(self, x):
        x = self.visionpartlyfrozen(x)
        return x


class VideoNet(nn.Module):

    def __init__(self, lstm_layers, lstm_hidden_size):
        super(VideoNet, self).__init__()

        # 1)
        resnet = models.resnet18(pretrained=False)  # set num_ftrs = 512

        # 2)
        #my_semifrozen_model = My_Customized_Resnet_Model()


        # 3)
        #myconvnet = MyConvNet()

        num_ftrs = 512

        self.lstm_input_size = num_ftrs
        self.lstm_layers = lstm_layers
        self.lstm_hidden_size = lstm_hidden_size

        self.features = nn.Sequential(
            # Uncomment if you chose 1) above
            *list(resnet.children())[:-1]  # drop the last FC layer

            # Uncomment if you chose 2) above
            #my_semifrozen_model

            # Uncomment if you chose 3) above
            #myconvnet
        )

        self.lstm_video = nn.LSTM(input_size=self.lstm_input_size,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=self.lstm_layers,
                            bidirectional=False)

        self.vad_video = nn.Linear(self.lstm_hidden_size, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, lengths):

        batch, frames, channels, height, width = x.size()

        # Reshape to (batch * seq_len, channels, height, width)
        x = x.view(batch*frames, channels, height, width)
        x = self.features(x).squeeze()  # output shape - Batch X Features X seq len
        #x = self.dropout(x)
        # Reshape to (batch, seq_len, Features)
        x = x.view(batch, frames, -1)

        total_length = x.size(1)  # to make unpacking work with DataParallel
        # print("Total length", total_length)
        x = pack_padded_sequence(x, lengths=lengths.cpu(), enforce_sorted=False, batch_first=True)
        # print(x)

        out, _ = self.lstm_video(x)  # The h and c are reset after every batch

        out, lens_unpacked = pad_packed_sequence(out, batch_first=True, total_length=total_length)
        # out = self.dropout(out)
        # print("out after many to many ", out.shape) #[batch size, nr of frames, 1024 (lstm-hidden-size)]
        out = self.vad_video(out)
        # print("out after vad_video ", out.shape)
        return out
