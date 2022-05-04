import argparse
import random
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Preprocessing_data import DataLoader_test
import os, glob
from model import model
import numpy as np
from utils import *
import pandas as pd
import sklearn.metrics
from sklearn.metrics import plot_precision_recall_curve
import h5py
import matplotlib.pyplot as plt
from matplotlib.image import imread
from PIL import Image
import os.path, sys
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms as tt
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np
import pylab as pl

### Testing settings
parser = argparse.ArgumentParser(description="Pytorch ensemble sonar classification")
parser.add_argument("--batchsize", type=int, default=299, help="Training batch size")
parser.add_argument("--num_iter_toprint", type=int, default=30, help="Training patch size")
parser.add_argument("--patchsize", type=int, default=512, help="Training patch size")

### input for test Ensemble model
parser.add_argument("--path_data", default="./H5/Test/72_samples_Test_32_slices.h5", type=str, help="Training datapath")
parser.add_argument("--save_model_path", default="/cvdata2/trung/Course/Final Project - PRML/Code/Python implementation/Checkpoint/Train 2 Densenet", type=str, help="Save model path")
parser.add_argument("--data_length", type=int, default=32, help="The number of slices")
parser.add_argument("--confusion_matrix", default="./Result/72 samples/32 Slices", type=str, help="Confusion matrix")

parser.add_argument("--nEpochs", type=int, default=75, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.00012, help="Learning Rate, Default=0.1")
parser.add_argument("--lr_reduce", type=float, default=0.5, help="rate of reduction of learning rate, Default=0.4")
parser.add_argument("--num_out", type=int, default=2, help="how many classes in outputs?")
parser.add_argument("--block_config", type=int, default=(8,12,8,8), help="Training patch size")
parser.add_argument("--step", type=int, default=10, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default=5")
parser.add_argument("--cuda", type=str, default='0')
parser.add_argument("--resume", default="./model/dense_cbam_cmv_BloodOrCSF_onlyPIH_ct_2D3D_32_fold5of5/model_epoch_40000.pth" , type=str, help="Path to checkpoint, Default=None")
parser.add_argument("--start_epoch", default=1, type = int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.01, help="Clipping Gradients, Default=0.01")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default=1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default=0.9")
parser.add_argument("--weight_decay", "--wd", default=1e-6, type=float, help="Weight decay, Default=1e-4")
parser.add_argument("--pretrained", default="", type=str, help='path to pretrained model_files, Default=None')
parser.add_argument("--ID", default="", type=str, help='ID for training')

def main():
    global opt, model

    opt = parser.parse_args()
    save_model_path = opt.save_model_path
    path_data = opt.path_data
    confusion_matrix = opt.confusion_matrix

    ## Load model
    device = get_default_device()
    model = to_device(model(), device)

    for checkpoint_save_path in glob.glob(save_model_path + '/*'):
        print(checkpoint_save_path)
        model.load_state_dict(torch.load(checkpoint_save_path))
        test_acc = check_accuracy(DataLoader_test, model)
        print("==> Accuracy for test set of {} is: {:.4f}".format(checkpoint_save_path, test_acc))


def check_accuracy(dataloader, model):
    model.eval()
    accuracy = 0.0
    total = 0.0
    total_label = []
    total_predict = []

    with torch.no_grad():
        for data in dataloader:
            images, label = data
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            label = label.view(-1)
            accuracy += (predicted == label).sum().item()
            total_label.extend(label.cpu())
            total_predict.extend(predicted.cpu())
            # sensitivity = ((torch.sum(((predicted == 1) & (label == 1))) / torch.sum(label == 1))).float().cpu().numpy()
            # specificity = ((torch.sum(((predicted == 0) & (label == 0))) / torch.sum(label == 0))).float().cpu().numpy()
            # sens = np.append(sens, sensitivity)
            # spec = np.append(spec, specificity)

    # compute the metrics over all images
    accuracy = (100 * accuracy / total)
    cm = confusion_matrix(total_label, total_predict)
    print(cm)
    # average_sens = np.average(sens)
    # average_spec = np.average(spec)
    return accuracy

if __name__ == "__main__":
    Test_path = '/cvdata2/trung/Course/Final Project - PRML/data/wallpapers/Test'

    preprocess = tt.Compose([
        tt.CenterCrop(256),
        tt.ToTensor()
    ])

    Test_dataset = ImageFolder(Test_path, transform=preprocess)
    device = get_default_device()
    batch_size = 32
    DataLoader_test = DataLoader(dataset=Test_dataset, batch_size=batch_size, shuffle=True,
                                 pin_memory=True)
    DataLoader_test = DeviceDataLoader(DataLoader_test, device)
    main()