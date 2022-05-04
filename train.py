import argparse
import random
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Preprocessing_data import DataLoader_train, DataLoader_val
import os, glob
from model import model
import numpy as np
from utils import *
from tqdm import tqdm
import pandas as pd
from torch.autograd import Variable
import h5py

### Training settings
parser = argparse.ArgumentParser(description="Pytorch ensemble sonar classification")
parser.add_argument("--batchsize", type=int, default=64, help="Training batch size")
parser.add_argument("--num_iter_toprint", type=int, default=32, help="Training patch size")
parser.add_argument("--patchsize", type=int, default=512, help="Training patch size")

### input Ensemble approach
parser.add_argument("--path_data", default="./H5/Train and Validation/72_samples_Train_and_Validation_16_slices.h5", type=str, help="Training datapath")
parser.add_argument("--data_length", type=int, default=16, help="The number of slices")

parser.add_argument("--save_path_result", default="/cvdata2/trung/Course/Final Project - PRML/Code/Python implementation/Result/Train 2", type=str, help="Result folder")
parser.add_argument("--save_path_csv", default="/cvdata2/trung/Course/Final Project - PRML/Code/Python implementation/Result/Train 2/results_train_2_densenet.csv", type=str, help="Save CSV result")
parser.add_argument("--save_model_path", default="/cvdata2/trung/Course/Final Project - PRML/Code/Python implementation/Checkpoint/Train 2", type=str, help="Save model path")

parser.add_argument("--path_data_te", default="", type=str, help="Training datapath")
parser.add_argument("--nEpochs", type=int, default=4, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate, Default=0.1")
parser.add_argument("--lr_reduce", type=float, default=0.5, help="rate of reduction of learning rate, Default=0.4")
parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs to train for")
parser.add_argument("--num_out", type=int, default=2, help="how many classes in outputs?")

parser.add_argument("--block_config", type=int, default=(8,12,8,8), help="Training patch size")
parser.add_argument("--step", type=int, default=10, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default=5")
parser.add_argument("--cuda", type=str, default='0')
parser.add_argument("--start_epoch", default=1, type = int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--resume", default="./model/dense_cbam_cmv_BloodOrCSF_onlyPIH_ct_2D3D_32_fold5of5/model_epoch_40000.pth" , type=str, help="Path to checkpoint, Default=None")
parser.add_argument("--clip", type=float, default=0.01, help="Clipping Gradients, Default=0.01")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default=1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default=0.9")
parser.add_argument("--weight_decay", "--wd", default=1e-6, type=float, help="Weight decay, Default=1e-4")
parser.add_argument("--pretrained", default="", type=str, help='path to pretrained model_files, Default=None')
parser.add_argument("--ID", default="", type=str, help='ID for training')

def main():
    global opt, model
    opt = parser.parse_args()
    data_length = opt.data_length
    batchsize = opt.batchsize
    lr = opt.lr
    momentum = opt.momentum
    threads = opt.threads
    weight_decay = opt.weight_decay
    ID = opt.ID
    path_data = opt.path_data
    num_epochs = opt.num_epochs
    save_model_path = opt.save_model_path
    start_epoch = opt.start_epoch

    device = get_default_device()
    model = to_device(model(), device)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=opt.weight_decay, momentum=opt.momentum)
    train_loader = DeviceDataLoader(DataLoader_train, device)
    validation_loader = DeviceDataLoader(DataLoader_val, device)

    val_acc_epoch = []
    train_acc_epoch = []
    ave_loss = []

    for epoch in range(start_epoch, num_epochs + 1):
        val_acc = check_accuracy(validation_loader, model)
        val_acc_epoch.append(val_acc)
        train_acc= check_accuracy(train_loader, model)
        train_acc_epoch.append(train_acc)
        print("==> Accuracy for validation in epoch {} is: {:.4f}".format(epoch-1,
                                                val_acc))
        print("==> Accuracy for train in epoch {} is: {:.4f}".format(epoch-1,
                                                train_acc))
        average_loss = train(train_loader, optimizer, model, epoch, criterion, opt)
        ave_loss.append(average_loss)
        plot_check(val_acc_epoch, train_acc_epoch, ave_loss, epoch)



def plot_check(val_acc_epoch, train_acc_epoch, average_loss, epoch):
    plt.figure()

    plt.subplot(2,1,1)
    plt.plot(val_acc_epoch, color = 'tab:red', label = 'Val Accuracy')
    plt.plot(train_acc_epoch, color = 'tab:blue', label = 'Train Accuracy')
    plt.ylim([0, 100])

    plt.subplot(2,1,2)
    plt.plot(average_loss)
    plt.ylim([0, 0.8])

    plt.savefig(os.path.join(opt.save_path_result, 'Result_after_epoch{}'.format(epoch)), dpi=200)
    plt.close()

    df = pd.DataFrame({'epoch': epoch, 'Val Accuracy': val_acc_epoch, 'Train Accuracy': train_acc_epoch,
                       'Loss': average_loss})
    df.to_csv(opt.save_path_csv)
    if epoch%3 == 0:
        saveModel_per_epoch(epoch)

def saveModel_per_epoch(epoch):
    model_out_path = os.path.join(opt.save_model_path, "model_epoch_{}.pth".format(epoch))
    torch.save(model.state_dict(), model_out_path)

def adjust_learning_rate(epoch):
    lr = opt.lr * (opt.lr_reduce ** (epoch // opt.step))
    return lr

def check_accuracy(dataloader, model):
    model.eval()
    accuracy = 0.0
    total = 0.0
    sens = []
    spec = []

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
            # sensitivity = ((torch.sum(((predicted == 1) & (label == 1))) / torch.sum(label == 1))).float().cpu().numpy()
            # specificity = ((torch.sum(((predicted == 0) & (label == 0))) / torch.sum(label == 0))).float().cpu().numpy()
            # sens = np.append(sens, sensitivity)
            # spec = np.append(spec, specificity)

    # compute the metrics over all images
    accuracy = (100 * accuracy / total)
    # average_sens = np.average(sens)
    # average_spec = np.average(spec)
    return accuracy

def train(training_data_loader, optimizer, model, epoch,  criterion, opt):
    lr = adjust_learning_rate(epoch-1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        loss_tr = []
        input_data, label = batch
        label = label.view(-1)
        # form the tables of label and select data to train
        # input_data: N x 1 x H x W;
        # out: N x num_classes;
        # out is unnormalized
        out = model(input_data.cuda())
        loss = criterion(out, label.cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_tr.append(loss.cpu().detach().numpy())
        if iteration%opt.num_iter_toprint == 0:
             print("===> Epoch[{}]({}/{},lr:{:.8f}): loss:{:.6f}".format(epoch, iteration,
                                                    len(training_data_loader), lr, loss))
        average_loss = np.average(loss_tr)
    return average_loss

if __name__ == "__main__":
    main()