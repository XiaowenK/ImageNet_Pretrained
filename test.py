
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Author   : Xiaowen Ke
## Email    : xiaowen.herman@gmail.com
## Version  : v0.2.0
## Date     : 2021/04/01
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import cv2
import argparse
import numpy as np
import albumentations as A
import datetime
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from apex import amp
from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



parser = argparse.ArgumentParser(description='Testing on ImageNet Dataset.')
parser.add_argument('-data', default=r"./database", type=str, metavar='DATA', help='path to dataset')
parser.add_argument('-txt', default=r"./labels", type=str, metavar='TXT', help='path to txt files')
parser.add_argument('-num_class', default=1000, type=int, metavar='NUM_CLASS', help='number of class, output channels')
parser.add_argument('-height', default=224, type=int, metavar='HEIGHT', help='height of input')
parser.add_argument('-width', default=224, type=int, metavar='WIDTH', help='width of input')
parser.add_argument('-bs', default=512, type=int, metavar='BATCH SIZE', help='batch size')
parser.add_argument('-fold', default=5, type=int, metavar='FOLD', help='k-fold CV')
parser.add_argument('arch', type=str, metavar='ARCH', help='model architecture')



transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ])



transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    ])



class ImageNetDataset(Dataset):

    def __init__(self, path_data, path_txt, mode):
        self.mode = mode
        self.list_label = []
        self.list_path_img = []
        fileDescriptor = open(path_txt, "r")
        line = True
        while line:
            line = fileDescriptor.readline()
            if line:
                lineItems = line.split()
                path_img = r"{}/{}".format(path_data, lineItems[0])
                self.list_path_img.append(path_img)
                label = lineItems[1:]
                label = [int(i) for i in label]
                self.list_label.append(label)
        fileDescriptor.close()

    def __getitem__(self, idx):
        img = Image.open(self.list_path_img[idx]).convert("RGB")
        # img = img.resize((256, 256), resample=Image.LANCZOS)
        if self.mode == "train":
            img = transform_train(img)
        elif self.mode == "inf":
            img = transform_val(img)
        label = torch.FloatTensor(self.list_label[idx])
        return img, label

    def __len__(self):
        return len(self.list_path_img)



criterion = nn.CrossEntropyLoss().to(device)



def cal_top5_error(preds, labels):
    """
    """

    preds_np = preds.data.cpu().numpy()    # (batch, 1000)
    labels_np = labels.data.cpu().numpy()    # (batch, 1000)

    n_sum = int(preds.shape[0])
    n_error = 0
    for i in range(preds.shape[0]):
        pred_i = preds_np[i, :]
        pred_idx_top5 = pred_i.argsort()[::-1][0:5].tolist()
        label_i = labels_np[i, :].tolist()
        label_idx = label_i.index(max(label_i)) 
        if label_idx not in pred_idx_top5:
            n_error += 1

    return n_sum, n_error



def cal_mean_stddev(list_loss_fold):

    mean_numerator = 0
    for loss_i in list_loss_fold:
        mean_numerator += loss_i
    mean = mean_numerator / len(list_loss_fold)

    variance_numerator = 0
    for loss_i in list_loss_fold:
        variance_numerator += ((loss_i - mean) ** 2)
    std_dev = (variance_numerator / len(list_loss_fold)) ** 0.5
    return mean, std_dev



def selec_model(arch):

    if arch=="AlexNet":
        model = torchvision.models.alexnet().to(device)    # 37.5% top-1 & 17% top-5
    elif arch == "VGG11":
        model = torchvision.models.vgg11().to(device)
    elif arch == "VGG11-bn":
        model = torchvision.models.vgg11_bn().to(device)
    elif arch == "VGG13":
        model = torchvision.models.vgg13().to(device)
    elif arch == "VGG13-bn":
        model = torchvision.models.vgg13_bn().to(device)
    elif arch == "VGG16":
        model = torchvision.models.vgg16().to(device)
    elif arch == "VGG16-bn":
        model = torchvision.models.vgg16_bn().to(device)
    elif arch == "VGG19":
        model = torchvision.models.vgg19().to(device)
    elif arch == "VGG19-bn":
        model = torchvision.models.vgg19_bn().to(device)    # 6.8% top-5
    elif arch == "Inception-v1":
        model = torchvision.models.googlenet().to(device)    # 6.67% top-5
    elif arch == "Inception-v3":
        model = torchvision.models.inception_v3().to(device)    # 3.58% top-5
    elif arch == "ResNet18":
        model = torchvision.models.resnet18().to(device)
    elif arch == "ResNet34":
        model = torchvision.models.resnet34().to(device)
    elif arch == "ResNet50":
        model = torchvision.models.resnet50().to(device)
    elif arch == "ResNet101":
        model = torchvision.models.resnet101().to(device)
    elif arch == "ResNet152":
        model = torchvision.models.resnet152().to(device)
    elif arch == "ResNeXt50-32x4d":
        model = torchvision.models.resnext50_32x4d().to(device)
    elif arch == "ResNeXt101-32x8d":
        model = torchvision.models.resnext101_32x8d().to(device)
    elif arch == "Wide-ResNet50-2":
        model = torchvision.models.wide_resnet50_2().to(device)
    elif arch == "Wide-ResNet101-2":
        model = torchvision.models.wide_resnet101_2().to(device)
    return model



def test():

    args = parser.parse_args()

    # ---- start testing ----
    list_loss = []
    list_top5error = []
    with torch.no_grad():
        for k in range(1, args.fold+1):
            print("========== Fold {} ==========".format(k))
            # ---- setting model ----
            model = selec_model(args.arch)
            model = torch.nn.DataParallel(model).to(device)
            model_ckpt = torch.load(r"./checkpoints/fold_{}.pth.tar".format(k))
            model.load_state_dict(model_ckpt)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            # ---- loading data ----
            path_txt_test = r"{}/fold_{}/test.txt".format(args.txt, k)
            dataset = ImageNetDataset(args.data, path_txt_test, "inf")
            data = DataLoader(dataset=dataset, batch_size=args.bs, num_workers=12, pin_memory=True)
            # ---- start iterating ----
            loss_test_sum = 0
            n_sum = 0
            n_error = 0
            for step, (inputs, labels) in enumerate(tqdm(data)):
                # ---- inputs & labels ----
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)
                # ---- fp ----
                outputs = model(inputs)
                labels_max = torch.max(labels, 1)[1]
                # ---- bp ----
                loss = criterion(outputs, labels_max)
                loss_test_sum += loss.item()
                n_sum_i, n_error_i = cal_top5_error(outputs, labels)
                n_sum += n_sum_i
                n_error += n_error_i
            # ---- test loss & top-5 error ----
            loss_test = loss_test_sum / len(data)
            top5_error_test = round(n_error/n_sum, 3)
            list_loss.append(loss_test)
            list_top5error.append(top5_error_test)
        # ---- loss/accuracy: Mean +- Standard Deviation ----
        mean_loss, std_dev_loss = cal_mean_stddev(list_loss)
        mean_top5, std_dev_top5 = cal_mean_stddev(list_top5error)
        print("Loss: Mean({:.3f}), Standard Deviation({:.3f}) ".format(mean_loss, std_dev_loss))
        print("Top-5 Error: Mean({:.3f}), Standard Deviation({:.3f}) ".format(mean_top5, std_dev_top5))




if __name__ == "__main__":

    test()


