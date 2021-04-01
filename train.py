
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



parser = argparse.ArgumentParser(description='Training on ImageNet Dataset.')
parser.add_argument('-data', default=r"./database", type=str, metavar='DATA', help='path to dataset')
parser.add_argument('-txt', default=r"./labels", type=str, metavar='TXT', help='path to txt files')
parser.add_argument('-num_class', default=1000, type=int, metavar='NUM_CLASS', help='number of class, output channels')
parser.add_argument('-height', default=224, type=int, metavar='HEIGHT', help='height of input')
parser.add_argument('-width', default=224, type=int, metavar='WIDTH', help='width of input')
parser.add_argument('-bs', default=512, type=int, metavar='BATCH SIZE', help='batch size')
parser.add_argument('-epoch', default=1000, type=int, metavar='EPOCH', help='max training epoch')
parser.add_argument('-lr', default=0.0001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('-optm', default="Adam", type=str, metavar='OPTM', help='optimizer')
parser.add_argument('-apex', default=True, type=bool, metavar='APEX', help='nvidia apex module')
parser.add_argument('-fold', default=5, type=int, metavar='FOLD', help='k-fold CV')
parser.add_argument('arch', type=str, metavar='ARCH', help='model architecture')
parser.add_argument('start_fold', type=int, metavar="START_FOLD", help='Start from begining or a checkpoint')



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



def train():
    """
    """

    args = parser.parse_args()

    if args.start_fold == 0:
        start_fold = 1
    else:
        start_fold = args.start_fold

    for k in range(start_fold, args.fold+1):
        print("========== Fold {} ==========".format(k))
        # ---- setting model & optimizer ----
        model = selec_model(args.arch)

        if args.optm == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
            # optimizer = optim.SGD(model.parameters(), lr=args.lr)
        elif args.optm == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=100, verbose=True, eps=1e-6)

        if args.apex:
            amp.register_float_function(torch, 'sigmoid')
            amp.register_float_function(F, 'softmax')
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

        model = torch.nn.DataParallel(model)
        if args.start_fold != 0:
            model_ckpt = torch.load(r"./checkpoints/fold_{}.pth.tar".format(k))
            model.load_state_dict(model_ckpt)
        model.train()

        # ---- Tensorboard ----
        writer = SummaryWriter()

        # ---- start training ----
        path_txt_train = r"{}/fold_{}/train.txt".format(args.txt, k)
        loss_min = 100
        n_plateau = 0
        for epoch in range(1, args.epoch+1):
            # ---- timer ----
            starttime = datetime.datetime.now()
            # ---- loading data ----
            print("Loading data...")
            dataset = ImageNetDataset(args.data, path_txt_train, "train")
            data = DataLoader(dataset=dataset, batch_size=args.bs, shuffle=True, num_workers=12, pin_memory=True)
            # ---- loop for all train data ----
            loss_train_sum = 0
            n_sum = 0
            n_error = 0
            for step, (inputs, labels) in enumerate(tqdm(data)):
                # ---- inputs & labels ----
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)    # [batch, 1000] & long
                # print("Input shape:{}".format(inputs.shape))
                # print("Label shape:{}".format(labels.shape))
                # ---- fp ----
                preds = model(inputs)    # [batch, 1000]
                labels_max = torch.max(labels, 1)[1]
                loss = criterion(preds, labels_max)
                loss_train_sum += loss.item()
                if args.apex:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # ---- top 5 error ----
                n_sum_i, n_error_i = cal_top5_error(preds, labels)
                n_sum += n_sum_i
                n_error += n_error_i
            # ---- train loss ----
            loss_train = loss_train_sum / len(data)
            # ---- train error ----
            top5_error_train = round(n_error/n_sum, 3)
            # ---- validation ----
            loss_val, top5_error_val = validation(model, k, args)
            scheduler.step(loss_val)

            # ---- saving ckpt: minimum val loss ----
            if loss_val < loss_min:
                loss_min = loss_val
                print("Best model saved at epoch {}!".format(epoch))
                torch.save(model.state_dict(), r"./checkpoints/fold_{}.pth.tar".format(k))
                n_plateau = 0
            else:
                n_plateau += 1
            # ---- timer ----
            endtime = datetime.datetime.now()
            elapsed = (endtime - starttime).seconds
            # ---- printing ----
            print("fold #{}, epoch #{}, train: {:.4f}, val: {:.4f}, top5_train:{}, top5_val:{}, elapsed: {}s"
                  .format(k, epoch, loss_train, loss_val, top5_error_train, top5_error_val, elapsed))
            print('-' * 60)
            if n_plateau >= 20:
                break



def validation(model, k, args):
    """
    """

    # ---- setting model ----
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    # ---- loading data ----
    path_txt_val = r"{}/fold_{}/val.txt".format(args.txt, k)
    dataset = ImageNetDataset(args.data, path_txt_val, "inf")
    data = DataLoader(dataset=dataset, batch_size=args.bs, num_workers=12, pin_memory=True)
    # ---- loop for all val data ----
    loss_val_sum = 0
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
        loss_val_sum += loss.item()
        n_sum_i, n_error_i = cal_top5_error(outputs, labels)
        n_sum += n_sum_i
        n_error += n_error_i
    # ---- val loss ----
    loss_val = loss_val_sum / len(data)
    # ---- train error ----
    top5_error_val = round(n_error/n_sum, 3)
    # ---- reset the model's mode ----
    model.train()
    for param in model.parameters():
        param.requires_grad = True
    return loss_val, top5_error_val





if __name__ == "__main__":

    train()


