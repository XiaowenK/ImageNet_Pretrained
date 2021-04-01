
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
import tarfile
import scipy
import shutil
import random
import numpy as np

from tqdm import tqdm
from scipy import io



def gen_txt(path_data, path_txt):
    """
    """

    path_train = r"{}/train".format(path_data)
    path_val = r"{}/val".format(path_data)

    list_synset_train = os.listdir(path_train)
    list_synset_val = os.listdir(path_val)

    list_train_all = []
    list_val_all = []
    list_test_all = []
    for synset_train in list_synset_train:
        # ---- get onhot encoder ----
        idx = int(list_synset_train.index(synset_train))
        label_str = " 0" * idx + " 1" + " 0" * (1000 - 1 - idx)
        #
        list_img_syn_train = os.listdir(r"{}/{}".format(path_train, synset_train))
        list_img_syn_val = os.listdir(r"{}/{}".format(path_val, synset_train))
        # ---- for train ----
        for img_train in list_img_syn_train:
            str_img_train = "train/{}/{}{}".format(synset_train, img_train, label_str)
            list_train_all.append(str_img_train)
        # ---- for val ----
        for img_val in list_img_syn_val:
            str_img_val = "val/{}/{}{}".format(synset_train, img_val, label_str)
            list_val_all.append(str_img_val)

    # ---- for 5-fold train & test  ----
    random.shuffle(list_train_all)
    list_train_all_temp = list_train_all[0:]
    n_5 = round(len(list_train_all_temp) / 5)
    for i in tqdm(range(1, 6)):
        if i==1:
            list_test_i = list_train_all_temp[((i-1)*n_5):(i*n_5)]
            list_train_i = list_train_all_temp[(i*n_5):]
        elif i!=1 and i!=5:
            list_test_i = list_train_all_temp[((i-1)*n_5):(i*n_5)]
            list_train_i = list_train_all_temp[0:((i-1)*n_5)] + list_train_all_temp[(i*n_5):]
        elif i==5:
            list_test_i = list_train_all_temp[((i-1)*n_5):]
            list_train_i = list_train_all_temp[0:((i-1)*n_5)]
        np.savetxt(r"{}/fold_{}/train.txt".format(path_txt, i), np.reshape(list_train_i, -1), delimiter=',', fmt='%5s')
        np.savetxt(r"{}/fold_{}/val.txt".format(path_txt, i), np.reshape(list_val_all, -1), delimiter=',', fmt='%5s')
        np.savetxt(r"{}/fold_{}/test.txt".format(path_txt, i), np.reshape(list_test_i, -1), delimiter=',', fmt='%5s')





if __name__ == "__main__":

    path_data = r"./database"
    path_txt = r"./labels"
    gen_txt(path_data, path_txt)


