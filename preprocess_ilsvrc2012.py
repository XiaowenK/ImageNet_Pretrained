
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Author   : Xiaowen Ke
## Email    : xiaowen.herman@gmail.com
## Version  : v0.1.0
## Date     : 2021/02/18
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import tarfile
import scipy
import shutil

from tqdm import tqdm
from scipy import io




def prepare_train_set(path_train):
    """
    """

    # ---- Unzip ----
    list_files = os.listdir(path_train)
    for file_tar in tqdm(list_files):
        os.mkdir("{}/{}".format(path_train, file_tar.split(".")[0]))
        path_tar = "{}/{}".format(path_train, file_tar)
        tar = tarfile.open(path_tar, mode="r")
        file_names = tar.getnames()
        for file_name in file_names:
            tar.extract(file_name, "{}/{}".format(path_train, file_tar.split(".")[0]))
        tar.close()
        os.remove("{}/{}".format(path_train, file_tar))

    # ---- Counting total number of images ----
    list_itms = os.listdir(path_train)
    n = 0
    list_folders = []
    for itm in list_itms:
        if not itm.endswith(".tar"):
            list_jpg = os.listdir("{}/{}".format(path_train, itm))
            for jpg in list_jpg:
                if jpg.endswith(".JPEG"):
                    n += 1
    print("Total number of images in training set: {}".format(n))




def prepare_val_set(path_synset, path_gt, path_val):
    """
    """

    synset = scipy.io.loadmat(path_synset)
    ground_truth = open(path_gt)
    lines = ground_truth.readlines()
    labels = [int(line[:-1]) for line in lines]

    root, _, filenames = next(os.walk(path_val))
    for filename in filenames:
        # val image name -> ILSVRC ID -> WIND
        val_id = int(filename.split(".")[0].split("_")[-1])
        ILSVRC_ID = labels[val_id-1]
        WIND = synset['synsets'][ILSVRC_ID-1][0][1][0]
        print("val_id:{}, ILSVRC_ID:{}, WIND:{}".format(val_id, ILSVRC_ID, WIND))
        # move val images
        output_dir = os.path.join(path_val, WIND)
        if os.path.isdir(output_dir):
            pass
        else:
            os.mkdir(output_dir)
        shutil.move(os.path.join(root, filename), os.path.join(output_dir, filename))




if __name__ == "__main__":

    path_train = r"./database/train"
    prepare_train_set(path_train)


    path_synset = r"./database/ILSVRC2012_devkit_t12/data/meta.mat"
    path_gt = r"./database/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
    path_val = r"./database/val"
    prepare_val_set(path_synset, path_gt, path_val)


