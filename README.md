# Introduction

This project will introduced how to use ImageNet dataset to pretrained a model step by step. My implementation is mainly based on Pytorch. This repo will be a work in progress and I will keep on updating.

Furthermore, for more details, please visit my [blog](https://www.cnblogs.com/hmlovetech/).

**WIP**





# Dataset
ILSVRC2012:
- [train](https://pan.baidu.com/s/14F8d_0d2Ly4ZuGP-6ggxLQ), keyword: yoos;
- [val](https://pan.baidu.com/s/1df3SldpQ64Me6h8oRmONRg), keyword: yl8m;
- [test](https://pan.baidu.com/s/1RkW_cw6EDb2EjvvmZyHvxg), keyword: jumt;
- [devkit_t12](https://pan.baidu.com/s/1e_uWi086CI1BEPmRfUfARw), keyword: yoos;





# How to use?
### Dependencies
WIP
### Prepare the data
First, download the dataset above and put them into the /database folder. Then unzip them in the /database folder by following commands:
```
# mkdir train && cp ILSVRC2012_img_train.tar.gz train && cd train && tar -xvf *.gz && rm -rf *.gz && cd ..
# mkdir val && cp ILSVRC2012_img_val.tar.gz val && cd val && tar -xvf *.gz && rm -rf *.gz && cd ..
# mkdir test && cp ILSVRC2012_img_test.tar.gz test && cd test && tar -xvf *.gz && rm -rf *.gz && cd ..
# tar -xvf ILSVRC2012_devkit_t12.tar
```
Now we get the following files and sub-folders:
```
# --- /database
#            |--- ILSVRC2012_devkit_t12/
#            |--- test/
#            |--- train/
#            |       |--- n01440764.tar
#            |       |--- n01443537.tar
#            |       |--- ...
#            |--- val/
#            |       |--- ILSVRC2012_val_00000001.JPEG
#            |       |--- ILSVRC2012_val_00000002.JPEG
#            |       |--- ...
#            |--- ILSVRC2012_devkit_t12.tar
#            |--- ILSVRC2012_img_test.tar.gz
#            |--- ILSVRC2012_img_train.tar.gz
#            |--- ILSVRC2012_img_val.tar.gz
```
Finally, we still have to unzip .tar files in train/ and organize all val images by synsets:
```
python3 preprocess_ilsvrc2012.py
```
### Training
WIP
### Testing
WIP





# License
The license is MIT.