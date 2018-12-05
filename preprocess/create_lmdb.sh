#!/bin/bash
DATA=/home/user/caffe-master/train/dataset/Database1000
#Format=_158x158
Format=_79x79


echo "Creating train lmdb..."
rm -rf $DATA/train_lmdb$Format
/home/user/caffe-master/build/tools/convert_imageset --shuffle \
/home/user/caffe-master/train/dataset/Database1000/ $DATA/Train.txt  $DATA/train_lmdb$Format
echo "Done~"

echo "Creating valid lmdb..."
rm -rf $DATA/valid_lmdb$Format
/home/user/caffe-master/build/tools/convert_imageset --shuffle \
/home/user/caffe-master/train/dataset/Database1000/ $DATA/Valid.txt  $DATA/valid_lmdb$Format
echo "Done~"
