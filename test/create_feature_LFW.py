#-*- coding: UTF-8 -*- 
#!/usr/bin/env python
import sklearn.metrics.pairwise as pw
from  sklearn import preprocessing
import numpy as np
import math
import sys
import os


caffe_root = '/home/user/lumo/caffe-amsoftmax/'  # this file should be run from {caffe_root}/train/dataset (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')
import caffe
caffe.set_mode_cpu()

# 遍历指定目录，显示目录下的所有文件名
def GetFiles(filepath,FaceList):
    pathDir =  os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join('%s/%s' % (filepath, allDir))
        FaceList.append(child.decode('gbk')) # .decode('gbk')是解决中文显示乱码问题

# 遍历指定目录，显示目录下的所有文件夹
def GetFolders(path,Folders):
    pathDir =  os.listdir(path)
    for allDir in pathDir:
        child = os.path.join('%s/%s' % (path, allDir))
        Folders.append(child.decode('gbk')) # .decode('gbk')是解决中文显示乱码问题

def GetFeature(FileList, FeatureFile, n, m):
    for i in range(n, m):
        image1 = caffe.io.load_image(FileList[i])
        transformed_image1 = transformer.preprocess('data', image1)
        # copy the image data into the memory allocated for the net
        net.blobs['data'].data[i-n] = transformed_image1
    ### perform classification
    output = net.forward()
    FileHandle = open(FeatureFile,'a')
    for i in range(0, m-n):
        feat = net.blobs['norm1'].data[i]
        #FileHandle.write(FileList[i+n])
        #FileHandle.write('\n')
        FileHandle.write(' '.join(str(a) for a in feat))
        FileHandle.write('\n')
    FileHandle.close


#for Prof. Deng's model
model_def = '/home/user/lumo/resnet_amsoftmax_webface/deploy_mirror.prototxt'
model_weights = '/home/user/lumo/resnet_amsoftmax_webface/test.caffemodel'
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
# load the mean ImageNet image (as distributed with Caffe) for subtraction
#mu = np.load(caffe_root + 'train/dataset/Database-test/mean_128x128.npy')
#mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
mu=np.array([0.5,0.5,0.5])
print 'mean-subtracted values:', zip('BGR', mu)
# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(50,        # batch size
                          3,         # 3-channel (BGR) images
                          112, 96)  # image size is 227*227
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
#transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

###Process the images
FolderPath = "/home/user/NEST-3T/dataset/face_rec/LFW/crop&align"
f = open("/home/user/NEST-3T/dataset/face_rec/LFW/pairs_same_new.txt","r") 
FileList = []
lines = f.readlines()#读取全部内容 
f.close();
for line in lines:
    child = os.path.join('%s/%s' % (FolderPath, line))
    child = child.strip('\n')
    FileList.append(child.decode('gbk')) # .decode('gbk')是解决中文显示乱码问题
for j in range(0,len(FileList),50):
    print 'Index = ',j, min(j+50, len(FileList))
    GetFeature(FileList, 'same_finetune.txt', j, min(j+50, len(FileList)))


f = open("/home/user/NEST-3T/dataset/face_rec/LFW/pairs_diff_new.txt","r") 
FileList = []
FileList = []
lines = f.readlines()#读取全部内容 
f.close();
for line in lines:
    child = os.path.join('%s/%s' % (FolderPath, line))
    child = child.strip('\n')
    FileList.append(child.decode('gbk')) # .decode('gbk')是解决中文显示乱码问题
for j in range(0,len(FileList),50):
    print 'Index = ',j, min(j+50, len(FileList))
    GetFeature(FileList, 'diff_finetune.txt', j, min(j+50, len(FileList)))



