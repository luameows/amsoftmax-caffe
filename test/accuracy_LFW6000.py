
# -*- coding:utf-8 -*-

__author__ = 'lumo_wang'
"""用于测试lfw6000对准确度python版本

距离度量采用余弦夹角与欧氏距离
其中余弦夹角做了归一化处理dist=0.5+0.5*cos
欧氏距离未做归一化处理"""
import re
import numpy as np



# 计算余弦相似度，并完成归一化,dist=0.5+0.5*cos
def cosDistance(array_encodings):
    dist_cos=[]
    length=int (len(array_encodings)/2)
    for i in range(0, length):
        num=np.dot(array_encodings[2*i],array_encodings[2*i+1])
        cos=num/(np.linalg.norm(array_encodings[2*i])*np.linalg.norm(array_encodings[2*i+1]))
        sim=0.5+0.5*cos # 归一化
        dist_cos.append(sim)
    return dist_cos

def cosAccuracy(same_feature,diff_feature):
    same_cos=cosDistance(same_feature)
    same_cos=np.array(same_cos)
    diff_cos=cosDistance(diff_feature)
    diff_cos=np.array(diff_cos)
    threshold=0
    accuracy=0
    for thres in np.arange(0,1,0.025):
        sum_all=np.size(np.where(same_cos>thres))+np.size(np.where(diff_cos<thres))
        accu=sum_all/6000
        if accu>accuracy:
            accuracy=accu
            threshold=thres
    print ('***cosine-distance***\naccuracy: %s\nthreshold: %s'%(accuracy,threshold))

#计算欧氏距离
def oDistance(array_encodings):
    dist_o = []
    for i in range(0, 3000):
        sim = np.linalg.norm(array_encodings[2*i]-array_encodings[2*i+1])
        dist_o.append(sim)
    return dist_o

def oAccuracy(same_feature,diff_feature):
    same_cos=oDistance(same_feature)
    same_cos=np.array(same_cos)
    diff_cos=oDistance(diff_feature)
    diff_cos=np.array(diff_cos)
    threshold=0
    accuracy=0
    for thres in np.arange(0,200,0.5):
        sum_all=np.size(np.where(same_cos<thres))+np.size(np.where(diff_cos>thres))
        accu=sum_all/(3000+2927)
        if accu>accuracy:
            accuracy=accu
            threshold=thres
    print('***o-distance***\naccuracy: %s\nthreshold: %s'%(accuracy,threshold))

if __name__ =='__main__':
    # 读取same与diff的txt文件
    same_path = r'G:\科研\faceRec\人脸识别\神经网络\same_NPD.txt'
    same_encodings = np.loadtxt(same_path)
    diff_path = r'G:\科研\faceRec\人脸识别\神经网络\diff_NPD.txt'
    diff_encodings = np.loadtxt(diff_path)
    cosAccuracy(same_encodings,diff_encodings)
    