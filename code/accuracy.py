import os
from os import listdir
import numpy as np
import json

label_path = './dataset/val/'#预测标记文件路径
mark_path = './dataset/mark/'#真实标记文件路径

num_right=0
all_point=0
label = np.zeros((26,2))
for i,fn in enumerate(os.listdir(label_path)):
    if 'json' in  fn:
        mark = np.loadtxt(mark_path+fn[:-4]+'txt', delimiter=',')
        with open(label_path+fn) as f:
            labels = json.load(f)
        
        first = labels[0]['label']-1#读取标签文件
        last = labels[-1]['label']-1
        for i,landmark in enumerate(labels):
            label[i+first][0] = landmark['X']
            label[i+first][1] = landmark['Y']
        
        Mark = mark[first:last+1]#读取预测文件
        Label = label[first:last+1]
        error  = np.sqrt(np.sum(np.square(Mark-Label),axis=1))#计算预测值与真实值像素距离
        right = error[error<13.33]#一个像素代表0.75mm，真实距离小于1cm视为预测正确
        num_right += right.shape[0]#预测正确的脊椎数目
        all_point += Label.shape[0]#总共预测的脊椎数目
     
        print(label)
        print(mark)
        print(error)
    
    accuracy = num_right /all_point#计算正确率
    print(accuracy)
