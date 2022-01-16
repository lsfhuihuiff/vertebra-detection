from unet_model import UNet
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
import json
import matplotlib.pyplot as plt
import torch.utils.data as Data

from PIL import Image
from PIL import ImageFilter
from torchvision import transforms

import cv2
import  argparse

num_vertebreas = 26

#定义参数，便于测试时调整侧视图像和模型
parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, default='./model/150modelR12.pkl', help='the path of model' )
parser.add_argument('--image-path', type=str, default='./test/LIDC-IDRI-0020.png', help='the path of image' )

args = parser.parse_args()
#定义设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#初始化模型并加载
net = UNet(n_channels=1,n_classes=num_vertebreas).to(device).train()
net.load_state_dict(torch.load(args.model_path))
#net.load_state_dict(torch.load(model_path, map_location={'cuda:0':'cuda:2'}))
#读取数据
X_test = plt.imread(args.image_path)
X_test = (X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test))
x_test = torch.from_numpy(X_test)
x_test = x_test.to(torch.float32)
x_test = torch.unsqueeze(x_test, 0)#转化为pytorch网络可以接收的输入维度
x_test = torch.unsqueeze(x_test, 0)
x_test = x_test.to(device)

#预测并进行后处理
with torch.no_grad():
    prediction = net(x_test)

    prediction = prediction.cpu().detach().numpy()
    #print(prediction.shape)
    pred_flatten = prediction.reshape(x_test.shape[0], num_vertebreas, -1)
    index = np.argmax(pred_flatten,axis=2)#得出最亮点的索引

    #计算最亮点在图像中的坐标
    y = np.floor((index + 1)/512)
    #print(x)
    x = (index + 1)%512 -1
    y[x<0] -= 1
    x[x<0] = 511
    #形成坐标对
    x = x.reshape(x.shape[0],x.shape[1],1)
    y = y.reshape(y.shape[0],y.shape[1],1)
    pred_position = np.concatenate((x,y), axis=2)
    pred_position = np.squeeze(pred_position, 0)
    #print(pred_position)
#加载标签文件便于检测测试结果
with open(args.image_path[:-3]+'json') as f:
    labels = json.load(f)
#确定图像中包含的脊椎
first = labels[0]['label']-1#读取标签文件
last = labels[-1]['label']-1
pred_position = pred_position[first:last+1]
print(pred_position)
#根据预测的坐标在图像上绘制红色圆圈加一体现
image = cv2.imread(args.image_path)
for i in range(np.shape(pred_position)[0]):
    cv2.circle(image, (int(pred_position[i][0]),int(pred_position[i][1])), radius=4, thickness=2, color=(0,0,255) )
#存储标记后的图像
#cv2.imshow('result', image)
save_path = './result/'+args.image_path[7:]
cv2.imwrite(save_path, image)