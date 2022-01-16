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

num_val = 135
num_vertebreas = 26

model_path = './model/150modelR12.pkl'
#model_path = './1e-3modelR10130.pkl'
val_path = './dataset/val/'

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')#设置使用的GPU

#加载模型
net = UNet(n_channels=1,n_classes=num_vertebreas).to(device).train()

net.load_state_dict(torch.load(model_path, map_location={'cuda:0':'cuda:2'}))
#net = nn.DataParallel(net)

#读取测试集X，y
X_val = np.zeros((num_val,512,512))
position = np.zeros((num_val, num_vertebreas, 2))
k = 0
first = []
last = []

#生成训练集的Heatmap
for i,fn in enumerate(os.listdir(val_path)):#i代表第i个文件

    if '.json' in fn:

        X_val[k] = plt.imread(val_path + fn[:-4]+'png')
        X_val[k] = (X_val[k]-np.min(X_val[k]))/(np.max(X_val[k])-np.min(X_val[k]))  
        with open (val_path + fn) as f:
            landmarks = json.load(f)
            first.append(landmarks[0]['label'] -1)
            last.append (landmarks[-1]['label'] -1)#直接存储索引
            #print(first[k])
            for j,landmark in enumerate(landmarks):#j代表第i个文件中第j个标记
                position[k][j+first[k]][0] = landmark['X']
                position[k][j+first[k]][1] = landmark['Y']
        k += 1


fn_set = []#存储读取文件顺序
for i,fn in enumerate(os.listdir(val_path)):#i代表第i个文件

    if '.json' in fn:
        fn_set.append(fn)
        

#预测并得出坐标
x_val = torch.from_numpy(X_val)
x_val = x_val.to(torch.float32)
x_val = torch.unsqueeze(x_val, 1)
x_val = x_val.to(device) 

x_batch = Data.DataLoader(dataset = x_val, batch_size = 4, shuffle = False, num_workers = 0,)


for num,X in enumerate(x_batch):#由于GPU的限制只能四张四张预测
    #print('X:',X.size())
    prediction = net(X)

    prediction_np = prediction.data.cpu().detach().numpy()[0]
    #y_np = y.data.cpu().detach().numpy()[0]
    prediction_np_show = np.mean(prediction_np,axis=0,keepdims=False)#对预测结果通道取均值
    
    plt.subplot(111)
    plt.imshow(prediction_np_show,plt.cm.gray)#展示预测结果
  
    plt.savefig("./val_image/{}.png".format(num))#存储预测结果

    #计算预测坐标并存储为对应文件
    prediction = prediction.cpu().detach().numpy()#计算预测坐标并存储为对应文件
  
    pred_flatten = prediction.reshape(X.shape[0], num_vertebreas, -1)
    index = np.argmax(pred_flatten,axis=2)
    

    y = np.floor((index + 1)/512) 
    #print(x)
    x = (index + 1)%512 -1
    y[x<0] -= 1
    x[x<0] = 511

    x = x.reshape(x.shape[0],x.shape[1],1)
    y = y.reshape(y.shape[0],y.shape[1],1)
    pred_position = np.concatenate((x,y), axis=2)
   
    if num < 33:#由于测试集总数不能整除4
        list1 = range(4)
    else:
        list1 = range(3)

    for m in list1:
        
        np.savetxt('./dataset/mark/' + fn_set[num*4+m][:-4]+'txt', pred_position[m], fmt='%f',delimiter=',')#存储预测的文件


