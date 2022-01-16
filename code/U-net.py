import json
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from PIL import Image
import os

import torch
import torch.nn.functional as F
import torch.utils.data as Data
from unet_model import UNet

#数据集路径
train_path = './dataset/train/'
val_path = './dataset/val/'

#图片尺寸、脊椎数目、训练集测试集数目
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
num_vertebreas = 26
num_train = 403
num_val = 135

#初始化输入矩阵和目标矩阵
X_train = np.zeros((num_train, IMAGE_HEIGHT, IMAGE_WIDTH))
Gauss_map = np.zeros((num_train, num_vertebreas, IMAGE_HEIGHT, IMAGE_WIDTH))

#元素值为所在行数矩阵
x1 = np.arange(IMAGE_WIDTH)
x_map = np.matlib.repmat(x1, IMAGE_HEIGHT, 1)
#元素值为所在列数矩阵
y1 = np.arange(IMAGE_HEIGHT)
y_map = np.matlib.repmat(y1, IMAGE_WIDTH, 1)
y_map = np.transpose(y_map)

R = 12#高斯分布δ

k = 0
first = []#存储图片中包含的第一个脊椎的标号
last = []#存储图片中包含的最后一个脊椎的标号

position = np.zeros((num_train, num_vertebreas, 2))#初始化脊椎位置矩阵

for i,fn in enumerate(os.listdir(train_path)):
    if '.json' in fn:#保证标签和图片的顺序一致
        X_train[k] = plt.imread(train_path + fn[:-4]+'png')#读取图片
        X_train[k] = (X_train[k]-np.min(X_train[k]))/(np.max(X_train[k])-np.min(X_train[k]))#归一化

        with open (train_path + fn) as f:
            landmarks = json.load(f)
            #first = landmarks[0]['label']
            first.append(landmarks[0]['label']-1)#存储首尾标号索引
            last.append(landmarks[-1]['label']-1)

            for j,landmark in enumerate(landmarks):#为第i个文件中第j个脊椎标记生成Heatmap 
                center_x = landmark['X']
                center_y = landmark['Y']
                position[k][j+first[k]][0] = center_x#标签矩阵
                position[k][j+first[k]][1] = center_y

                mask_x = np.matlib.repmat(center_x, IMAGE_HEIGHT, IMAGE_WIDTH)
                mask_y = np.matlib.repmat(center_y, IMAGE_HEIGHT, IMAGE_WIDTH)

                #Gauss_map[i][j+first-1] = np.sqrt((x_map-mask_x)**2+(y_map-mask_y)**2)
                Gauss_map[k][j+first[k]] = ((x_map-mask_x)**2+(y_map-mask_y)**2)/(2*R*R)#x_map-mask_X得出距离
                #Gauss_map[i][j+first-1] = 100*np.exp(-0.5*Gauss_map[i][j+first-1]/R)
                Gauss_map[k][j+first[k]] = 1/(2*np.pi*R*R) * np.exp(-Gauss_map[k][j+first[k]])
                Max = np.max(Gauss_map[k][j+first[k]])
                Min = np.min(Gauss_map[k][j+first[k]])
                Gauss_map[k][j+first[k]] = (Gauss_map[k][j+first[k]]-Min)/(Max-Min)#热力图归一化

        k += 1
 
Y_train = Gauss_map

#model_path = './model/130modelR1210.pkl'#kequ

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')#初始化模型
model = UNet(n_channels=1, n_classes=num_vertebreas).to(device)
#model.load_state_dict(torch.load(model_path, map_location={'cuda:0':'cuda:1'}))#kequ
#model = torch.nn.DataParallel(model)

model.train()

optim = torch.optim.SGD(model.parameters(), lr = 1e-1, momentum=0.9, weight_decay=3e-4)#初始化优化器
#dataloader = ...
epochs = 100
BATCH_SIZE = 4
x_train = torch.from_numpy(X_train)#变量类型转换为模型输入所需的类型
x_train = x_train.to(torch.float32)
y_train = torch.from_numpy(Y_train)
y_train = y_train.to(torch.float32)

Dataset = Data.TensorDataset(x_train, y_train)#数据集包
loader = Data.DataLoader(dataset = Dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 0,)

Loss = torch.nn.BCEWithLogitsLoss()#初始化损失函数

#lossSum = 0
for i in range(epochs):#训练
    lossSum = 0
    #for X, y in zip(x_train,y_train):
    k = 0
    for X, y in loader:
        #X = torch.unsqueeze(X, 0)
        X = torch.unsqueeze(X, 1)
        #y = torch.unsqueeze(y, 0)
        X = X.to(device)  # [N, 1, H, W]将变量转到GPU中
        y = y.to(device)  # [N, H, W] with class indices (0, 1)
        model.train()
        prediction = model(X)  # [N, 2, H, W]
      
        loss = Loss(prediction, y)

        prediction_np = prediction.data.cpu().detach().numpy()[0]#取单个batch中的一个样本
        y_np = y.data.cpu().detach().numpy()[0]
        #prediction_np_show = np.mean(prediction_np,axis=0,keepdims=False)
        #y_np_show = np.mean(y_np,axis=0,keepdims=False)
        prediction_np_show = prediction_np[10]#取一个样本中的一块脊椎的拟合情况进行观察
        y_np_show = y_np[10]
        plt.subplot(121)
        plt.imshow(prediction_np_show)
        plt.subplot(122)
        plt.imshow(y_np_show)
        plt.savefig("./tmp_image/{}.png".format(k))
        plt.clf()

        lossSum += loss #优化  
        optim.zero_grad()
        loss.backward()
        optim.step()
        #print('step:',k, 'lossSum:', lossSum)
        k += 1


        if k==2:#输出第二个batch当前的预测结果
            model.eval()
            pred_train = model(X)
            
            y0 = y.data.cpu().detach().numpy()[0]#取出这个batch中的第一个样本
            y0_flatten = y0.reshape(1,num_vertebreas, -1)
            max_index = np.argmax(y0_flatten, axis=2)

            y_label = np.floor((max_index+1)/512)#计算出预测的位置
            x_label = (max_index+1)%512 -1
            y_label[x_label<0] -= 1
            x_label[x_label<0] = 511
            
            #输出预测结果和标签
            print('X_label:', x_label)
            print('Y_label', y_label)
            
            print('position:',position[4+0])

            pred_train = pred_train.cpu().detach().numpy()
            pred_flatten = pred_train.reshape(pred_train.shape[0],num_vertebreas, -1)#铺平feature map
            index = np.argmax(pred_flatten, axis=2)#计算这个batch所有的坐标
            y_axis = np.floor((index+1)/512)
            x_axis = (index+1)%512 -1
            y_axis[x_axis<0] -= 1
            x_axis[x_axis<0] = 511
            x_axis = x_axis.reshape(x_axis.shape[0], x_axis.shape[1], 1)
            y_axis = y_axis.reshape(y_axis.shape[0], y_axis.shape[1], 1)
            pred_position = np.concatenate((x_axis,y_axis), axis=2)#坐标拼接
            print('pred_position:', pred_position[0])
            
            errors = []#计算本batch的距离误差
            for m in range(pred_position.shape[0]):
                error = np.sum(np.square(pred_position[m,first[4+m]:last[4+m]] - position[m+4,first[4+m]:last[4+m]]))
                errors.append(error/(last[4+m]-first[4+m]+1))
            #errorsum += error
            print('errors:',errors)

    print('epoch',i,': loss:',lossSum)
    # if i % 10 ==0:
    #     torch.save(model.state_dict(),'./1e-3modelR101'+str(i)+'.pkl')
torch.save(model.state_dict(), './model.pkl')




