import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import AE
import copy
import myutils as utils
import torch.nn.functional as F
import os

torch.autograd.set_detect_anomaly(True)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

plt.switch_backend('TKAgg')

try:
    AdaParams = utils.get_params('ShiftAdapter')
except Exception as e:
    print('Error: Fail to Get Params in <configs.yml>', e)
    exit(-1)
# trp_except:用于捕获异常，try内放置可能会出现异常的句子，如果出现异常则进入exception内


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ShiftAdapter:
    def __init__(self, model):
        self.model = model
    def adapt(self,remain_X,feature_size,batch_size=AdaParams['batch_size']):
        self.model.add_task(feature_size)
        def np2ts(X):
            return torch.from_numpy(X).type(torch.float).to(device)

        remain_X = np2ts(remain_X)
        optimizier = optim.Adam(self.model.parameters(), lr=0.00001)
        # print(len(remain_X))
        torch_dataset = Data.TensorDataset(remain_X, remain_X)  # x_train为何是x_train的标签？----重构学习
        # dataloader将封装好的数据进行加载
        loader = Data.DataLoader(
            dataset=torch_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        num_epochs = 30
        alpha2=0.01
        alpha1 =1-alpha2
        for epoch in range(num_epochs):
            for step, (batch_x, batch_y) in enumerate(loader):
                output = self.model(batch_x)  # 得到经过自编码器的输出值
                reconstruction_loss = nn.MSELoss()(output, batch_x)
                l2_reg = 0
                for param in self.model.parameters():
                    l2_reg += torch.norm(param, p=2)
                loss = alpha1 * reconstruction_loss + alpha2 * l2_reg
                loss.backward()
                optimizier.step()
            print('epoch:{}/{}'.format(epoch, num_epochs), '|Loss:', loss.item())


