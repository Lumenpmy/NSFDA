import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import myutils as utils
from sklearn.metrics.pairwise import rbf_kernel
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

# 获取自编码器的一些参数设定
Params = utils.get_params('AE')
EvalParams = utils.get_params('Eval')  # 得到评估的参数

# torch.device：指定在哪个设备上执行模型操作
# 切换设备的操作 先判断GPU设备是否可用，如果可用则从第一个标识开始，如果不可用，则选择在cpu设备上开始
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()  # 常用的名词 在使用损失函数的是时候经常搭配criterion
getMSEvec = nn.MSELoss(reduction='none')


# 构造自编码器
class autoencoder(nn.Module):
    def __init__(self, feature_size):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(feature_size, int(feature_size * 0.75)),
                                     nn.ReLU(True),  # 参数inplace设置为True是为了节约内存（false为开辟内存进行计算）
                                     nn.Linear(int(feature_size * 0.75), int(feature_size * 0.5)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size * 0.5), int(feature_size * 0.25)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size * 0.25), int(feature_size * 0.1)))

        self.decoder = nn.Sequential(nn.Linear(int(feature_size * 0.1), int(feature_size * 0.25)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size * 0.25), int(feature_size * 0.5)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size * 0.5), int(feature_size * 0.75)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size * 0.75), int(feature_size)),
                                     )
        self.feature_size = feature_size
        self.X_train = None
        self.gradient_sorted = None

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return decode


# reduction为none表示将每个数据和其对应的特征之间的均方误差作为向量的形式输出
# 即每个元素为（ypre-y_label)^2


# 函数的意义-----求均方根误差RMSE(RMSE和MSE的关系，RMSE=MSE^(1/2))
def se2rmse(a):
    return torch.sqrt(sum(a.t()) / a.shape[1])


# t.()是将张量a进行转置
# shape[1]表示张量a的第2维数度
# sqrt表示将括号内的东西取平方根
# 返回的结果是含有一系列数据的均方根误差的张量


def train(X_train, feature_size, epoches=Params['epoches'], lr=Params['lr']):
    # 自编码器模型进行训练，创建autoencoder的实体类model
    model = autoencoder(feature_size).to(device)  # model.to(device) 表示的是将模型加载到对应的设备上
    # Adam可以自适应的调整学习率，有助于快速的收敛
    optimizier = optim.Adam(model.parameters(), lr=lr, weight_decay=Params['weight_decay'])
    # 开始训练模型
    model.train()
    # 将数组类型转换为float的tensor类型
    X_train = torch.from_numpy(X_train).type(torch.float)
    if torch.cuda.is_available(): X_train = X_train.cuda()  # .cuda()函数：表示将数据拷贝在GPU上
    # TensorDataset():进行数据的封装
    torch_dataset = Data.TensorDataset(X_train, X_train)  # x_train为何是x_train的标签？----重构学习
    # dataloader将封装好的数据进行加载
    dataloader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=Params['batch_size'],
        shuffle=True,
    )
    # 进行每一轮的训练
    for epoch in range(epoches):
        for step, (batch_x, batch_y) in enumerate(dataloader):
            output = model(batch_x)  # 得到经过自编码器的输出值
            loss = criterion(output, batch_y)
            optimizier.zero_grad()
            loss.backward()
            optimizier.step()
        if EvalParams['verbose_info']:
            print('epoch:{}/{}'.format(epoch, epoches), '|Loss:', loss.item())

    prev_params = [param.clone().detach() for param in model.parameters()]
    grads = [param.grad for param in model.parameters()]
    memory_factors = [torch.abs(param) for param in grads]
    # 利用验证集合，模型选择，根据训练的模型来确定超参的值
    model.eval()
    output = model(X_train)
    mse_vec = getMSEvec(output, X_train)
    rmse_vec = se2rmse(mse_vec).cpu().data.numpy()  # 得到均方根误差，并将类型转换为数组类型（因为gpu上无法进行数据的类型转换）

    if EvalParams['verbose_info']:
        print("max AD score", max(rmse_vec))

    # thres = max(rmse_vec)
    rmse_vec.sort()  # 将列表进行排列，默认为升序
    pctg = Params['percentage']
    thres = rmse_vec[int(len(rmse_vec) * pctg)]
    # ❌：将thres定义为模型经过训练输出的均方根误差的最大值
    # thres是rmse_vec的99%的那个，例如rmse_vec有200个。则thres=rmse_vec[198]
    return model, thres, prev_params, memory_factors  # 返回训练好的模型以及计算得到的阈值以及得到的重要参数


@torch.no_grad()
def test(model, thres, X_test):
    model.eval()
    X_test = torch.from_numpy(X_test).type(torch.float)
    X_test = X_test.to(device)
    output = model(X_test)
    mse_vec = getMSEvec(output, X_test)
    rmse_vec = se2rmse(mse_vec).cpu().data.numpy()
    y_pred = np.asarray([0] * len(rmse_vec))  # 生成了一个和均方根误差长度相同的元素为0的数组
    idx_mal = np.where(rmse_vec > thres)  # 找到均方根误差大于阈值的样本，并输出其位置
    # print(idx_mal)
    y_pred[idx_mal] = 1  # 使用异常置信度模型，标记异常

    return y_pred, rmse_vec


