from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import myutils as utils
from sklearn.metrics.pairwise import rbf_kernel
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.manifold import TSNE

# 获取自编码器的一些参数设定
Params = utils.get_params('AE')
EvalParams = utils.get_params('Eval')  # 得到评估的参数

# torch.device：指定在哪个设备上执行模型操作
# 切换设备的操作 先判断GPU设备是否可用，如果可用则从第一个标识开始，如果不可用，则选择在cpu设备上开始
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()  # 常用的名词 在使用损失函数的是时候经常搭配criterion
getMSEvec = nn.MSELoss(reduction='none')


class ProgressiveAutoencoder(nn.Module):
    def __init__(self, feature_size):
        super(ProgressiveAutoencoder, self).__init__()
        self.encoders = nn.ModuleList([self._make_encoder(feature_size)])
        self.decoders = nn.ModuleList([self._make_decoder(feature_size)])
        self.current_task = 0

    def _make_encoder(self, feature_size, new_task=False):
        layers = [
            nn.Linear(feature_size, int(feature_size * 0.75)),
            nn.ReLU(True),
            nn.Linear(int(feature_size * 0.75), int(feature_size * 0.5)),
            nn.ReLU(True),
        ]
        if new_task:  # 如果是新任务，添加一个额外的全连接层
            layers.append(nn.Linear(int(feature_size * 0.5), int(feature_size * 0.5)))
            layers.append(nn.ReLU(True))
        layers.extend([
            nn.Linear(int(feature_size * 0.5), int(feature_size * 0.25)),
            nn.ReLU(True),
        ])
        if new_task:  # 如果是新任务，添加一个额外的全连接层
            layers.append(nn.Linear(int(feature_size * 0.25), int(feature_size * 0.25)))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(int(feature_size * 0.25), int(feature_size * 0.1)))
        return nn.Sequential(*layers)

    def _make_decoder(self, feature_size, new_task=False):
        layers = [
            nn.Linear(int(feature_size * 0.1), int(feature_size * 0.25)),
            nn.ReLU(True),
        ]
        if new_task:  # 如果是新任务，添加一个额外的全连接层
            layers.append(nn.Linear(int(feature_size * 0.25), int(feature_size * 0.25)))
            layers.append(nn.ReLU(True))
        layers.extend([
            nn.Linear(int(feature_size * 0.25), int(feature_size * 0.5)),
            nn.ReLU(True),
        ])
        if new_task:  # 如果是新任务，添加一个额外的全连接层
            layers.append(nn.Linear(int(feature_size * 0.5), int(feature_size * 0.5)))
            layers.append(nn.ReLU(True))
        layers.extend([
            nn.Linear(int(feature_size * 0.5), int(feature_size * 0.75)),
            nn.ReLU(True),
            nn.Linear(int(feature_size * 0.75), feature_size),
        ])
        return nn.Sequential(*layers)

    def forward(self, x):
        # 对于每个任务，数据通过所有任务的编码器和当前任务的解码器
        encoded = self.encoders[self.current_task](x)
        decoded = self.decoders[self.current_task](encoded)
        return decoded

    def encode(self, x):
        encoded = self.encoders[self.current_task](x)
        return encoded

    def add_task(self, feature_size):
        # 冻结所有旧任务的参数
        for param in self.parameters():
            param.requires_grad = False

        # 添加新任务的编码器和解码器层，新任务标记为True
        self.encoders.append(self._make_encoder(feature_size, new_task=True))
        self.decoders.append(self._make_decoder(feature_size, new_task=True))

        # 更新当前任务索引
        self.current_task += 1
# 假设你已经完成了第一个任务的训练，然后添加一个新任务


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
    model = ProgressiveAutoencoder(feature_size).to(device)  # model.to(device) 表示的是将模型加载到对应的设备上
    optimizier = optim.Adam(model.parameters(), lr=lr, weight_decay=Params['weight_decay'])
    model.train()
    X_train = torch.from_numpy(X_train).type(torch.float)
    if torch.cuda.is_available(): X_train = X_train.cuda()  # .cuda()函数：表示将数据拷贝在GPU上
    torch_dataset = Data.TensorDataset(X_train, X_train)  # x_train为何是x_train的标签？----重构学习
    dataloader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=Params['batch_size'],
        shuffle=True,
    )
    for epoch in range(epoches):
        for step, (batch_x, batch_y) in enumerate(dataloader):
            output = model(batch_x)
            loss = criterion(output, batch_y)
            optimizier.zero_grad()
            loss.backward()
            optimizier.step()
        if EvalParams['verbose_info']:
            print('epoch:{}/{}'.format(epoch, epoches), '|Loss:', loss.item())

    # prev_params = [param.clone().detach() for param in model.parameters()]
    # grads = [param.grad for param in model.parameters()]
    # memory_factors = [torch.abs(param) for param in grads]
    model.eval()
    output = model(X_train)
    mse_vec = getMSEvec(output, X_train)
    rmse_vec = se2rmse(mse_vec).cpu().data.numpy()  # 得到均方根误差，并将类型转换为数组类型（因为gpu上无法进行数据的类型转换）

    if EvalParams['verbose_info']:
        print("max AD score", max(rmse_vec))
    rmse_vec.sort()
    pctg = Params['percentage']
    thres = rmse_vec[int(len(rmse_vec) * pctg)]
    # thres是rmse_vec的99%的那个，例如rmse_vec有200个。则thres=rmse_vec[198]
    return model, thres  # 返回训练好的模型以及计算得到的阈值以及得到的重要参数


@torch.no_grad()
def test(model, thres, X_test):
    model.eval()
    X_test = torch.from_numpy(X_test).type(torch.float)
    X_test = X_test.to(device)
    output = model(X_test)
    mse_vec = getMSEvec(output, X_test)
    rmse_vec = se2rmse(mse_vec).cpu().data.numpy()
    y_pred = np.asarray([0] * len(rmse_vec))
    idx_mal = np.where(rmse_vec > thres)
    y_pred[idx_mal] = 1

    return y_pred, rmse_vec


def PSA(model, X_o, X_n, y_o, y_n, old_num, label_num):

    def encode_data(model, data_loader):
        model.eval()
        encoded_samples = []
        for batch, _ in data_loader:
            encoded = model.encode(batch.to(device)).detach().cpu().numpy()
            encoded_samples.append(encoded)
        encoded_samples = np.vstack(encoded_samples)
        return encoded_samples

    def get_dataloader(X):
        X = torch.from_numpy(X).type(torch.float)
        if torch.cuda.is_available(): X = X.cuda()
        torch_dataset = Data.TensorDataset(X, X)
        dataloader = Data.DataLoader(
            dataset=torch_dataset,
            batch_size=Params['batch_size'],
            shuffle=True,
        )
        return dataloader


    X_o_normal = X_o[y_o == 0]
    old_normal_indices = np.where(y_o == 0)[0]
    # 对筛选后的正常样本进行编码
    old_latent_normal_representations = encode_data(model, get_dataloader(X_o_normal))
    # 对正常样本进行K-means聚类，这里n_clusters=1，因为只对正常样本聚类
    kmeans = KMeans(n_clusters=1, n_init=10, random_state=0).fit(old_latent_normal_representations)
    # 计算所有正常样本到聚类中心的距离
    closest_indices, distances = pairwise_distances_argmin_min(old_latent_normal_representations,
                                                               kmeans.cluster_centers_)
    # 选择距离聚类中心最近的old_num个样本
    old_representative_normal_idx = np.argsort(distances)[:old_num]
    old_representative_normal_idx = old_normal_indices[old_representative_normal_idx]
    old_representative_normal_samples = X_o[old_representative_normal_idx]
    # 注意：这个序列是相对于原始x_o的，不是x_o_normal的
    # 编码新数据
    new_latent_representations = encode_data(model, get_dataloader(X_n))
    # 计算新数据到正常聚类中心的距离，选择距离最远的label_num个样本
    distances_to_normal_cluster = np.linalg.norm(new_latent_representations - kmeans.cluster_centers_[0], axis=1)
    new_representative_idx = np.argsort(-distances_to_normal_cluster)[:label_num]
    new_representative_samples = X_n[new_representative_idx]
    ReturnValues = namedtuple('ReturnValues', ['old_representative_normal_samples', 'old_representative_normal_idx',
                                               'old_latent_normal_representations','old_normal_indices', 'new_representative_samples',
                                               'new_representative_idx', 'new_latent_representations'])
    return ReturnValues(old_representative_normal_samples, old_representative_normal_idx, old_latent_normal_representations,old_normal_indices, new_representative_samples, new_representative_idx, new_latent_representations)
def HumanLabel(new_representative_samples, new_representative_idx, X_n, y_n, label_num):
    print('NOTICE: simulating labelling...')
    remain_y_n = y_n[new_representative_idx]
    print('Filter', len(remain_y_n[remain_y_n == 1]), 'anomalies in X_i_rep')
    new_representative_normal_samples = new_representative_samples[remain_y_n == 0]
    new_representative_normal_idx = np.where(remain_y_n == 0)[0]
    print(" (label_num:%d, X_i_rep_normal:%d, X_i:%d)" % (label_num, len(new_representative_normal_samples), len(X_n)))
    new_representative_normal_idx = new_representative_idx[new_representative_normal_idx]
    return new_representative_normal_samples, new_representative_normal_idx

def PSAPlot(old_representative_normal_idx, old_latent_representations,old_normal_idx, new_representative_normal_idx):
    def tsne_reduce_and_plot(latent_space, representative_indices=None, title='t-SNE plot'):
        tsne = TSNE(n_components=2, random_state=0)
        tsne_results = tsne.fit_transform(latent_space)

        plt.figure(figsize=(8, 8))
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c='gray', alpha=0.5, label='Data')

        if representative_indices is not None:
            plt.scatter(tsne_results[representative_indices, 0], tsne_results[representative_indices, 1], c='red',
                        label='Representative Samples')

        plt.title(title)
        plt.legend()
        plt.show()

    def map_indices(full_indices, valid_indices):
        """
        full_indices: 一个包含从完整数据集中选择出的样本索引的数组
        valid_indices: 一个包含只有子集（例如正常样本）索引的数组
        """
        # 创建一个映射从完整数据集到子集
        index_mapping = {index: i for i, index in enumerate(valid_indices)}

        # 使用映射来转换full_indices到子集空间
        mapped_indices = [index_mapping[idx] for idx in full_indices if idx in index_mapping]
        return mapped_indices

    # 假设 old_valid_idx 是一个包含所有旧数据正常样本索引的数组
    mapped_old_representative_idx = map_indices(old_representative_normal_idx, old_normal_idx)
    # 旧数据的代表性样本和旧数据的潜在空间分布图
    tsne_reduce_and_plot(
        old_latent_representations,
        representative_indices=mapped_old_representative_idx,
        title='Old Data and Representative Samples in Old Data Latent Space'
    )
    # 新数据的代表性样本和旧数据的潜在空间分布图
    tsne_reduce_and_plot(
        old_latent_representations,
        representative_indices=new_representative_normal_idx,
        title='New Representative Samples on Old Data Latent Space'
    )
def test_plot(rmse_vec, thres, file_name=None, label=None):
    plt.figure()
    plt.plot(np.linspace(0, len(rmse_vec) - 1, len(rmse_vec)), [thres] * len(rmse_vec), c='black',
             label='99th-threshold')
    # 得到点和线的关系
    # plt.ylim(0,thres*2.)

    if label is not None:

        idx = np.where(label == 0)[0]  # 返回正常样本的所在的行
        plt.scatter(idx, rmse_vec[idx], s=8, color='blue', alpha=0.4, label='Normal')  # 用scatter画散点图

        idx = np.where(label == 1)[0]
        plt.scatter(idx, rmse_vec[idx], s=8, color='red', alpha=0.7, label='Anomalies')
    else:
        plt.scatter(np.linspace(0, len(rmse_vec) - 1, len(rmse_vec)), rmse_vec, s=8, alpha=0.4, label='Test samples')

    plt.legend(loc='upper right')  # 设置图例的位置、大小等函数
    plt.xlabel('Sample NO.')
    plt.ylabel('Anomaly Score (RMSE)')
    plt.title('Per-sample Score')
    plt.show()

