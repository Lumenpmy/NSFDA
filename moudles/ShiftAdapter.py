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

    def __init__(self):
        self.model = None

    def select_Tab(self, X_o, X_n, y_n,label_num,feature_size):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        class DomainClassifier(nn.Module):
            def __init__(self, input_size, hidden_size, dropout_rate=0.6):
                super(DomainClassifier, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.fc3 = nn.Linear(hidden_size, 1)  # 输出层只有一个节点
                self.dropout = nn.Dropout(dropout_rate)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = F.relu(self.fc2(x))
                x = self.dropout(x)
                weights = torch.sigmoid(self.fc3(x))  # 使用 sigmoid 将输出限制在 0 到 1 之间，表示样本权重
                return weights

        class AttentionNetwork(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(AttentionNetwork, self).__init__()
                self.layer1 = nn.Linear(input_size, hidden_size)
                self.layer2 = nn.Linear(hidden_size, output_size)
                self.relu = nn.LeakyReLU()

            def forward(self, x):
                x = self.layer1(x)
                x = self.relu(x)
                x = self.layer2(x)
                return F.softmax(x, dim=1)

        def estimulate_labeling(X_n,y_n, top_samples_B, top_indices_B, label_num):
            print('NOTICE: simulating labelling...')
            remain_y_t = y_n[top_indices_B]
            print('Filter', len(remain_y_t[remain_y_t == 1]), 'anomalies in remain_X_tre')
            top_samples_B = top_samples_B[remain_y_t == 0]
            top_indices_B = np.where(remain_y_t == 0)[0]
            print(" (label_num:%d, remain_X_n:%d, X_n:%d)" % (label_num, len(top_samples_B), len(X_n)))
            return top_samples_B, top_indices_B

        def get_explain_sample(domain_weights_A, domain_weights_B, label_num):

            # 假设阈值设为0.5
            threshold = 0.5
            print(len(domain_weights_A))
            print(len(domain_weights_B))
            # 从分布A中选择权重小于阈值的样本
            selected_samples_A = X_o[domain_weights_A.flatten() < threshold]
            selected_weights_A = domain_weights_A[domain_weights_A.flatten() < threshold]
            # 从分布B中选择权重大于阈值的样本
            selected_samples_B = X_n[domain_weights_B.flatten() > threshold]
            selected_weights_B = domain_weights_B[domain_weights_B.flatten() > threshold]

            print(len(selected_samples_A))
            print(len(selected_samples_B))
            # 归一化权重
            normalized_weights_A = selected_weights_A / np.sum(selected_weights_A)
            normalized_weights_B = selected_weights_B / np.sum(selected_weights_B)

            # 创建一个包含样本索引和权重的元组列表
            indexed_weights_A = [(index, weight) for index, weight in enumerate(normalized_weights_A)]
            # 根据权重对样本进行排序
            sorted_samples_A = sorted(indexed_weights_A, key=lambda x: x[1], reverse=True)
            # 选择权重最大的前50000个样本的索引
            top_indices_A = [sample[0] for sample in sorted_samples_A[:50000]]
            # 根据索引获取对应的样本
            top_samples_A = [X_o[i] for i in top_indices_A]
            # 创建一个包含样本索引和权重的元组列表
            indexed_weights_B = [(index, weight) for index, weight in enumerate(normalized_weights_B)]
            # 根据权重对样本进行排序
            sorted_samples_B = sorted(indexed_weights_B, key=lambda x: x[1], reverse=True)
            # 选择权重最大的前label_num个样本的索引
            top_indices_B = [sample[0] for sample in sorted_samples_B[:label_num]]
            # 根据索引获取对应的样本
            top_samples_B = [X_n[i] for i in top_indices_B]

            return top_samples_A, top_samples_B, top_indices_A, top_indices_B

        # 训练领域对抗网络
        def train_domain_adversarial(model, domain_classifier, data_loader_A, data_loader_B, optimizer_classifier,
                                     criterion, num_epoches, weight_adjust_factor_A, weight_adjust_factor_B,attention_net):
            domain_weights_A_all = []
            domain_weights_B_all = []

            for epoch in range(num_epoches):
                for data_A, data_B in zip(data_loader_A, data_loader_B):
                    inputs_A, _ = data_A
                    inputs_B, _ = data_B
                    optimizer_classifier.zero_grad()
                    features_A = model.encoder(inputs_A)
                    features_B = model.encoder(inputs_B)

                    # 计算注意力权重
                    attention_weights_A = attention_net(features_A)
                    attention_weights_B = attention_net(features_B)

                    # 领域分类器对抗训练
                    domain_output_A = domain_classifier(features_A)  # 计算样本A的权重
                    domain_output_B = domain_classifier(features_B)  # 计算样本B的权重

                    # 应用注意力权重到领域分类器输出
                    domain_weights_A = domain_output_A * attention_weights_A
                    domain_weights_B = domain_output_B * attention_weights_B

                    # 根据权重调整对抗训练损失函数
                    weighted_domain_loss_A = criterion(domain_weights_A, torch.zeros(
                        (len(inputs_A), 1))) * weight_adjust_factor_A  # 使用权重调整因子调整样本A的损失
                    weighted_domain_loss_B = criterion(domain_weights_B, torch.ones(
                        (len(inputs_B), 1))) * weight_adjust_factor_B  # 样本B的损失不加权

                    domain_loss = weighted_domain_loss_A + weighted_domain_loss_B
                    domain_loss.backward(retain_graph=True)
                    optimizer_classifier.step()
                    if (epoch == num_epoches - 1):
                        domain_weights_A_all.append(domain_weights_A.detach().cpu().numpy())
                        domain_weights_B_all.append(domain_weights_B.detach().cpu().numpy())
                print('epoch:{}/{}'.format(epoch, num_epoches), "domain_loss", domain_loss.item())
                if (epoch == num_epoches - 1):
                    domain_weights_A_all = np.concatenate(domain_weights_A_all)
                    domain_weights_B_all = np.concatenate(domain_weights_B_all)
            return {"domain_weights_A": domain_weights_A_all, "domain_weights_B": domain_weights_B_all}

        # 实例化自编码器M和领域分类器

        num_epoches = 50
        learning_rate = 0.001

        model = AE.autoencoder(feature_size).to(device)
        domain_classifier = DomainClassifier(input_size=int(feature_size * 0.1), hidden_size=100)
        attention_net = AttentionNetwork(input_size=int(feature_size * 0.1), hidden_size=600,output_size=1)
        criterion_domain = nn.BCELoss()
        optimizer_classifier = optim.Adam(list(model.parameters()) + list(domain_classifier.parameters()),
                                          lr=learning_rate)

        # TensorDataset():进行数据的封装
        X_o = torch.from_numpy(X_o).type(torch.float)
        if torch.cuda.is_available(): X_o = X_o.cuda()  # .cuda()函数：表示将数据拷贝在GPU上
        # TensorDataset():进行数据的封装
        torch_dataset = Data.TensorDataset(X_o, X_o)  # x_train为何是x_train的标签？----重构学习
        # dataloader将封装好的数据进行加载
        data_loader_A = Data.DataLoader(
            dataset=torch_dataset,
            batch_size=1024,
            shuffle=True,
        )

        X_n = torch.from_numpy(X_n).type(torch.float)
        if torch.cuda.is_available(): X_n = X_n.cuda()  # .cuda()函数：表示将数据拷贝在GPU上
        torch_dataset = Data.TensorDataset(X_n, X_n)  # x_train为何是x_train的标签？----重构学习
        # dataloader将封装好的数据进行加载
        data_loader_B = Data.DataLoader(
            dataset=torch_dataset,
            batch_size=1024,
            shuffle=True,
        )
        result = train_domain_adversarial(model, domain_classifier, data_loader_A, data_loader_B, optimizer_classifier,
                                          criterion_domain, num_epoches, 0.3, 1.0, attention_net)
        domain_weights_A = result["domain_weights_A"]
        domain_weights_B = result["domain_weights_B"]
        top_samples_A, top_samples_B, top_indices_A, top_indices_B = get_explain_sample(domain_weights_A,
                                                                                        domain_weights_B,label_num)

        # 将列表转换为单个的 numpy.ndarray
        top_samples_B_np = np.array([item.numpy() for item in top_samples_B])
        top_samples_A_np = np.array([item.numpy() for item in top_samples_A])
        top_samples_B = torch.tensor(top_samples_B_np)
        top_samples_A = torch.tensor(top_samples_A_np)

        # X_tre_indices=get_normal_t_idx
        result = estimulate_labeling(X_n,y_n, top_samples_B, top_indices_B,label_num)
        top_samples_B = result[0]
        self.M_c = domain_weights_A[top_indices_A]
        # 使用 torch.cat 进行张量拼接
        mixed_distribution = torch.cat((top_samples_A, top_samples_B), dim=0)
        # print('Remain_X_o.shape', top_samples_A.shape, 'Remain X_n.shape', top_samples_B.shape)

        # 计算这些样本之间的Wasserstein距离
        w_dist = wasserstein_distance(mixed_distribution.ravel(), X_n[:len(mixed_distribution)].ravel())
        print("Wasserstein distance between the two distributions is: ", w_dist)

        self.remain_X_con = top_samples_A
        # print('保留的控制空间的样本个数为i:',len(selected_samples_B))
        self.remain_X_tre = top_samples_B
        self.remain_X = mixed_distribution
        self.EXPLAIN_RES = {
            'remain_X_n': top_samples_B,
            'remain_X_o': top_samples_A,
            'remain_X': mixed_distribution
        }
        return self.EXPLAIN_RES

    def select_Seq(self, X_o, X_n, y_n, label_num):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 复杂化的领域分类器
        class DomainClassifier(nn.Module):
            def __init__(self, input_size, hidden_size, dropout_rate=0.4):
                super(DomainClassifier, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size * 2)  # 增加隐藏层节点数
                self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
                self.fc3 = nn.Linear(hidden_size, 1)
                self.dropout = nn.Dropout(dropout_rate)

            def forward(self, x):
                x = F.leaky_relu(self.fc1(x))  # 更换为leaky_relu激活函数
                x = self.dropout(x)
                x = F.leaky_relu(self.fc2(x))  # 更换为leaky_relu激活函数
                x = self.dropout(x)
                weights = torch.sigmoid(self.fc3(x))
                return weights

        class AttentionNetwork(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(AttentionNetwork, self).__init__()
                self.layer1 = nn.Linear(input_size, hidden_size)
                self.layer2 = nn.Linear(hidden_size, output_size)
                self.relu = nn.LeakyReLU()

            def forward(self, x):
                x = self.layer1(x)
                x = self.relu(x)
                x = self.layer2(x)
                return F.softmax(x, dim=1)

        def estimulate_labling(X_n, y_n, top_samples_B, top_indices_B, label_num):
            print('NOTICE: simulating labelling...')
            remain_y_n = y_n[top_indices_B]
            print('Filter', len(remain_y_n[remain_y_n == 1]), 'anomalies in remain_X_n')
            # 获取布尔数组对应的索引
            indices = np.where(remain_y_n == 0)[0]

            # 使用索引来筛选字典中的元素
            filtered_input = [top_samples_B['input'][i] for i in indices]
            filtered_output = [top_samples_B['output'][i] for i in indices]
            top_samples_B = {'input': filtered_input, 'output': filtered_output}
            # print(len(top_samples_B['input']))
            # print(len(top_samples_B['output']))
            top_indices_B = np.where(remain_y_n == 0)[0]
            # print(top_indices_B)
            print(" (label_num:%d, remain_X_n:%d, X_n:%d)" % (
                label_num, len(top_samples_B['input']), len(X_n['input'])))
            return top_samples_B, top_indices_B


        def get_explain_sample(domain_weights_A, domain_weights_B,label_num):

            # 假设阈值设为0.5
            threshold = 0.5
            # 从分布A中选择权重小于阈值的样本
            selected_samples_A = {'input': X_o['input'][domain_weights_A.flatten() < threshold],
                                  'output': X_o['output'][domain_weights_A.flatten() < threshold]}
            selected_weights_A = domain_weights_A[domain_weights_A.flatten() < threshold]
            print(len(selected_samples_A['input']))
            # 从分布B中选择权重大于阈值的样本
            selected_samples_B = {'input': X_n['input'][domain_weights_B.flatten() > threshold],
                                  'output': X_n['output'][domain_weights_B.flatten() > threshold]}
            selected_weights_B = domain_weights_B[domain_weights_B.flatten() > threshold]
            print(len(selected_samples_B['input']))
            # 归一化权重
            normalized_weights_A = selected_weights_A / np.sum(selected_weights_A)
            normalized_weights_B = selected_weights_B / np.sum(selected_weights_B)
            # 创建一个包含样本索引和权重的元组列表
            indexed_weights_A = [(index, weight) for index, weight in enumerate(normalized_weights_A)]
            # 根据权重对样本进行排序
            sorted_samples_A = sorted(indexed_weights_A, key=lambda x: x[1], reverse=True)

            top_indices_A = [sample[0] for sample in sorted_samples_A[:50000]]
            # 根据索引获取对应的样本
            top_samples_A = {'input': [X_o['input'][i] for i in top_indices_A],
                             'output': [X_o['output'][i] for i in top_indices_A]}
            # 创建一个包含样本索引和权重的元组列表
            indexed_weights_B = [(index, weight) for index, weight in enumerate(normalized_weights_B)]
            # 根据权重对样本进行排序
            sorted_samples_B = sorted(indexed_weights_B, key=lambda x: x[1], reverse=True)
            top_indices_B = [sample[0] for sample in sorted_samples_B[:label_num]]
            # 根据索引获取对应的样本
            top_samples_B = {'input': [X_n['input'][i] for i in top_indices_B],
                             'output': [X_n['output'][i] for i in top_indices_B]}
            return top_samples_A, top_samples_B, top_indices_A, top_indices_B

        # 训练领域对抗网络
        def train_domain_adversarial(model, domain_classifier,attention_net,data_loader_A, data_loader_B,
                                     optimizer_classifier, criterion, num_epoches, weight_adjust_factor_A,
                                     weight_adjust_factor_B):
            domain_weights_A_all = []
            domain_weights_B_all = []
            for epoch in range(num_epoches):
                for data_A, data_B in zip(data_loader_A, data_loader_B):
                    inputs_A, _ = data_A
                    inputs_B, _ = data_B
                    optimizer_classifier.zero_grad()
                    features_A = model.encoder(inputs_A)
                    features_B = model.encoder(inputs_B)

                    # 计算注意力权重
                    attention_weights_A = attention_net(features_A)
                    attention_weights_B = attention_net(features_B)

                    # 领域分类器对抗训练
                    domain_output_A = domain_classifier(features_A)  # 计算样本A的权重
                    domain_output_B = domain_classifier(features_B)  # 计算样本B的权重

                    # 应用注意力权重到领域分类器输出
                    domain_weights_A = domain_output_A * attention_weights_A
                    domain_weights_B = domain_output_B * attention_weights_B

                    # 根据权重调整对抗训练损失函数
                    weighted_domain_loss_A = criterion(domain_weights_A, torch.zeros(
                        (len(inputs_A), 1))) * weight_adjust_factor_A  # 使用权重调整因子调整样本A的损失
                    weighted_domain_loss_B = criterion(domain_weights_B, torch.ones(
                        (len(inputs_B), 1))) * weight_adjust_factor_B  # 样本B的损失不加权

                    domain_loss = weighted_domain_loss_A + weighted_domain_loss_B
                    domain_loss.backward(retain_graph=True)
                    optimizer_classifier.step()
                    if (epoch == num_epoches - 1):
                        domain_weights_A_all.append(domain_weights_A.detach().cpu().numpy())
                        domain_weights_B_all.append(domain_weights_B.detach().cpu().numpy())
                print('epoch:{}/{}'.format(epoch, num_epoches), "domain_loss", domain_loss.item())
                if (epoch == num_epoches - 1):
                    domain_weights_A_all = np.concatenate(domain_weights_A_all)
                    domain_weights_B_all = np.concatenate(domain_weights_B_all)
            return {"domain_weights_A": domain_weights_A_all, "domain_weights_B": domain_weights_B_all}

        # 实例化自编码器M和领域分类器

        num_epoches = 50
        learning_rate = 0.001
        feature_size = 10
        model = AE.autoencoder(feature_size).to(device)
        domain_classifier = DomainClassifier(input_size=int(feature_size * 0.1), hidden_size=100)
        attention_net = AttentionNetwork(int(feature_size * 0.1), 1, 1)
        criterion_domain = nn.BCELoss()
        optimizer_classifier = optim.Adam(list(model.parameters()) + list(domain_classifier.parameters()),
                                          lr=learning_rate)
        if torch.cuda.is_available(): X_o = X_o.cuda()  # .cuda()函数：表示将数据拷贝在GPU上
        # TensorDataset():进行数据的封装
        X_con_input, X_con_output = X_o['input'], X_o['output']
        dataset_A = Data.TensorDataset(torch.tensor(X_con_input, dtype=torch.float), torch.tensor(X_con_output))
        data_loader_A = Data.DataLoader(dataset_A, batch_size=1024, shuffle=True, pin_memory=True)
        # print(X_con_input.shape)
        if torch.cuda.is_available(): X_n = X_n.cuda()  # .cuda()函数：表示将数据拷贝在GPU上
        X_tre_input, X_tre_output = X_n['input'], X_n['output']
        dataset_B = Data.TensorDataset(torch.tensor(X_tre_input, dtype=torch.float), torch.tensor(X_tre_output))
        data_loader_B = Data.DataLoader(dataset_B, batch_size=1024, shuffle=True, pin_memory=True)

        result = train_domain_adversarial(model, domain_classifier,attention_net,data_loader_A, data_loader_B,
                                          optimizer_classifier,
                                          criterion_domain, num_epoches, 0.5, 0.3)
        domain_weights_A = result["domain_weights_A"]
        domain_weights_B = result["domain_weights_B"]
        top_samples_A, top_samples_B, top_indices_A, top_indices_B = get_explain_sample(domain_weights_A,
                                                                                        domain_weights_B,label_num)

        result = estimulate_labling(X_n, y_n, top_samples_B, top_indices_B, label_num)
        top_samples_B = result[0]
        mixed_distribution = {
            'input': np.concatenate((top_samples_A['input'], top_samples_B['input']), axis=0),
            'output': np.concatenate((top_samples_A['output'], top_samples_B['output']), axis=0)
        }
        print(len(top_samples_A['input']))
        print(len(top_samples_B['input']))
        print('Mixed_Data', len(mixed_distribution['input']))

        # 计算这些样本之间的Wasserstein距离
        w_dist = wasserstein_distance(mixed_distribution['input'].ravel(),
                                      X_n['input'][:len(mixed_distribution)].ravel())
        print("Wasserstein distance between the two distributions is: ", w_dist)

        self.remain_X_con = top_samples_A
        self.remain_X_tre = top_samples_B
        self.remain_X = mixed_distribution
        self.EXPLAIN_RES = {
            'remain_X_tre': top_samples_B,
            'remain_X_con': top_samples_A,
            'remain_X': mixed_distribution
        }

        return self.EXPLAIN_RES

    def adapter_t(self, model, label_num, prev_params, memory_factors, batch_size=AdaParams['batch_size']):
        def np2ts(X):
            return torch.from_numpy(X).type(torch.float).to(device)

        def calculate_mas_loss(model, prev_params, memory_factors, alpha):
            mas_loss = 0
            for param, prev_param, mem_factor in zip(model.parameters(), prev_params, memory_factors):
                mem_factor = torch.Tensor(mem_factor)
                mas_loss += torch.sum((param - prev_param) ** 2 * mem_factor)
            return 0.5 * alpha * mas_loss

        # # OWAD
        # remain_X = np2ts(self.remain_X)
        remain_X = self.remain_X
        self.model = copy.deepcopy(model)
        optimizier = optim.Adam(self.model.parameters(), lr=0.001)
        # print(type(remain_X))
        torch_dataset = Data.TensorDataset(remain_X, remain_X)  # x_train为何是x_train的标签？----重构学习
        # dataloader将封装好的数据进行加载
        loader = Data.DataLoader(
            dataset=torch_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        num_epochs = 50
        alpha = 0.1
        alpha1 = 0.5
        alpha2 = 0.4
        alpha3 = 0.1
        lambda_reg = 0.001 # 调整L2正则化项的权重

        for epoch in range(num_epochs):
            for step, (batch_x, batch_y) in enumerate(loader):
                output = self.model(batch_x)  # 得到经过自编码器的输出值
                mas_loss = calculate_mas_loss(self.model, prev_params, memory_factors, alpha)
                reconstruction_loss = nn.MSELoss()(output, batch_x)
                anomaly_score = torch.abs(batch_x - output).sum(dim=1).mean()
                optimizier.zero_grad()
                if (label_num >= 30000):
                    # 添加L2正则化项
                    l2_reg = 0
                    for param in self.model.parameters():
                        l2_reg += torch.norm(param, p=2)
                    loss = alpha1 * reconstruction_loss + alpha2 * anomaly_score + lambda_reg * l2_reg+ alpha3 * mas_loss
                else:
                    loss = alpha1 * reconstruction_loss + alpha2 * anomaly_score + alpha3 * mas_loss
                loss.backward(retain_graph=True)  # 添加retain_graph参数
                optimizier.step()
            print('epoch:{}/{}'.format(epoch, num_epochs), '|Loss:', loss.item())

    def adapter_s(self,
                  model,
                  prev_params, memory_factors,
                  lr=AdaParams['lr'],
                  batch_size=AdaParams['batch_size']):
        remain_X = self.remain_X
        self.model = copy.deepcopy(model)
        X_input, X_output = remain_X['input'], remain_X['output']

        dataset = Data.TensorDataset(torch.tensor(X_input, dtype=torch.long), torch.tensor(X_output))
        dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        alpha = 0.1
        alpha1 = 0.5
        alpha2 = 0.4
        alpha3 = 0.1
        num_epochs = 30

        def calculate_mas_loss(model, prev_params, memory_factors, alpha):
            mas_loss = 0
            for param, prev_param, mem_factor in zip(model.parameters(), prev_params, memory_factors):
                mem_factor = torch.Tensor(mem_factor)
                mas_loss += torch.sum((param - prev_param) ** 2 * mem_factor)
            return 0.5 * alpha * mas_loss

        for epoch in range(num_epochs):
            for step, (seq, label) in enumerate(dataloader):
                seq = seq.clone().detach().view(-1, utils.get_params('LSTM')['window_size']).to(device)
                seq = F.one_hot(seq, num_classes=utils.get_params('LSTM')['num_classes']).float()
                output = self.model(seq)
                loss_1 = criterion(output, label.to(device))
                mas_loss = calculate_mas_loss(self.model, prev_params, memory_factors, alpha)
                broadcasted_label = label.to(device).unsqueeze(1).expand(-1, 1605)
                loss_2 = torch.mean(torch.abs(broadcasted_label - output), dim=1)
                loss = alpha1 * loss_1 + alpha2 * torch.mean(loss_2)+alpha3 * mas_loss
                loss = torch.mean(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print('epoch:{}/{}'.format(epoch, num_epochs), '|Loss:', loss.item())
