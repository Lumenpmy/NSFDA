import numpy as np
import torch
from scipy.stats import wasserstein_distance
import myutils as utils
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


class ShiftDetector:
    def __init__(self, method=None):

        assert method in ['Uncalibrate', 'linear interpolation', 'min_max', 'Sigmoid']
        self.method = method
        if self.method == 'Uncalibrate':
            pass
        elif self.method == 'linear interpolation':
            pass
        elif self.method == 'min_max':
            self.max = 1
        elif self.method == 'Sigmoid':
            pass
        else:
            print('Error Params <method>')
            exit(-1)

    def process(self,
                model_res):
        # model_res是指异常检测模型的输出概率
        if self.method == 'Uncalibrate':  # return Uncalibrated model outputs
            return model_res
        elif self.method == 'linear interpolation':
            model_res = torch.tensor(model_res)
            data_sorted, indices = torch.sort(model_res)
            new_data = torch.zeros(len(model_res))
            unique_data, unique_indices = torch.unique(data_sorted, return_inverse=True)
            j = 0
            for i in range(len(unique_data)):
                unique_index = (unique_indices == i).nonzero(as_tuple=False)
                min_bound = j * (1 / len(data_sorted))
                max_bound = (j + 1) * (1 / len(data_sorted))
                if (len(unique_index > 1)):
                    j += (len(unique_index) - 1)
                j += 1
                normalized_data = (unique_data[i] - unique_data[0]) / (unique_data[-1] - unique_data[0])
                new_data[indices[unique_index]] = min_bound + normalized_data * (max_bound - min_bound)
            mean = torch.mean(new_data)
            variance = torch.var(new_data)
            print("均值：", mean.item())
            print("方差：", variance.item())
            return np.asarray(new_data)
        elif self.method == 'min_max':
            # 数据从小到大排序
            model_res = torch.tensor(model_res)
            data_sorted, indices = torch.sort(model_res)
            # 创建一个新的tensor来存放映射后的数据
            new_data = torch.zeros(len(model_res))
            # 将数据映射到区间内的一个点，而不是区间的边界
            for i in range(len(data_sorted)):
                # min-max归一化数据
                normalized_data = (data_sorted[i] - data_sorted[0]) / (data_sorted[-1] - data_sorted[0])
                # 将归一化的数据映射到新的区间内
                new_data[indices[i]] = normalized_data
            mean = torch.mean(new_data)
            variance = torch.var(new_data)
            print("均值：", mean.item())
            print("方差：", variance.item())
            return np.asarray(new_data)
        elif self.method == 'Sigmoid':
            model_res = torch.tensor(model_res)
            new_data = torch.sigmoid(model_res)
            mean = torch.mean(new_data)
            variance = torch.var(new_data)
            print("均值：", mean.item())
            print("方差：", variance.item())
            return np.asarray(new_data)



    def Monte_Carlo(self, x, y, alpha=0.05, iterations=1000):
        np.random.seed(0)
        n = len(x)
        m = len(y)
        observed_stat = wasserstein_distance(x, y)
        print("Wasserstein distance between old set and new set is:", observed_stat)
        # 计算x和y的直方图
        bins = np.linspace(min(x.min(), y.min()), max(x.max(), y.max()), 100)
        x_hist, _ = np.histogram(x, bins=bins, density=True)
        y_hist, _ = np.histogram(y, bins=bins, density=True)

        def generate_samples(n):
            # 生成两个正太分布的随机样本
            x = np.random.normal(0, 1, n)
            y = np.random.normal(0, 1, n)
            return x, y

        count = 0
        for i in range(iterations):
            x_sample, y_sample = generate_samples(max(n, m))
            simulated_stat = wasserstein_distance(x_sample, y_sample)
            bins = np.linspace(min(x_sample.min(), y_sample.min()), max(x_sample.max(), y_sample.max()), 10)
            x_hist, _ = np.histogram(x_sample, bins=bins, density=True)
            y_hist, _ = np.histogram(x_sample, bins=bins, density=True)
            if simulated_stat >= observed_stat:
                count += 1

        p_value = (count + 1) / (iterations + 1)
        conf_int = np.percentile(p_value, [alpha / 2 * 100, (1 - alpha / 2) * 100])
        return p_value, conf_int

        # 可视化直方图

    def visualize_hists(self,
                        res_1,
                        res_2,
                        color_1='#F8BA63',
                        color_2='#AAAAAA'):
        self.bin_num = utils.get_params('ShiftDetector')['test_bin_num']  # 为10
        self.bin_array = np.linspace(0., 1., self.bin_num + 1)  # 将[0,1]分成含有11个元素的均匀分布的序列
        legend_1 = 'old shifting score'  # (Calibrated)
        legend_2 = 'new shifting score'  # (Calibrated)

        res_1 = list(np.histogram(res_1, bins=self.bin_array))  # 将元组结果转成列表
        res_1[0] = res_1[0] / np.sum(res_1[0])  # cres[0]表示列表形式的每个区间的元素个数，更新后，得到频率分布的列表
        res_2 = list(np.histogram(res_2, bins=self.bin_array))
        res_2[0] = res_2[0] / np.sum(res_2[0])

        x = (res_1[1][:-1] + res_1[1][1:]) / 2
        width = x[1] - x[0]  # 区间宽度为0.1

        plt.figure('MenuBar', figsize=(10, 6))
        plt.grid(True, linewidth=1, linestyle='--')  # 网格线条

        plt.bar(x, res_1[0], width=width, alpha=0.7, ec='black', label=legend_1, color=color_1)
        plt.bar(x, res_2[0], width=width, alpha=0.5, ec='black', label=legend_2, color=color_2, hatch='//')

        def get_smooth_axis(res):
            x = (res[1][:-1] + res[1][1:]) / 2
            x = np.insert(x, 0, 0.)  # 使用数值插入函数，在x中的第0个位置插入0.
            x = np.insert(x, len(x),
                          1.)  # 最后一个插入1.,得到此时的x为[0，0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95，1]
            y = res[0]  # 表示每个区间的元素个数
            y = np.insert(y, 0, 0.)
            y = np.insert(y, len(y), 0.)
            # 进行曲线的平滑处理
            X_Y_Spline = make_interp_spline(x, y)
            X_ = np.linspace(x.min(), x.max(), 300)
            Y_ = X_Y_Spline(X_)
            return X_, Y_

        X, Y = get_smooth_axis(res_1)
        plt.plot(X, Y, '-', linewidth=8, color=color_1)
        X, Y = get_smooth_axis(res_2)
        plt.plot(X, Y, '-', linewidth=8, color=color_2)

        plt.ylim(ymin=0)
        plt.xlabel('Shifting Score', fontsize=20, fontweight='bold')  # 添加横坐标标签
        plt.ylabel('Frequency', fontsize=20, fontweight='bold')  # 添加纵坐标标签
        plt.legend(prop={'size': 20, 'weight': 'bold'})
        plt.xticks(fontsize=20, fontweight='bold')  # 增大字体并加粗
        plt.yticks(fontsize=20, fontweight='bold')  # 增大字体并加粗

        plt.show()
