import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import ks_2samp, combine_pvalues
from scipy.spatial.distance import cosine


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class ShiftDetector:
    def __init__(self, model):
        self.new_features = None
        self.old_features = None
        self.model = model

    def detect(self, X_o, X_n, rmse_o, rmse_n, significance_level=0.05, permutation_count=1000):
        self.model.eval()
        with torch.no_grad():
            X_o = torch.from_numpy(X_o).type(torch.float).to(device)
            X_n = torch.from_numpy(X_n).type(torch.float).to(device)

            # KS检验
            ks_stat, ks_p_value = ks_2samp(rmse_o, rmse_n)
            # 余弦相似度
            softmax = torch.nn.Softmax(dim=1)
            self.old_features = softmax(self.model.encode(X_o)).numpy()
            self.new_features = softmax(self.model.encode(X_n)).numpy()
            
            actual_cos_similarity = cosine(self.old_features.mean(axis=0), self.new_features.mean(axis=0))

            # 置换测试计算余弦相似度的p值
            combined_features = np.concatenate((self.old_features, self.new_features))
            count_extreme_values = 0
            for _ in range(permutation_count):
                np.random.shuffle(combined_features)
                permuted_old_features = combined_features[:len(self.old_features)]
                permuted_new_features = combined_features[len(self.old_features):]
                permuted_cos_similarity = cosine(permuted_old_features.mean(axis=0), permuted_new_features.mean(axis=0))
                if abs(permuted_cos_similarity) >= abs(actual_cos_similarity):
                    count_extreme_values += 1
            cos_p_value = count_extreme_values / permutation_count

            # 基于信息熵的方法
            old_entropy = -np.sum(self.old_features * np.log(self.old_features.clip(1e-10)), axis=1).mean()
            new_entropy = -np.sum(self.new_features * np.log(self.new_features.clip(1e-10)), axis=1).mean()
            entropy_threshold = old_entropy * 0.05
            entropy_change = new_entropy > old_entropy + entropy_threshold

            # 统计显著性集成
            # 这里我们使用Fisher方法合并KS检验和余弦相似度的p值（余弦相似度的p值需要另外计算）
            combined_p_value = combine_pvalues([ks_p_value, cos_p_value], method='fisher')[1]

            # 输出校准（投票）
            votes = 0
            # votes += not ci_intersection  # 如果没有交集，则计一票
            votes += combined_p_value < significance_level  # 如果组合p值低于显著性水平，则计一票
            votes += entropy_change  # 如果熵变化超过阈值，则计一票

            # 根据投票结果判断是否发生了normality shift
            return votes >= 1  # 需要大多数投票

    # 为了简化，这里假定所有特征都是二维的
    def visualize_distributions(self, rmse_o, rmse_n):

        sns.set(style="whitegrid")
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # RMSE分布图
        sns.histplot(rmse_o, color="skyblue", label="Old RMSE", kde=True, ax=axs[0, 0])
        sns.histplot(rmse_n, color="salmon", label="New RMSE", kde=True, ax=axs[0, 0])
        axs[0, 0].legend()
        axs[0, 0].set_title("Distribution of RMSE for Old and New Data")
        axs[0, 0].set_xlabel("RMSE")
        axs[0, 0].set_ylabel("Density")

        # KS检验的CDF图
        sns.ecdfplot(rmse_o, color="skyblue", label="Old Data CDF", ax=axs[0, 1])
        sns.ecdfplot(rmse_n, color="salmon", label="New Data CDF", ax=axs[0, 1])
        axs[0, 1].legend()
        axs[0, 1].set_title("CDF Comparison for Old and New Data")
        axs[0, 1].set_xlabel("RMSE")
        axs[0, 1].set_ylabel("CDF")

        # 使用PCA进行降维，将特征空间从多维降到二维(保持共同的PCA模型）
        # 确保主成分（即PCA变换后的新特征）对两组数据都有相同的含义
        combined_features = np.vstack((self.old_features, self.new_features))
        # 使用PCA进行降维
        pca = PCA(n_components=2)
        pca.fit(combined_features)

        # 分别转换特征
        old_features_pca = pca.transform(self.old_features)
        new_features_pca = pca.transform(self.new_features)

        # 特征空间散点图
        axs[1, 0].scatter(old_features_pca[:, 0], old_features_pca[:, 1], color="skyblue", alpha=0.5,
                          label="Old Features")
        axs[1, 0].scatter(new_features_pca[:, 0], new_features_pca[:, 1], color="salmon", alpha=0.5,
                          label="New Features")
        axs[1, 0].legend()
        axs[1, 0].set_title("Feature Space Distribution after PCA")
        axs[1, 0].set_xlabel("Principal Component 1")
        axs[1, 0].set_ylabel("Principal Component 2")

        # 计算信息熵和绘制箱型图
        old_entropy = np.sum(-self.old_features * np.log(np.clip(self.old_features, 1e-10, None)), axis=1)
        new_entropy = np.sum(-self.new_features * np.log(np.clip(self.new_features, 1e-10, None)), axis=1)
        sns.boxplot(data=[old_entropy, new_entropy], palette=["skyblue", "salmon"], ax=axs[1, 1])
        axs[1, 1].set_xticklabels(["Old Entropy", "New Entropy"])
        axs[1, 1].set_title("Entropy Distribution for Old and New Data")
        axs[1, 1].set_ylabel("Entropy")

        plt.tight_layout()
        plt.show()
