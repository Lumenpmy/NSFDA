import torch
import random
import yaml
import os
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

# 输出该文件的绝对路径除去文件名----得到项目所在的位置(OWAN-main)
PROJECT_FILE = os.path.split(os.path.realpath(__file__))[0]
CONFIG_FILE = os.path.join(PROJECT_FILE, 'configs.yml')  # 得到配置文件的路径


def get_params(param_type):
    f = open(CONFIG_FILE, encoding="utf-8")
    params = yaml.safe_load(f)  # 读取config.yml的内容
    return params[param_type]  # 根据传入的参数类型，例如AE，到路径为CONFIG_FILE的文件中寻找与AE相关的参数并返回


def TPR_FPR(y_prob, y_true, thres, verbose=True):
    y_true = np.asarray(y_true)  # 转换为数组类型
    y_prob = np.asarray(y_prob)
    y_pred = np.where(y_prob >= thres, 1, 0)  # 将y的预测值与阈值进行比较（异常置信度模型）

    # 输出混淆矩阵中的值
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    # 假阳性率和真阳性率
    fpr = (fp / (fp + tn + 1e-10))
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * precision * recall / (precision + recall)

    if verbose:
        print("*********************** The relevant test indicators are as follows ***********************")
        print('FPR:', fpr, )
        print('Precision:',precision)
        print('Recall:',recall)
        print('F1_Score:',f1_score)

    return fpr

def multi_fpr_tpr(y_prob, y_true, thres_max, thres_min=0., split=1000, is_P_mal=True):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    fpr = []
    tpr = []

    # 一般是将[0,1]之间生成的均匀分布的数字（个数取决于num)中逐个设置为阈值后,挨个计算FPR TPR，然后将其绘制成一条曲线后得到ROC曲线
    thresholds = np.linspace(thres_min, thres_max, split)  # 输出一个包含1000个在thres_min和thres_max间的数字的数组
    for threshold in thresholds:
        if is_P_mal:
            y_pred = np.where(y_prob >= threshold, 1, 0)  # 选择使用异常置信度模型 如果预测值大于阈值，则是异常
        else:
            y_pred = np.where(y_prob <= threshold, 1, 0)

        fp = np.sum((y_pred == 1) & (y_true == 0))
        tp = np.sum((y_pred == 1) & (y_true == 1))

        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        fpr.append(fp / (fp + tn))  # 利用append方法将得到结果进行拼接
        tpr.append(tp / (tp + fn))

    return fpr, tpr  # 最后返回一系列的fpr和tpr


# AUC指标
def multi_metrics(probs,  # 预测的一系列值
                  labels,  # 本身的一系列值
                  thres_max=1.,
                  thres_min=0.,
                  split=1000,
                  is_P_mal=True,
                  condition=None,
                  # plot_file=None,
                  plot_file='FRONTEND',
                  ):
    fprs, tprs = multi_fpr_tpr(probs, labels, thres_max, thres_min=thres_min, split=split, is_P_mal=is_P_mal)
    roc_auc = metrics.auc(fprs, tprs)  # 利用sklearn.metrics.auc()得到AUC指标
    print('AUC:', roc_auc)

    # 画ROC曲线
    if plot_file:
        plt.figure()
        plt.plot(fprs, tprs)
        plt.title('Receiver Operating Characteristic')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.text(0.6, 0.2, 'AUC = %0.2f' % roc_auc, fontsize=12)  # 在指定位置添加AUC值
        if plot_file == 'FRONTEND':
            # plt.savefig(r'C:\Users\Depth\Desktop\OWAD模型\OWAD模型实验--方案三\images/result_plot_2.png')
            plt.show()
        else:
            plt.savefig(plot_file)

    if condition is not None:
        fprs, tprs = np.asarray(fprs), np.asarray(tprs)
        if 'tpr' in condition:
            print('fpr: %.4f' % np.min(fprs[tprs >= condition['tpr']]), '(@tpr %.4f)' % condition['tpr'])
        if 'fpr' in condition:
            print('tpr: %.4f' % np.max(tprs[fprs <= condition['fpr']]), '(@fpr %.4f)' % condition['fpr'])

    # return fprs, tprs


# 下面函数的每个设置都是为了保证每次运行网络的时候相同输入的输出是固定的
def set_random_seed(seed=42, deterministic=True):
    random.seed(seed)  # 为随机数设定随机数种子
    np.random.seed(seed)  # 设置生成的数组的随机数种子
    torch.manual_seed(seed)  # 为cpu设置随机数种子
    torch.cuda.manual_seed_all(seed)  # 为所有的Gpu设备设置随机数种子
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
