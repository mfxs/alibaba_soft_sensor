# 导入库
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

# 设置绘图风格
plt.style.use('seaborn-dark')


# 绘制邻接矩阵
def plot_adj(adj):
    sns.heatmap(pd.DataFrame(adj, index=element, columns=element), cmap='YlGnBu', annot=True, square=True, fmt='.2f')
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    plt.xticks(fontsize=20, fontweight='black')
    plt.yticks(fontsize=20, fontweight='black', rotation='horizontal')


# 绘制损失曲线
def plot_loss(loss, rep, n):
    plt.figure(n)
    plt.plot(range(1, loss.shape[1] + 1), loss.iloc[rep, :], lw=3)
    ax = plt.gca()
    x_locator = MultipleLocator(20)
    ax.xaxis.set_major_locator(x_locator)
    plt.grid()
    plt.xlim(0, loss.shape[1])
    # plt.ylim(0, 500)
    plt.legend(['Training loss'], fontsize=20)
    plt.xticks(fontsize=20, fontweight='black')
    plt.yticks(fontsize=20, fontweight='black')
    plt.xlabel('Epoch', fontsize=30, fontweight='black')
    plt.ylabel('Loss', fontsize=30, fontweight='black')
    plt.title('The loss curve on training set', fontsize=35, fontweight='black')
    plt.show()


# 绘制r2&rmse曲线
def plot_r2_rmse(mean, std, ylabel, title):
    mean.T.plot.bar(yerr=std.T, ecolor='magenta', capsize=3)
    # plt.ylim(0, 1)
    plt.xticks(fontsize=20, fontweight='black', rotation='horizontal')
    plt.yticks(fontsize=20, fontweight='black')
    plt.xlabel('Element composition', fontsize=30, fontweight='black')
    plt.ylabel(ylabel, fontsize=30, fontweight='black')
    plt.legend(fontsize=16, loc=2)
    plt.title(title, fontsize=35, fontweight='black')
    plt.grid()
    plt.show()


# 导入数据
path = 'pearson/'
adj = np.load(path + 'adj.npy').item()
r2 = np.load(path + 'r2.npy').item()
rmse = np.load(path + 'rmse.npy').item()
loss = np.load(path + 'loss.npy').item()

# 模型名和元素名
model = ['PLS', 'SVR', 'FC', 'ELM', 'GCN', 'Multi-channel GCN']
element = r2['pls'].columns

# 计算r2和rmse的均值和标准差
f_mean = lambda x: x.mean()
f_std = lambda x: x.std()
r2_mean = pd.DataFrame(np.stack(list(map(f_mean, r2.values()))), columns=element, index=model)
rmse_mean = pd.DataFrame(np.stack(list(map(f_mean, rmse.values()))), columns=element, index=model)
r2_std = pd.DataFrame(np.stack(list(map(f_std, r2.values()))), columns=element, index=model)
rmse_std = pd.DataFrame(np.stack(list(map(f_std, rmse.values()))), columns=element, index=model)

# 存储均值和方差结果
result_r2 = r2_mean.copy()
result_rmse = r2_mean.copy()
for i in range(r2_mean.shape[0]):
    for j in range(r2_mean.shape[1]):
        result_r2.iloc[i, j] = str(round(r2_mean.iloc[i, j], 2)) + ' ± ' + str(round(r2_std.iloc[i, j], 3))
        result_rmse.iloc[i, j] = str(round(rmse_mean.iloc[i, j], 3)) + ' ± ' + str(round(rmse_std.iloc[i, j], 4))
result_r2.to_csv(path + 'result_r2.csv', encoding='gbk')
result_rmse.to_csv(path + 'result_rmse.csv', encoding='gbk')

# 画图
plot_adj(adj['mcgcn'][0])
plot_loss(loss['gcn_train'], 0, 2)
plot_loss(loss['mcgcn_train'], 0, 3)
plot_r2_rmse(r2_mean, r2_std, 'R2', 'R2 on different element compositions')
plot_r2_rmse(rmse_mean, rmse_std, 'RMSE', 'RMSE on different element compositions')
