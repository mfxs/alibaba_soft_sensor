# 导入库
import copy
import math
import torch
import datetime
import warnings
import cvxpy as cvx
import numpy as np
import pandas as pd
from torch import nn
import torch.optim as optim
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPRegressor
from torch.nn import Linear, ReLU, Sequential, Dropout
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

# 忽略警告，设置GPU
warnings.filterwarnings('ignore')
cpu = torch.device('cpu')
gpu = torch.device('cuda:0')


# 计算R2和RMSE
def r2_rmse(y_true, y_pred):
    r2 = r2_score(y_true, y_pred, multioutput='raw_values')
    rmse = np.sqrt(mean_squared_error(y_true, y_pred, multioutput='raw_values'))
    for i in range(r2.shape[0]):
        print('{}: R2: {:.2f} RMSE: {:.4f}'.format(element[i], r2[i], rmse[i]))
        f.write('{}: R2: {:.2f} RMSE: {:.4f}\n'.format(element[i], r2[i], rmse[i]))
    print('Averaged R2: {:.2f}'.format(np.mean(r2)))
    f.write('Averaged R2: {:.2f}\n'.format(np.mean(r2)))
    print('Averaged RMSE: {:.4f}'.format(np.mean(rmse)))
    f.write('Averaged RMSE: {:.4f}\n\n'.format(np.mean(rmse)))
    return r2, rmse


# 计算邻接矩阵并标准化
def adjacency_matrix(X, mode, epsilon=0.1, scale=0.4, c=0.05, self_con=0.2):
    x = X.cpu().numpy()
    if mode == 'rbf':
        x = x.T
        k = RBF(length_scale=scale)
        A = k(x, x)
        A[A < epsilon] = 0
    elif mode == 'pearson':
        x = x.T
        A = np.corrcoef(x)
        A[np.abs(A) < epsilon] = 0
    elif mode == 'sparse coding':
        A = cvx.Variable((x.shape[1], x.shape[1]))
        term1 = cvx.norm(x * A - x, p='fro')
        term2 = cvx.norm1(A)
        constraints = []
        for i in range(x.shape[1]):
            constraints.append(A[i, i] == 0)
            for j in range(x.shape[1]):
                constraints.append(A[i, j] >= 0)
        constraints.append(A == A.T)
        objective = cvx.Minimize(term1 + c * term2)
        prob = cvx.Problem(objective, constraints)
        prob.solve()
        A = A.value
        A = A + self_con * np.eye(x.shape[1])
        A[A < epsilon] = 0
    D = np.diag(np.sum(A, axis=1) ** (-0.5))
    A = np.matmul(np.matmul(D, A), D)
    A = torch.tensor(A, device=gpu)
    return A


# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index, :], self.label[index, :]

    def __len__(self):
        return self.data.shape[0]


# 自定义图卷积运算
class GraphConvolution(nn.Module):
    def __init__(self, n_input, n_output):
        super(GraphConvolution, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.weight = Parameter(torch.FloatTensor(n_input, n_output))
        self.reset_parameters()

    def forward(self, x, adj):
        temp = torch.matmul(x, self.weight)
        res = torch.matmul(adj.float(), temp.float())
        return res

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)


# 自定义Loss(+L2范数)
class MyLoss(nn.Module):
    def __init__(self, gamma=10):
        super(MyLoss, self).__init__()
        self.gamma = gamma

    def forward(self, y1, y2, model):
        mseloss = nn.MSELoss(reduction='sum')
        loss1 = mseloss(y1, y2)
        loss2 = 0
        for gc in model.gc:
            loss2 += torch.sum(gc.weight ** 2)
        return loss1 + self.gamma * loss2


# 自定义图卷积神经网络模型
class GCN(nn.Module):
    def __init__(self, n_variable, hid, n_output):
        super(GCN, self).__init__()
        self.n_variable = n_variable
        self.n_hid = [n_variable] + list(hid) + [n_output]
        self.n_output = n_output
        self.gc = nn.ModuleList()
        for i in range(len(hid) + 1):
            self.gc.append(GraphConvolution(self.n_hid[i], self.n_hid[i + 1]))
        self.act = ReLU()

    def forward(self, x):
        adj = adjacency_matrix(x.T, 'rbf')
        h = x
        for i in range(len(self.n_hid) - 2):
            h = self.act(self.gc[i](h, adj))
        res = self.gc[-1](h, adj)
        return res


# 自定义多通道图卷积神经网络模型
class MCGCN(nn.Module):
    def __init__(self, n_variable, in_fc, gc, out_fc, n_output, direct_link=False, dropout=False):
        super(MCGCN, self).__init__()
        self.n_variable = n_variable
        self.n_in_fc = [n_variable] + list(in_fc)
        self.n_gc = [in_fc[-1]] + list(gc)
        self.n_out_fc = [gc[-1]] + list(out_fc)
        self.n_output = n_output
        self.dl = direct_link
        self.dropout = dropout
        self.act = ReLU()
        self.drop = Dropout(p=0.5)

        # 输入全连接层
        self.in_fc = nn.ModuleList()
        for i in range(len(in_fc)):
            temp = nn.ModuleList()
            for j in range(self.n_output):
                temp.append(Sequential(Linear(self.n_in_fc[i], self.n_in_fc[i + 1]), ReLU()))
            self.in_fc.append(temp)

        # 图卷积层
        self.gc = nn.ModuleList()
        for i in range(len(gc)):
            self.gc.append(GraphConvolution(self.n_gc[i], self.n_gc[i + 1]))

        # 输出全连接层
        self.out_fc = nn.ModuleList()
        for i in range(len(out_fc)):
            temp = nn.ModuleList()
            for j in range(self.n_output):
                temp.append(Sequential(Linear(self.n_out_fc[i], self.n_out_fc[i + 1]), ReLU()))
            self.out_fc.append(temp)

        # 输出层
        self.out = nn.ModuleList()
        for j in range(self.n_output):
            if self.dl:
                self.out.append(Linear(out_fc[-1] + n_variable, 1))
            else:
                self.out.append(Linear(out_fc[-1], 1))

    def forward(self, x, adj):
        feat_list = []

        # 输入全连接层
        for i in range(self.n_output):
            feat = x
            for fc in self.in_fc:
                feat = fc[i](feat)
            feat_list.append(feat)
        feat = torch.stack(feat_list, dim=1)

        # 图卷积层
        for gc in self.gc:
            feat = gc(feat, adj)
            feat = self.act(feat)

        # 输出全连接层
        res_list = []
        for i in range(self.n_output):
            res = feat[:, i, :]
            for fc in self.out_fc:
                res = fc[i](res)
            if self.dl:
                res = torch.cat((res, x), 1)
            if self.dropout:
                res = self.drop(res)
            res = self.out[i](res)
            res_list.append(res.squeeze())
        res = torch.stack(res_list, dim=1)
        return res


# 极限学习机
class ELM(BaseEstimator, RegressorMixin):
    def __init__(self, n_input, n_output, n_hid=512, c=1):
        super(ELM, self).__init__()
        self.c = c
        self.n_input = n_input
        self.n_hid = n_hid
        self.n_output = n_output
        self.std = 1. / math.sqrt(self.n_input)
        self.w1 = np.random.uniform(-self.std, self.std, (self.n_input + 1, self.n_hid))

    def relu(self, x):
        return np.maximum(0, x)

    def add_bias(self, x):
        m = x.shape[0]
        x = np.concatenate((x, np.ones((m, 1))), axis=1)
        return x

    def fit(self, x, y):
        x = self.add_bias(x)
        feat = self.relu(np.matmul(x, self.w1))
        feat = self.add_bias(feat)
        self.w2 = np.matmul(np.linalg.inv(np.matmul(feat.T, feat) + np.eye(feat.shape[1]) / self.c),
                            np.matmul(feat.T, y))

    def predict(self, x):
        x = self.add_bias(x)
        feat = self.relu(np.matmul(x, self.w1))
        feat = self.add_bias(feat)
        y = np.matmul(feat, self.w2)
        return y


# 导入数据
data = pd.read_csv('data_preprocess.csv', index_col=0)
data.drop(['记录时间', '熔炼号', '钢种', '实际值-低碳锰铁', 'sb_record_time', 'sb_record_time_x', 'sb_record_time_y'], axis=1,
          inplace=True)
X = data.iloc[:, :-12]
y = data.iloc[:, -12:]
element = y.columns.map(lambda x: x[4:-2].capitalize())

# 变量曲线绘制
# plt.plot(data.iloc[:, 0])
# plt.show()

# 选择模型并输入参数
model = list(input('[1]PLS\n[2]SVR\n[3]FC\n[4]ELM\n[5]GCN\n[6]MCGCN\nSelect models:'))
is_default = bool(int(input('Use default setting?(0-No/1-Yes)')))
if is_default:
    graph_reg, loss_reg, self_con, n_rep, epoch1, epoch2, add_dl, add_dropout = 0.05, 10, 0.2, 3, 200, 200, False, False
else:
    graph_reg = float(input('Input the regularization coefficient of graph(Default 0.05):'))
    loss_reg = float(input('Input the regularization coefficient of loss(Default 10):'))
    self_con = float(input('Input the self-connection coefficient(Default 0.2):'))
    n_rep = int(input('Input the number of repeated times(Default 3):'))
    epoch1 = int(input('Input the number of epochs for GCN(Default 200):'))
    epoch2 = int(input('Input the number of epochs for MCGCN(Default 200):'))
    add_dl = bool(int(input('Add direct link from input to output?(0-No/1-Yes)')))
    add_dropout = bool(int(input('Add dropout in the last layer?(0-No/1-Yes)')))

# 初始化结果存储字典
f = open('params.txt', 'w+')
f.write('Parameters setting:\n')
f.write('Regularization coefficient of graph:' + str(graph_reg) + '\n')
f.write('Regularization coefficient of loss:' + str(loss_reg) + '\n')
f.write('Coefficient of self-connection:' + str(self_con) + '\n')
f.write('Number of epochs for GCN:' + str(epoch1) + '\n')
f.write('Number of epochs for MCGCN:' + str(epoch2) + '\n')
f.write('Add direct link from input to output:' + str(add_dl) + '\n')
f.write('Add dropout in the last layer:' + str(add_dropout) + '\n\n')
adj_ = {'mcgcn': []}
temp1 = pd.DataFrame(np.zeros((n_rep, y.shape[1])), columns=element)
r2 = {'pls': temp1.copy(), 'svr': temp1.copy(), 'fc': temp1.copy(), 'elm': temp1.copy(),
      'gcn': temp1.copy(), 'mcgcn': temp1.copy()}
rmse = copy.deepcopy(r2)
temp2 = pd.DataFrame(np.zeros((n_rep, epoch1)))
temp3 = pd.DataFrame(np.zeros((n_rep, epoch2)))
loss = {'gcn_train': temp2.copy(), 'gcn_test': temp2.copy(), 'mcgcn_train': temp3.copy(), 'mcgcn_test': temp3.copy()}

# 重复多次实验
for rep in range(n_rep):
    print('=====Experiment ({}/{})====='.format(rep + 1, n_rep))
    f.write('=====Experiment ({}/{})=====\n'.format(rep + 1, n_rep))

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rep)
    f.write('Seed:{}\n'.format(rep))

    # 数据标准化
    scaler_x = MinMaxScaler().fit(X_train)
    scaler_y = MinMaxScaler().fit(y_train)
    X_train = scaler_x.transform(X_train)
    X_test = scaler_x.transform(X_test)
    y_train = scaler_y.transform(y_train)
    y_test = scaler_y.transform(y_test)

    # PLS
    if '1' in model:
        print('=====Partial Least Squares=====')
        f.write('=====Partial Least Squares=====\n')
        param_pls = {'n_components': range(1, 21)}
        mdl = PLSRegression()
        reg = GridSearchCV(mdl, param_pls, 'r2', iid=True, cv=5).fit(X_train, y_train)
        print(reg.best_params_)
        f.write(str(reg.best_estimator_.get_params()) + '\n')
        y_pred = reg.predict(X_test)
        r2['pls'].iloc[rep, :], rmse['pls'].iloc[rep, :] = r2_rmse(scaler_y.inverse_transform(y_test),
                                                                   scaler_y.inverse_transform(y_pred))

    # SVR
    if '2' in model:
        print('=====Support Vector Regression=====')
        f.write('=====Support Vector Regression=====\n')
        y_pred = np.zeros(y_test.shape)
        param_svr = {'C': np.logspace(-2, 3, 6)}
        mdl = SVR(kernel='rbf', gamma='scale', epsilon=0.05)
        for i in range(y_train.shape[1]):
            reg = GridSearchCV(mdl, param_svr, 'r2', iid=True, cv=5, verbose=1, n_jobs=20).fit(X_train, y_train[:, i])
            print(reg.best_params_)
            f.write(str(reg.best_estimator_.get_params()) + '\n')
            y_pred[:, i] = reg.predict(X_test)
        r2['svr'].iloc[rep, :], rmse['svr'].iloc[rep, :] = r2_rmse(scaler_y.inverse_transform(y_test),
                                                                   scaler_y.inverse_transform(y_pred))

    # FC
    if '3' in model:
        print('=====Fully Connected Network=====')
        f.write('=====Fully Connected Network=====\n')
        param_fc = {'alpha': np.logspace(-3, 3, 7)}
        mdl = MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=500)
        reg = GridSearchCV(mdl, param_fc, 'r2', iid=True, cv=5).fit(X_train, y_train)
        print(reg.best_params_)
        f.write(str(reg.best_estimator_.get_params()) + '\n')
        y_pred = reg.predict(X_test)
        r2['fc'].iloc[rep, :], rmse['fc'].iloc[rep, :] = r2_rmse(scaler_y.inverse_transform(y_test),
                                                                 scaler_y.inverse_transform(y_pred))

    # ELM
    if '4' in model:
        print('=====Extreme Learning Machine=====')
        f.write('=====Extreme Learning Machine=====\n')
        param_elm = {'n_hid': np.logspace(5, 10, 6, base=2, dtype=int), 'c': np.logspace(-3, 3, 1000)}
        mdl = ELM(X_train.shape[1], y_train.shape[1])
        reg = RandomizedSearchCV(mdl, param_elm, 100, 'r2', iid=True, cv=5).fit(X_train, y_train)
        print(reg.best_params_)
        f.write(str(reg.best_estimator_.get_params()) + '\n')
        y_pred = reg.predict(X_test)
        r2['elm'].iloc[rep, :], rmse['elm'].iloc[rep, :] = r2_rmse(scaler_y.inverse_transform(y_test),
                                                                   scaler_y.inverse_transform(y_pred))

    # 数据迁移至GPU
    X_train = torch.tensor(X_train, dtype=torch.float32, device=gpu)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=gpu)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=gpu)
    y_test = torch.tensor(y_test, dtype=torch.float32, device=gpu)

    # GCN
    if '5' in model:
        print('=====Graph Convolution Network=====')
        f.write('=====Graph Convolution Network=====\n')

        # 模型超参数设置以及生成
        gc = (256, 256)
        gcn = GCN(X_train.shape[1], gc, y_train.shape[1]).to(gpu)
        f.write('Net structure of GCN:' + str(gc) + '\n')

        # 模型训练
        t0 = datetime.datetime.now()
        gcn.train()
        criterion1 = nn.MSELoss(reduction='sum')
        optimizer1 = optim.Adam(gcn.parameters(), lr=0.001, weight_decay=1e-3)
        scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=50, gamma=0.5)

        # 每个epoch
        for epoch in range(epoch1):
            t1 = datetime.datetime.now()

            # 生成数据集和数据加载器
            data_train = MyDataset(X_train, y_train)
            dataloader = DataLoader(data_train, batch_size=64, shuffle=True)

            # 每个batch
            for item in dataloader:
                batch_X, batch_y = item[0], item[1]
                optimizer1.zero_grad()

                # 计算训练误差并反向传播
                output_train = gcn(batch_X)
                loss_train = criterion1(output_train, batch_y)
                loss_train.backward()
                loss['gcn_train'].iloc[rep, epoch] += loss_train.item()

                # 模型参数调整
                optimizer1.step()
            scheduler1.step()

            # 计算测试误差
            output_test = gcn(X_test)
            loss_test = criterion1(output_test, y_test)
            loss['gcn_test'].iloc[rep, epoch] = loss_test.item()

            # 打印
            t2 = datetime.datetime.now()
            print('Epoch: {:03d} loss_train: {:.4f} loss_test: {:.4f} time: {}'.format(epoch + 1,
                                                                                       loss['gcn_train'].iloc[
                                                                                           rep, epoch],
                                                                                       loss_test.item(), t2 - t1))
        t3 = datetime.datetime.now()
        print('Optimization Finished! Time:', t3 - t0)

        # 模型测试
        gcn.eval()
        y_pred = gcn(X_test)
        r2['gcn'].iloc[rep, :], rmse['gcn'].iloc[rep, :] = r2_rmse(
            scaler_y.inverse_transform(y_test.cpu().detach().numpy()),
            scaler_y.inverse_transform(y_pred.cpu().detach().numpy()))
        gcn = gcn.to(cpu)

    # MC_GCN
    if '6' in model:
        print('=====Multi-channel Graph Convolution Network=====')
        f.write('=====Multi-channel Graph Convolution Network=====\n')

        # 计算邻接矩阵
        adj = adjacency_matrix(y_train, 'sparse coding', c=graph_reg, self_con=self_con)
        adj_['mcgcn'].append(adj.cpu().numpy())

        # 模型超参数设置以及生成
        in_fc = (1024,)
        gc = (256,)
        out_fc = (256, 256)
        f.write('Net structure of MCGCN:' + str(in_fc) + str(gc) + str(out_fc) + '\n')
        mcgcn = MCGCN(X_train.shape[1], in_fc, gc, out_fc, y_train.shape[1], add_dl, add_dropout).to(gpu)

        # 模型训练
        t0 = datetime.datetime.now()
        mcgcn.train()
        criterion2 = MyLoss(gamma=loss_reg)
        optimizer2 = optim.Adam(mcgcn.parameters(), lr=0.001)
        scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=50, gamma=0.5)

        # 每个epoch
        for epoch in range(epoch2):
            t1 = datetime.datetime.now()

            # 生成数据集和数据加载器
            data_train = MyDataset(X_train, y_train)
            dataloader = DataLoader(data_train, batch_size=64, shuffle=True)

            # 每个batch
            for item in dataloader:
                batch_X, batch_y = item[0], item[1]
                optimizer2.zero_grad()

                # 计算训练误差并反向传播
                output_train = mcgcn(batch_X, adj)
                loss_train = criterion2(output_train, batch_y, mcgcn)
                loss_train.backward()
                loss['mcgcn_train'].iloc[rep, epoch] += loss_train.item()

                # 模型参数调整
                optimizer2.step()
            scheduler2.step()

            # 计算测试误差
            output_test = mcgcn(X_test, adj)
            loss_test = criterion2(output_test, y_test, mcgcn)
            loss['mcgcn_test'].iloc[rep, epoch] = loss_test.item()

            # 打印
            t2 = datetime.datetime.now()
            print('Epoch: {:03d} loss_train: {:.4f} loss_test: {:.4f} time: {}'.format(epoch + 1,
                                                                                       loss['mcgcn_train'].iloc[
                                                                                           rep, epoch],
                                                                                       loss_test.item(), t2 - t1))
        t3 = datetime.datetime.now()
        print('Optimization Finished! Time:', t3 - t0)

        # 模型测试
        mcgcn.eval()
        y_pred = mcgcn(X_test, adj)
        r2['mcgcn'].iloc[rep, :], rmse['mcgcn'].iloc[rep, :] = r2_rmse(
            scaler_y.inverse_transform(y_test.cpu().detach().numpy()),
            scaler_y.inverse_transform(y_pred.cpu().detach().numpy()))

# 存储结果
np.save('adj.npy', adj_)
np.save('r2.npy', r2)
np.save('rmse.npy', rmse)
np.save('loss.npy', loss)
f.close()
