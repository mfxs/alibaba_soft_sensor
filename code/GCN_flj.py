# 导入库
from __future__ import division
from __future__ import print_function

import os
import math
import torch
import cvxpy as cvx
import numpy as np
import pandas as pd

from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import r2_score
from torch.optim import lr_scheduler
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.nn import Linear, ReLU, Sigmoid, Sequential

# 解决报错问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 计算邻接矩阵
def linear(X, a):
    W = cvx.Variable((X.shape[1], X.shape[1]))
    item1 = cvx.sum_squares(cvx.norm(X * W - X, p=2, axis=0))
    item2 = cvx.norm1(W)
    constraints = []
    for i in range(X.shape[1]):
        constraints.append(W[i, i] == 0)
    constraints.append(W == W.T)
    objective = cvx.Minimize((item1) + (a * item2))
    prob = cvx.Problem(objective, constraints)
    prob.solve()
    return W.value


# 将邻接矩阵归一化
def Normalize(W):
    D = np.mat(np.diag(np.sum(W, axis=1))).I
    return D * W


# 数据集类
class MyDataset(Dataset):
    def __init__(self, data, label):
        self.transform = transforms.ToTensor()
        self.data = self.transform(data)[0]
        self.label = self.transform(label)[0]

    def __getitem__(self, index):
        return self.data[index, :], self.label[index, :]

    def __len__(self):
        return self.data.shape[0]


# 图卷积运算
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj.float(), support.float())
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


# 模型类
class Model(nn.Module):
    def __init__(self, n_variable, n_hid1, n_hid2, n_hid3, n_target):
        super(Model, self).__init__()
        self.fc_list1 = []
        self.fc_list2 = []
        self.fc_list3 = []
        self.n_target = n_target
        self.n_variable = n_variable
        self.n_hid1 = n_hid1
        self.n_hid2 = n_hid2
        self.n_hid3 = n_hid3
        for i in range(n_target):
            self.fc_list1.append(Sequential(
                Linear(n_variable, n_hid1),
                ReLU()))
            self.fc_list2.append(Sequential(
                Linear(n_hid2, n_hid3),
                ReLU()))
            self.fc_list3.append(Linear(n_hid3 + n_variable, 1))

        self.graph1 = GraphConvolution(n_hid1, n_hid2)
        self.graph2 = GraphConvolution(n_hid2, n_hid2)
        self.act1 = ReLU()
        self.act2 = ReLU()

    def forward(self, x, adj):
        feat_list = []
        x = torch.tensor(x, dtype=torch.float32)
        for i in range(self.n_target):
            feat = self.fc_list1[i](x)
            feat_list.append(feat)
        feat = torch.stack(feat_list, dim=1)
        feat = self.graph1(feat, adj)
        feat = self.act1(feat)
        feat = self.graph2(feat, adj)
        feat = self.act2(feat)

        res_list = []
        for i in range(self.n_target):
            res = self.fc_list2[i](feat[:, i, :])
            res = torch.cat((res, x), 1)
            res = self.fc_list3[i](res)
            res_list.append(res.squeeze())
        res = torch.stack(res_list, dim=1)
        return res


class self_attention(nn.Module):
    def __init__(self, n_variable):
        super().__init__()
        self.n_variable = n_variable
        self.projection = nn.Sequential(
            Linear(n_variable, n_variable),
            Sigmoid()
        )

    def forward(self, inputs):
        energy = self.projection(inputs)
        weights = F.softmax(energy, dim=1)
        outputs = inputs.mul(weights)
        return outputs


# 导入数据
data = pd.read_csv('data_preprocess.csv', index_col=0)
data.drop(['记录时间', '熔炼号', '钢种', '实际值-低碳锰铁', 'sb_record_time', 'sb_record_time_x', 'sb_record_time_y'], axis=1,
          inplace=True)
input_data = data.iloc[:, :35]
output_data = data.iloc[:, 35:]

# 数据标准化
input_max_ = input_data.max()
input_min_ = input_data.min()
input_data = (input_data - input_min_) / (input_max_ - input_min_ + 1e-7)

output_max_ = output_data.max()
output_min_ = output_data.min()
output_data = (output_data - output_min_) / (output_max_ - output_min_ + 1e-7)

# 数据集划分
traindata, testdata, trainlabel, testlabel = train_test_split(input_data, output_data, test_size=0.2, random_state=0)

# 计算邻接矩阵并归一化
adj_ = linear(trainlabel.values, 5)
# adj_ = (pd.DataFrame(adj_).abs()>0.1).astype(int).values
# print(np.sum(adj_,axis=1))
adj_ = Normalize(adj_)
adj = (adj_ + 2 * np.eye(adj_.shape[0])) / 3

# adj = (trainlabel.cov().abs() >= 0.006).astype(int).values
# for i in range(adj.shape[0]):
#    if adj[i,i] == 0:
#        adj[i,i] += 1
# adj = Normalize(adj+16*np.eye(adj.shape[0]))

traindata = traindata.values
trainlabel = trainlabel.values
testdata = testdata.values
testlabel = testlabel.values

print(traindata.shape, testdata.shape, trainlabel.shape, testlabel.shape, adj.shape)

# ==========model1: LGBM================
from lightgbm import LGBMRegressor

rs_list1 = []
clf1_list = []
for i in range(trainlabel.shape[1]):
    loc_trainlabel = trainlabel[:, i]
    loc_testlabel = testlabel[:, i]
    clf1 = LGBMRegressor(max_depth=3, learning_rate=0.1, n_estimators=1000, subsample=1)
    clf1.fit(traindata, trainlabel[:, i])
    clf1_list.append(clf1)
    res = clf1.predict(testdata)
    print(i, output_data.columns[i], r2_score(res, loc_testlabel))
    rs_list1.append(r2_score(res, loc_testlabel))
print(np.mean(rs_list1))
print()

adj = torch.Tensor(adj)

# 生成数据集和数据加载器
train_dataset = MyDataset(traindata, trainlabel)
test_dataset = MyDataset(testdata, testlabel)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = Model(n_variable=input_data.shape[1], n_hid1=1024, n_hid2=256, n_hid3=256, n_target=output_data.shape[1])
model.train()

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# 网络训练
for epoch in range(200):
    train_loss = 0
    test_loss = 0
    for item in train_dataloader:
        batch_data, batch_label = item[0], item[1]
        optimizer.zero_grad()
        output = model(batch_data, adj)
        loss = nn.MSELoss(size_average=False)
        loss_train = loss(output, batch_label.float())

        loss_train.backward()
        train_loss += loss_train

        output_test = model(testdata, adj)
        loss_test = loss(output_test, torch.Tensor(testlabel).float())
        test_loss += loss_test

        optimizer.step()
    scheduler.step()
    print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.4f}'.format(train_loss.item()),
          'loss_test: {:.4f}'.format(test_loss.item()))
print("Optimization Finished!")

# 网络测试
model.eval()
res = []
label = []
for item in test_dataloader:
    batch_data, batch_label = item[0], item[1]
    output = model(batch_data, adj)
    res.append(output.detach().numpy())
    label.append(batch_label.detach().numpy())

res = np.row_stack(res)
label = np.row_stack(label)

score_list = []
for i in range(res.shape[1]):
    print(output_data.columns[i], r2_score(res[:, i], label[:, i]))
    score_list.append(r2_score(res[:, i], label[:, i]))
print(np.mean(score_list))
