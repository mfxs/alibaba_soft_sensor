# 导入库
import numpy as np

# 导入R2
pls_r2 = np.load('PLS/r2.npy').item()
svr_r2 = np.load('SVR/r2.npy').item()
fc_r2 = np.load('FC/r2.npy').item()
elm_r2 = np.load('ELM/r2.npy').item()
gcn_r2 = np.load('GCN/r2.npy').item()
mcgcn_r2 = np.load('MCGCN/r2.npy').item()

# 导入RMSE
pls_rmse = np.load('PLS/rmse.npy').item()
svr_rmse = np.load('SVR/rmse.npy').item()
fc_rmse = np.load('FC/rmse.npy').item()
elm_rmse = np.load('ELM/rmse.npy').item()
gcn_rmse = np.load('GCN/rmse.npy').item()
mcgcn_rmse = np.load('MCGCN/rmse.npy').item()

# 导入Loss
gcn_loss = np.load('GCN/loss.npy').item()
mcgcn_loss = np.load('MCGCN/loss.npy').item()

# 导入邻接矩阵
adj = np.load('MCGCN/adj.npy').item()

# 整合各模型结果
exp = 3
r2 = {'pls': pls_r2['pls'].iloc[:exp, :], 'svr': svr_r2['svr'].iloc[:exp, :], 'fc': fc_r2['fc'].iloc[:exp, :],
      'elm': elm_r2['elm'].iloc[:exp, :], 'gcn': gcn_r2['gcn'].iloc[:exp, :], 'mcgcn': mcgcn_r2['mcgcn'].iloc[:exp, :]}
rmse = {'pls': pls_rmse['pls'].iloc[:exp, :], 'svr': svr_rmse['svr'].iloc[:exp, :], 'fc': fc_rmse['fc'].iloc[:exp, :],
        'elm': elm_rmse['elm'].iloc[:exp, :], 'gcn': gcn_rmse['gcn'].iloc[:exp, :],
        'mcgcn': mcgcn_rmse['mcgcn'].iloc[:exp, :]}
loss = {'gcn_train': gcn_loss['gcn_train'].iloc[:exp, :], 'gcn_test': gcn_loss['gcn_test'].iloc[:exp, :],
        'mcgcn_train': mcgcn_loss['mcgcn_train'].iloc[:exp, :], 'mcgcn_test': mcgcn_loss['mcgcn_test'].iloc[:exp, :]}
adj['mcgcn'].pop()
adj['mcgcn'].pop()

# 存储
np.save('ALL/r2.npy', r2)
np.save('ALL/rmse.npy', rmse)
np.save('ALL/loss.npy', loss)
np.save('ALL/adj.npy', adj)
