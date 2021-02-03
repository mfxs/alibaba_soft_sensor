# 导入库
import warnings
import pandas as pd

# 忽略警告
warnings.filterwarnings('ignore')

# 导入数据
data1 = pd.read_excel('合金历史数据/配合金历史数据6.1-7.31.xls')
data2 = pd.read_excel('合金历史数据/配合金历史数据8.1-9.30.xls')
data3 = pd.read_excel('合金历史数据/配合金历史数据10.1-12.13.xls')
data4 = pd.read_csv('vc_steel_element_data/sqlResult_2734237.csv', encoding='gbk')
data5 = pd.read_csv('vc_bof_heat_detail/sqlResult_2734245.csv', encoding='gbk')

# 剔除无用的列
data1 = pd.concat((data1, data2, data3)).reset_index(drop=True)
data1.drop(
    columns=['班次', '班别', '模型是否投用', '计算值-无烟煤', '计算值-硅锰合金', '计算值-高碳锰铁', '计算值-中碳锰铁', '计算值-低碳锰铁', '计算值-金属锰', '计算值-硅铁',
             '计算值-铌铁', '计算值-钒铁', '计算值-钼铁', '计算值-微碳铬铁', '计算值-中碳铬铁', '计算值-铜', '计算值-镍', '计算值-铝铁', '合金理论增P量', '小平台钢水理论P含量',
             '合金理论增C量', '合金理论增N量'], inplace=True)
data4.drop(
    columns=['id', 'created', 'last_updater', 'version', 'addr_sample', 'catch_mode', 'code_sample', 'count_sample',
             'ele_al', 'ele_as', 'ele_b', 'ele_be', 'ele_bi', 'ele_ca', 'ele_ce', 'ele_ceq', 'ele_co',
             'ele_cr_mo_ni_cu_v', 'ele_cu10_sn', 'ele_fe', 'ele_h', 'ele_k', 'ele_la', 'ele_li', 'ele_mg', 'ele_na',
             'ele_o', 'ele_pb', 'ele_re', 'ele_sb', 'ele_se', 'ele_sn', 'ele_ti', 'ele_w', 'ele_zn', 'ele_zr', 'other1',
             'other2', 'other3', 'other4', 'other5', 'other6', 'time_analysis', 'time_log', 'unit_id', 'ele_n'],
    inplace=True)
data5 = data5.loc[:, ['heat_id', 'hm_temp', 'hm_weight', 'sb_record_time']]

# 根据type_sample划分加入合金前后的样本
data_before = data4[data4['type_sample'] == 312]
data_after = data4[data4['type_sample'] == 314]

# 统计各列缺失和零的情况，是否需要删除某些缺失过多的元素列(N元素需要删除)
# print(data1.isnull().sum())
# print(data_before.isnull().sum())
# print(data_after.isnull().sum())
# print((data_before == 0).sum())
# print((data_after == 0).sum())
# print((data5 == 0).sum())

# 删除含有缺失值和零的样本
data1.dropna(inplace=True)
data_before.dropna(inplace=True)
data_after.dropna(inplace=True)
data_before = data_before[~((data_before == 0).any(axis=1))]
data_after = data_after[~((data_after == 0).any(axis=1))]
data5 = data5[~((data5 == 0).any(axis=1))]

# 删除完全重复样本
data1 = data1[~data1.duplicated()]
data_before = data_before[~data_before.duplicated()]
data_after = data_after[~data_after.duplicated()]
data5 = data5[~data5.duplicated()]

# 删除多次测量的样本
before_index = data_before.heat_id.duplicated(keep=False)
after_index = data_after.heat_id.duplicated(keep=False)
bof_index = data5.heat_id.duplicated(keep=False)
before_id = data_before[before_index].heat_id.unique()
after_id = data_after[after_index].heat_id.unique()
bof_id = data5[bof_index].heat_id.unique()
time_before = pd.to_datetime(data_before.sb_record_time)
time_after = pd.to_datetime(data_after.sb_record_time)
time_bof = pd.to_datetime(data5.sb_record_time)

# 将重复heat_id中时间最新的索引修改为False
for i in before_id:
    before_index[time_before[data_before.heat_id == i].argmax()] = False
for j in after_id:
    after_index[time_after[data_after.heat_id == j].argmax()] = False
for k in bof_id:
    bof_index[time_bof[data5.heat_id == k].argmax()] = False

# 剔除索引为False的数据
data_before = data_before[~before_index]
data_after = data_after[~after_index]
data5 = data5[~bof_index]

# 依据heat_id合并数据表
data = pd.merge(data1, pd.merge(data5, pd.merge(data_before, data_after, on='heat_id'), on='heat_id'), left_on='熔炼号',
                right_on='heat_id')
data.drop(columns=['heat_id', 'type_sample_x', 'type_sample_y'], inplace=True)

# 数据导出
data.to_csv('data_preprocess.csv', encoding='utf-8-sig')
