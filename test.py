import pandas as pd
import numpy as np
import lightgbm as lgb
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score as auc
from sklearn.model_selection import StratifiedKFold

# 1. 数据读取
df_train = pd.read_csv('data/train.csv')  # 读取训练样本
df_test = pd.read_csv('data/evaluation_public.csv')  # 读取测试样本
print("在训练集中，共有{}条数据，其中每条数据有{}个特征".format(df_train.shape[0], df_train.shape[1]))
print("在测试集中，共有{}条数据，其中每条数据有{}个特征".format(df_test.shape[0], df_test.shape[1]))

# 2. 特征构造
df = pd.concat([df_train, df_test])  # 列对齐合并数据
# 2.1 分析数据特征（结合数据和网上资料）
print(df.info())  # 打印df的信息
# 2.2 特征处理
# 2.2.1 时间特征处理（数据类型转换和时间特征分割）
df['op_datetime'] = pd.to_datetime(df['op_datetime'])  # 时间格式转换
df['hour'] = df['op_datetime'].dt.hour  # 添加时间
df['dayofweek'] = df['op_datetime'].dt.dayofweek  # 添加每周的哪一天
df = df.sort_values(by=['user_name', 'op_datetime']).reset_index(drop=True)  # 根据用户名、日期排序
# 2.2.2 新特征构造（基于用户名和时间特征）
df['ts'] = df['op_datetime'].values.astype(np.int64) // 10 ** 9  # 将时间转化为int类型，并取名为ts
df['ts1'] = df.groupby('user_name')['ts'].shift(1)  # 以用户名做聚合，ts字段下移一位，命名为 ts1
df['ts2'] = df.groupby('user_name')['ts'].shift(2)  # 以用户名做聚合，ts字段下移二位，命名为 ts2
df['ts_diff1'] = df['ts1'] - df['ts']  # 同一用户第二次操作时间与第一次操作时间的间隔
df['ts_diff2'] = df['ts2'] - df['ts']  # 同一用户第三次操作时间与第一次操作时间的间隔

# 2.2.3 数据归一化（时间特征用sin函数和cos函数）
df['hour_sin'] = np.sin(df['hour'] / 24 * 2 * np.pi)  # 通过sin函数将时间映射到[-1,1]
df['hour_cos'] = np.cos(df['hour'] / 24 * 2 * np.pi)  # 将cos函数将时间映射到[-1,1]

LABEL = 'is_risk'
# 2.2.4 特征筛选（去掉无用旧特征，添加新特征）
# 选取样本集内十六个字段 其中的十二个特征字段  id、op_month、http_status_code、is_risk 没有选
cat_f = ['user_name', 'department', 'ip_transform', 'device_num_transform', 'browser_version', 'browser',
         'os_type', 'os_version', 'ip_type', 'op_city', 'log_system_transform', 'url', ]

# 2.2.5 特征编码 ：使用标签编码 将类别型特征从字符串转换为数字  便于后期构建树模型
for f in cat_f:
    le = LabelEncoder()
    # 将类别标签转化为数值
    df[f] = le.fit_transform(df[f])  # 相同的值分为一类
    df[f + '_ts_diff_mean'] = df.groupby([f])['ts_diff1'].transform('mean')  # 以 自身特征字段 做聚合，获得 ts_diff1 字段 的平均值
    df[f + '_ts_diff_std'] = df.groupby([f])['ts_diff1'].transform('std')  # 以 自身特征字段 做聚合，获得 ts_diff1 字段 的标准差
print(df)

#  依据LABEL是否为空将之前合并的数据集拆分为训练集和测试集
df_train = df[df[LABEL].notna()].reset_index(drop=True)  # 数据处理后的 训练集 47660 rows x 50 columns (重置索引)
df_test = df[df[LABEL].isna()].reset_index(drop=True)  # 数据处理后的 测试集 25710 rows x 50 columns (重置索引)

feats = [f for f in df_test if f not in [LABEL, 'id',
                                         'op_datetime', 'op_month', 'ts', 'ts1', 'ts2']]
# 打印feats
print("打印特征集合:\n{}".format(feats))  # 43个特征
# # # 打印
print(df_train[feats].shape, df_test[feats].shape)  # (47660, 43) (25710, 43)
print(df_train[feats].dtypes)  # 特征字段 全为 数值类型
print("**" * 100)

# 3 模型训练(lgb+StratifiedKFold)和模型评估(auc)
# lgb：LightGBM是个快速、分布式的、高性能的基于决策树算法的梯度提升框架。
# 基于决策树算法的，它采用最优的leaf-wise策略分裂叶子节点
# 当增长到相同的叶子节点，leaf-wise算法比level-wise算法减少更多的loss，因此导致更高的精度，而且速度快
params = {
    'learning_rate': 0.05,  # 学习速率
    'boosting_type': 'gbdt',  # 设置提升类型  梯度提升决策树
    'objective': 'binary',  # 目标函数
    'metric': 'auc',  # 评估函数
    'num_leaves': 64,  # 叶子节点数
    'verbose': -1,  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    'seed': 2222,
    'n_jobs': -1,

    'feature_fraction': 0.8,  # 建树的特征选择比例
    'bagging_fraction': 0.9,  # 建树的样本采集比例
    'bagging_freq': 4,  # k 意味着每k次迭代执行bagging
    # 'min_child_weight': 10,
}

fold_num = 5  # 分为5组
seeds = [2222]
oof = np.zeros(len(df_train))  # 含有47660个0的一维数组
importance = 0
pred_y = pd.DataFrame()
score = []
print(df_train.shape)  # (47660, 50)
print(df_train[feats].shape, df_train[LABEL].shape)  # (47660, 43) (47660,1)

for seed in seeds:
    # k折交叉验证 : 在KFold的基础上，加入了分层抽样的思想，使得测试集和训练集有相同的数据分布
    kf = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=seed)  # shuffle不会打乱样本顺序，它返回的只是index
    # kf = KFold(n_splits=fold_num, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(kf.split(df_train[feats], df_train[LABEL])):
        print('-----------', fold)
        print(train_idx.shape)  # (38128,)
        print(val_idx.shape)  # (9532,)
        train = lgb.Dataset(df_train.loc[train_idx, feats],
                            df_train.loc[train_idx, LABEL])  # 将数据保存到LightGBM二进制文件将使加载更快
        val = lgb.Dataset(df_train.loc[val_idx, feats],
                          df_train.loc[val_idx, LABEL])  # 创建验证集数据
        model = lgb.train(params, train, valid_sets=[val], num_boost_round=20000,  # 构建模型
                          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(2000)])  # 如果在100轮内验证集指标不提升我们就停止迭代
        # 预测数据集  model.predict()返回的一个预测概率
        oof[val_idx] += model.predict(df_train.loc[val_idx, feats]) / len(seeds)  # 9532 rows x 1 columns 验证集数据
        pred_y['fold_%d_seed_%d' % (fold, seed)] = model.predict(df_test[feats])   # 25710 rows x 1 columns 测试集数据
        importance += model.feature_importance(importance_type='gain') / fold_num   # 模型根据 信息增益 计算重要性
        score.append(auc(df_train.loc[val_idx, LABEL], model.predict(df_train.loc[val_idx, feats])))  # 使用验证集计算auc得分情况

feats_importance = pd.DataFrame()
feats_importance['name'] = feats
feats_importance['importance'] = importance
#  打印特征字段的重要性排名（取前30个）
print(feats_importance.sort_values('importance', ascending=False)[:30])
df_train['oof'] = oof  # 47660 rows x 1 columns  训练集的风险预测概率
print(np.mean(score), np.std(score))  # 平均分数和平均方差(评价指标)
score = np.mean(score)

# 4. 输出csv文件
df_test[LABEL] = pred_y.mean(axis=1).values  # 取预测得到的5组数据的平均值 作为 测试集的风险预测概率
df_test = df_test.sort_values('id').reset_index(drop=True)  # 按照id值给df_test重新建立索引  # 25710 rows x 2 columns
sub = pd.read_csv('data/submit_sample.csv')  # 读取需要提交的csv文件
sub['ret'] = df_test[LABEL].values  # 将训练集中is_risk的值赋给ret
sub.columns = ['id', LABEL]  # 给sub 列取名为 id 和 is_risk
# sub.to_csv(time.strftime('ans/lgb_%Y%m%d%H%M_') + '%.5f.csv' % score, index=False)  # 输出结果文件
