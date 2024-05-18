# coding=utf-8
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np


# 按方差选取特征，选取前n方差的特征，并归一化
def preprocess_data(data, n):
    '''
    data: n_sample, m_feature
    '''
    # 计算特征方差
    selector = VarianceThreshold()
    selector.fit(data)
    vars = selector.variances_
    # 对特征方差进行从大到小排序
    vars = sorted(vars, reverse=True)
    # 得到第n个方差的值
    thresh = vars[n]
    # 去除方差低于thresh值的特征
    selector = VarianceThreshold(threshold=thresh)
    value = selector.fit_transform(data)
    features = selector.get_support(indices=True)  # 获得选取特征的列号
    # data = pd.DataFrame(value, index=None, columns=None)
    data = pd.DataFrame(value, index=data.index, columns=data.columns.values[features])
    # 进行最大最小值归一化
    data = (data - data.min()) / (data.max() - data.min())
    return data


if __name__ == '__main__':
    file_path = ''
    csv_path = ''
    data = pd.read_csv(file_path, index_col=0)

    print(data.shape)
    select = 0.20
    data_pre = preprocess_data(data, int(data.shape[1] * select))
    data_pre.to_csv(csv_path.format(select), header=True, index=True)


