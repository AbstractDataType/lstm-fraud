import random
import time
import tensorflow as tf
tf.random.set_seed(42857)
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
import pandas as pd
import numpy as np


def cus_mse(Y_true, Y_pred):
    return metrics.mean_squared_error(Y_true[:, -5:-1, :], Y_pred[:, -5:-1, :])


# def cus_mse_2(Y_true, Y_pred):
#     return metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])
#
#
# def focal_loss(y_true, y_pred):
#     gamma = 10
#     alpha = 0.25
#     pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#     pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#     return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
#         (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
#

if __name__ == '__main__':
    model_1 = Sequential()
    model_1.add(layers.SimpleRNN(20, return_sequences=True))
    model_1.add(layers.SimpleRNN(20, return_sequences=True))
    model_1.add(layers.Dense(1, activation='sigmoid'))
    model_1.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=0.001), metrics=[cus_mse])

    store: pd.HDFStore = pd.HDFStore("./data.h5", mode='r')
    raw_all_data: pd.DataFrame = store["all_data"]
    store.close()
    features: int = raw_all_data.shape[1] - 4  # 减去 行业代码 fraud，code，year

    # spilt
    raw_train_data: pd.DataFrame = pd.DataFrame()
    raw_check_data: pd.DataFrame = pd.DataFrame()
    raw_test_data: pd.DataFrame = pd.DataFrame()

    prob_train: float = 0.8
    prob_check: float = 0.1

    r: random.Random = random.Random()
    r.seed(2853)  # 42 #2853

    for _, group in raw_all_data.groupby(["行业代码"]):
        com_list = group["code"].unique()
        split_1 = int(prob_train * len(com_list))
        split_2 = int(split_1 + prob_check * len(com_list))
        random.shuffle(com_list, random=r.random)
        for i in com_list[0:split_1]:
            raw_all_data.loc[
                raw_all_data[raw_all_data.code == i].index.tolist()
                , 'flag'] = 1
        for i in com_list[split_1:split_2]:
            raw_all_data.loc[
                raw_all_data[raw_all_data.code == i].index.tolist()
                , 'flag'] = 2
        for i in com_list[split_2:]:
            raw_all_data.loc[
                raw_all_data[raw_all_data.code == i].index.tolist()
                , 'flag'] = 3

    for name, group in raw_all_data.groupby("flag"):
        if name == 1:
            raw_train_data = raw_train_data.append(group)
        elif name == 2:
            raw_check_data = raw_check_data.append(group)
        elif name == 3:
            raw_test_data = raw_test_data.append(group)

    raw_all_data.drop(columns=["行业代码"], inplace=True)
    raw_train_data.drop(columns=["行业代码"], inplace=True)
    raw_test_data.drop(columns=["行业代码"], inplace=True)
    raw_check_data.drop(columns=["行业代码"], inplace=True)

    # sampling
    train_input = list()
    train_y = list()
    test_input = list()
    test_y = list()

    for name, group in raw_train_data.groupby("code"):
        if group["fraud"].sum() > 0:
            for i in range(0, 1):
                train_input.append(group.drop(columns=["year", "code", "fraud", "flag"]).values.tolist())
                train_y.append(group[["fraud"]].values.tolist())
        else:
            train_input.append(group.drop(columns=["year", "code", "fraud", "flag"]).values.tolist())
            train_y.append(group[["fraud"]].values.tolist())
    for name, group in raw_check_data.groupby("code"):
        if group["fraud"].sum() > 0:
            for i in range(0, 1):
                train_input.append(group.drop(columns=["year", "code", "fraud", "flag"]).values.tolist())
                train_y.append(group[["fraud"]].values.tolist())
        else:
            train_input.append(group.drop(columns=["year", "code", "fraud", "flag"]).values.tolist())
            train_y.append(group[["fraud"]].values.tolist())
    for name, group in raw_test_data.groupby("code"):
        test_input.append(group.drop(columns=["year", "code", "fraud", "flag"]).values.tolist())
        test_y.append(group[["fraud"]].values.tolist())

    train_input = np.array(train_input)
    train_y = np.array(train_y, dtype=np.uint8)
    test_input = np.array(test_input)
    test_y = np.array(test_y, dtype=np.uint8)
    history = model_1.fit(train_input, train_y, epochs=100)
    # model_1.save(f"./model_{time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime())}")
    y_prob = model_1.predict(test_input)

    # 计算f1
    f: pd.DataFrame = pd.DataFrame([[0, 0], [0, 0]], dtype=np.uint8)
    f.index = ["实际为0", "实际为1"]
    f.columns = ["预测为0", "预测为1"]
    thre: float = 0.15
    y_true = test_y
    for i in range(0, y_true.shape[0]):
        for j in range(5, y_true.shape[1]):
            if (y_true[i, j, 0] == 0) and (y_prob[i, j, 0] < thre):
                f.loc["实际为0", "预测为0"] = f.loc["实际为0", "预测为0"] + 1
            if (y_true[i, j, 0] == 0) and (y_prob[i, j, 0] >= thre):
                f.loc["实际为0", "预测为1"] = f.loc["实际为0", "预测为1"] + 1
            if (y_true[i, j, 0] == 1) and (y_prob[i, j, 0] < thre):
                f.loc["实际为1", "预测为0"] = f.loc["实际为1", "预测为0"] + 1
            if (y_true[i, j, 0] == 1) and (y_prob[i, j, 0] >= thre):
                f.loc["实际为1", "预测为1"] = f.loc["实际为1", "预测为1"] + 1

    print(f)
