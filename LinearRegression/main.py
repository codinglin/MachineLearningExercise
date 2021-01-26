# 附上原链接https://colab.research.google.com/drive/131sSqmrmWXfjFZ3jWSELl8cm0Ox5ah3C

# 记录下自己的学习，慢慢对代码进行更多的注释
# date:2021/1/24
# name:lin

import sys
import pandas as pd
import numpy as np

data = pd.read_csv('data/train.csv', encoding='big5')

data = data.iloc[:, 3:]  # 取所有的行与第三行后的数据，即为所有pm2.5的数据
data[data == 'NR'] = 0  # 将其中数据为NR的填充为0
raw_data = data.to_numpy()  # 将数据转为numpy数组

# 将原始的4320 * 18的数据依照每个月分为 12 * 18 (features) * 480 (hours) 的数据
# 训练集是取每个月的前20天，每隔1小时一组数据，也就是 20 * 24 =480
# 将每个月分好的数据sample 放入 month_data中
month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24: (day + 1) * 24] = raw_data[18 * (20 * month + day): 18 * (20 * month + day + 1), :]
    month_data[month] = sample

'''
    声明 train_x for previous 9-hr data,and train_y for 10th-hr pm2.5
    每个月有480小时，前9个小时作为一个data,则每个月有471个数据
    故总数据为471*12，而每个data有9*18的features [18(features) * 9(hours)]
    对应的target则有471 *12个
'''

x = np.empty([12 * 471, 18 * 9], dtype=float)
y = np.empty([12 * 471, 1], dtype=float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:, day * 24 + hour: day * 24 + hour + 9].reshape(1, -1)  # vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9]  # value
# print(x)
# print(y)

# 标准化
mean_x = np.mean(x, axis=0)  # 18 * 9
std_x = np.std(x, axis=0)  # 18 * 9
for i in range(len(x)):  # 12 * 471
    for j in range(len(x[0])):  # 18 * 9
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
# print(x)

# 把训练数据划分为训练集和验证集
import math

x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8):, :]
y_validation = y[math.floor(len(y) * 0.8):, :]
# print(x_train_set)
# print(y_train_set)
# print(x_validation)
# print(y_validation)
# print(len(x_train_set))
# print(len(y_train_set))
# print(len(x_validation))
# print(len(y_validation))

# Training
dim = 18 * 9 + 1
w = np.zeros([dim, 1])
x = np.concatenate((np.ones([12 * 471, 1]), x), axis = 1).astype(float)
learning_rate = 100
iter_time = 1000
adagrad = np.zeros([dim, 1])
eps = 0.0000000001
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2))/471/12)#rmse
    if(t%100==0):
        print(str(t) + ":" + str(loss))
    gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y) #dim*1
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
np.save('weight.npy', w)
# print(w)

# Testing
testdata = pd.read_csv('data/test.csv', header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, 18*9], dtype = float)
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)
# print(test_x)

# Prediction
w = np.load('weight.npy')
ans_y = np.dot(test_x, w)
print(ans_y)

# Save Prediction to CSV File
import csv
with open('submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)
