#数据预处理

# 首先创建自定义的文件csv文件
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')


# 导入pandas包并调用read_csv函数。
# 该数据集有四行三列。其中每行描述了房间数量（“NumRooms”）
# 巷子类型（“Alley”）和房屋价格（“Price”）。
import pandas as pd

data = pd.read_csv(data_file)
print(data)


inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.select_dtypes(include='number').mean())
print(inputs)

# 自动将此列转换为两列“Alley_Pave”和“Alley_nan”。 
# 巷子类型为“Pave”的行会将“Alley_Pave”的值设置为1，“Alley_nan”的值设置为0。 
# 缺少巷子类型的行会将“Alley_Pave”和“Alley_nan”分别设置为0和1。
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

#现在inputs和outputs中的所有条目都是数值类型，它们可以转换为张量格式。 
import tensorflow as tf

X = tf.constant(inputs.to_numpy(dtype=float))
y = tf.constant(outputs.to_numpy(dtype=float))

print(X)
print(y)