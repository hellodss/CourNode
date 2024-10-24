import time
# 输出训练进度
# 输出训练进度
def print_progress(i, start, end, step):
    # 计算进度百分比
    progress_percentage = (i - start) / ((end - 1) - start) * 100
    progress_bar = "▋" * int(progress_percentage // 2)  # 每 2% 显示一个进度块
    print("\rTraining progress: {:.0f}%: {}".format(progress_percentage, progress_bar), end="")
    time.sleep(0.05)

# 导入pandas库
import pandas as pd
# 使用pandas读取csv数据
dataset = pd.read_csv(r'机器学习\item8\item8-sx-adult-y.csv')
# 为数据集指定列名称，年龄、职业、周工作时长、性别、学历、单位性质、月薪
dataset.columns = ['年龄', '职业', '周工作时长', '性别', '学历', '单位性质','月薪']
# 显示数据集
print(dataset.head(32561))
#分别将职业、性别、学历、单位性质列转换为数值型数据
labels_zhiYe = dataset['职业'].unique().tolist()
labels_xingBie = dataset['性别'].unique().tolist()
labels_xueLi = dataset['学历'].unique().tolist()
labels_danWei = dataset['单位性质'].unique().tolist()

dataset['职业'] = dataset['职业'].apply(lambda x: labels_zhiYe.index(x))
dataset['性别'] = dataset['性别'].apply(lambda x: labels_xingBie.index(x))
dataset['学历'] = dataset['学历'].apply(lambda x: labels_xueLi.index(x))
dataset['单位性质'] = dataset['单位性质'].apply(lambda x: labels_danWei.index(x))

# 导入preprocessing模块，将标签列月薪转化为数值型标签
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset['月薪'] = le.fit_transform(dataset['月薪'])

print(dataset.head(32560))


x = dataset.iloc[range(0,32560),range(0,6)].values
#[0] 用于从重塑后的数组中提取第一行（在这种情况下就是唯一的一行），
# 得到一个一维数组，形状为 (32561,)，其中包含32561个标签值。
y = dataset.iloc[range(0,32560),range(6,7)].values.reshape(1,32560)[0]

# 导入train_test_split模块，用于分割数据集
from sklearn.model_selection import train_test_split
# 随机采样20%的数据作为测试集，其余作为训练集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import sys
import time
# 寻找模型中n_estimators最优值
score = []
for i in range(0,201, 10):
    model = RandomForestClassifier(n_estimators=i+1, random_state=0)
    model = model.fit(x_train, y_train)
    pred = model.predict(x_test)
    ac = accuracy_score(y_test, pred)
    score.append(ac)
    
    #输出进度
    print_progress(i, 0, 201, 10)

print("\nTraining completed.")
print(f'最大准确率：{max(score)}，对应的n_estimators：{score.index(max(score))+1}')

plt.plot(range(0,201,10), score)
plt.xlabel('n_estimators')
plt.ylabel('accuracy')
plt.show()

# 使用最优的n_estimators值训练模型
model = RandomForestClassifier(n_estimators=11, random_state=0)
model = model.fit(x_train, y_train)

# 导入classification_report模块，用于评估模型性能
from sklearn.metrics import classification_report
# 预测测试集
y_pred = model.predict(x_test)
# 评估模型性能
print(classification_report(y_test, y_pred))

x_new = [[40, 9,40, 1, 2, 2]]

# 预测新数据
y_new = model.predict(x_new)
if (y_new[0] == 1):
    print('该员工月薪应超过5万元。')
else:
    print('该员工月薪不应超过5万元。')