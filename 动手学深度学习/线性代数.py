# 标量和变量，标量及为常数，一般用小写字母表示
# 标量实例化
import tensorflow as tf
x = tf.constant(3.0)
y = tf.constant(2.0)

print(x + y)

# 一维张量表示向量
x = tf.range(4)

# 向量的长度也称向量的维度
print(x.shape)

# 二维张量表示矩阵
A = tf.reshape(tf.range(20), (5, 4))
print(A)

# 矩阵转置
tf.transpose(A)

#对称矩阵 转置和本身相等
B = tf.constant([[1, 2, 3], [2, 4, 5], [3, 5, 6]])

print(tf.transpose(B) == B)

# 降维操作：沿着不同的轴，将元素相加或者求平均值等
C = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# 按行求和
tf.reduce_sum(C, axis=0)
# 按列求和
tf.reduce_sum(C, axis=1)
# 按行求平均值
tf.reduce_mean(C, axis=0)
# 按列求平均值
tf.reduce_mean(C, axis=1)

# 非降维求和
C = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(tf.cumsum(A, axis=0))


#点积（Dot Product）：对应元素相乘求和
x = tf.constant([1, 2, 3])
y = tf.constant([4, 5, 6])
tf.tensordot(x, y, axes=1)

# 矩阵乘法：对应元素相乘，矩阵乘法的两个矩阵的行数必须相等，列数必须相等
A = tf.constant([[1, 2], [3, 4]])
B = tf.constant([[5, 6], [7, 8]])
tf.matmul(A, B)

# 张量的维度变换：reshape、transpose、expand_dims
A = tf.constant([[1, 2, 3], [4, 5, 6]])
# 改变矩阵的形状
tf.reshape(A, (3, 2))
# 矩阵转置
tf.transpose(A)
# 增加维度
tf.expand_dims(A, axis=1)