# 初始化模型
import tensorflow as tf
from d2l import tensorflow as d2l
import matplotlib.pyplot as plt

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


# 定义多层感知机的参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256

# 初始化第一层的权重和偏置
W1 = tf.Variable(tf.random.normal(
    shape=(num_inputs, num_hiddens), mean=0, stddev=0.01))
b1 = tf.Variable(tf.zeros(num_hiddens))

# 初始化第二层的权重和偏置
W2 = tf.Variable(tf.random.normal(
    shape=(num_hiddens, num_outputs), mean=0, stddev=0.01))
b2 = tf.Variable(tf.zeros(num_outputs))

# 将所有参数集合在一起
params = [W1, b1, W2, b2]

# 定义激活函数
def relu(X):
    return tf.math.maximum(X, 0)

# 定义模型
def net(X):
    """构建多层感知机的前向传播网络，使用ReLU激活函数。
    
    参数:
    X -- 输入数据，形状为(batch_size, num_inputs)
    
    返回:
    网络的输出，形状为(batch_size, num_outputs)
    """
    X = tf.reshape(X, (-1, num_inputs))
    H = relu(tf.matmul(X, W1) + b1)
    return tf.matmul(H, W2) + b2

# 定义损失函数
def loss(y_hat, y):
    return tf.losses.sparse_categorical_crossentropy(
        y, y_hat, from_logits=True)


# 设置训练的轮数和学习率
num_epochs, lr = 10, 0.9
# 创建更新器以更新模型参数
updater = tf.keras.optimizers.SGD(learning_rate=lr)

# 自定义训练函数
def train(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # 训练损失总和，训练准确度总和，样本数
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                # 前向传播并计算损失
                l = loss(net(X), y)
            # 计算梯度并更新参数
            grads = tape.gradient(l, [W1, b1, W2, b2])
            updater.apply_gradients(zip(grads, [W1, b1, W2, b2]))
            # 更新训练的总损失、准确率和样本数量
            metric.add(tf.reduce_sum(l), d2l.accuracy(net(X), y), y.shape[0])
        # 打印每轮训练的结果
        train_loss, train_acc = metric[0] / metric[2], metric[1] / metric[2]
        print(f'epoch {epoch + 1}, loss {train_loss:.4f}, accuracy {train_acc:.3f}')
        
    # 评估模型在测试集上的准确率
    test_acc = d2l.evaluate_accuracy(net, test_iter)
    print(f'test accuracy: {test_acc:.3f}')

# 设置训练的轮数
num_epochs = 10

# 使用自定义的训练函数进行训练
train(net, train_iter, test_iter, loss, num_epochs, updater)

# 自定义预测函数
def predict(net, test_iter, n=6):
    # 从测试集中获取 n 个样本
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y.numpy())
    preds = d2l.get_fashion_mnist_labels(tf.argmax(net(X), axis=1).numpy())
    # 显示真实标签和预测标签
    titles = [f'True: {true}\nPred: {pred}' for true, pred in zip(trues, preds)]
    d2l.show_images(tf.reshape(X[0:n], (n, 28, 28)), 1, n, titles=titles[0:n])

    plt.show()

# 在测试数据上进行预测
predict(net, test_iter)