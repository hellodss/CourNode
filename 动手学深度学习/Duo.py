import tensorflow as tf
from d2l import tensorflow as d2l
import matplotlib.pyplot as plt
import os

# 加载数据集并进行标准化
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 定义多层感知机的参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256

# 使用装饰器注册自定义的 MLP 类
@tf.keras.utils.register_keras_serializable()
class MLP(tf.keras.Model):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(num_hiddens, activation='relu',
                                            kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.output_layer = tf.keras.layers.Dense(num_outputs)
    
    def call(self, X, training=False):
        X = self.flatten(X)
        H = self.hidden(X)
        if training:
            H = self.dropout(H, training=training)
        return self.output_layer(H)

    def get_config(self):
        config = super(MLP, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



# 自定义预测函数
def predict(loaded_model, test_iter, n=6):
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y.numpy())
    preds = d2l.get_fashion_mnist_labels(tf.argmax(loaded_model(X), axis=1).numpy())
    titles = [f'True: {true}\nPred: {pred}' for true, pred in zip(trues, preds)]
    d2l.show_images(tf.reshape(X[0:n], (n, 28, 28)), 1, n, titles=titles[0:n])
    plt.show()

if __name__ == '__main__':
    # 创建模型实例
    model = MLP()

    # 编译模型
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    # 设置训练的轮数
    num_epochs = 20

    # 使用 fit 方法进行训练
    model.fit(train_iter, epochs=num_epochs, validation_data=test_iter)

    # 创建保存模型的目录
    os.makedirs('saved_model', exist_ok=True)

    # 保存整个模型
    model.save('saved_model/my_mlp.keras')

    # 加载保存的模型
    loaded_model = tf.keras.models.load_model('saved_model/my_mlp.keras')

    # 在测试数据上进行预测
    predict(loaded_model, test_iter)
