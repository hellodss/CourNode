import tensorflow as tf
from d2l import tensorflow as d2l
import matplotlib.pyplot as plt
from Duo import MLP
import os
from PIL import Image
import numpy as np

# 加载保存的模型
loaded_model = tf.keras.models.load_model('saved_model/my_mlp.keras', compile=False)
# 确保只加载模型而不重新训练
loaded_model.summary()  # 打印模型架构确认是否正确加载

# 自定义预测函数，使用自己的图片
def load_custom_images(image_paths, img_size=(28, 28)):
    images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('L')  # 转为灰度图
        img = img.resize(img_size)              # 调整为 28x28
        img = np.array(img) / 255.0             # 归一化
        img = np.expand_dims(img, axis=-1)      # 增加通道维度
        images.append(img)
    return tf.convert_to_tensor(images)

def predict_custom_images(loaded_model, image_paths):
    # 加载并预处理自定义图像
    images = load_custom_images(image_paths)
    preds = tf.argmax(loaded_model(images), axis=1).numpy()
    preds_labels = d2l.get_fashion_mnist_labels(preds)

    # 显示结果
    d2l.show_images(tf.reshape(images, (len(images), 28, 28)), 1, len(images),
                    titles=[f'Pred: {pred}' for pred in preds_labels])
    plt.show()



# 使用自己的图片进行预测
image_folder = 'photos'  # 替换为你的图片文件夹路径
image_files = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]
predict_custom_images(loaded_model, image_files[:6])  # 预测前 6 张图片
