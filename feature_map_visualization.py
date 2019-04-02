import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
image_size = 64  # 样本 size：28*28

PATH_TO_CKPT = 'E:\Scene_Recognition_V5\pb_model\DNN_3_layers.pb'  # 训练好的网络模型参数文件路径


def image_process():
    img = cv2.imread('E:\\Scene_Recognition_V5\\test_images\\1.jpg')
    img = cv2.resize(img, (64, 64))
    return img


images = image_process()

load_graph = tf.Graph()
with load_graph.as_default():  # 加载模型
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

with tf.Session(graph=load_graph) as sess:  # 开始一个会话
    # 加载模型 tensor
    example_test = images.reshape((1, image_size, image_size, 3))  # reshape 为 4D 张量

    x = load_graph.get_tensor_by_name('images:0')

    keep_prob = load_graph.get_tensor_by_name('Fc1/my_keep_prob:0')
    print(load_graph.get_operations())
    feature_map = load_graph.get_tensor_by_name('Conv1/h_conv1/Relu:0')

    feature_map = sess.run(feature_map, feed_dict={x: example_test, keep_prob: 1.0})
    feature_map = np.squeeze(feature_map, axis=0)
    # 全连接层需要按照其维度reshape
    # feature_map = np.reshape(feature_map, (-1, 240))
    feature_map_combination = []
    plt.figure(figsize=(4, 4))
    for i in range(0, 32):
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)
        plt.subplot(4, 8, i + 1)
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,
                            wspace=0, hspace=0)
        plt.imshow(feature_map_split)
        plt.axis('off')
    #全连接层直接将神经元imshow即可
    # plt.imshow(feature_map)
    # plt.axis('off')
    plt.show()
    plt.pause(1000)


