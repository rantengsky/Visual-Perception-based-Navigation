#encoding=utf-8
import os
import tensorflow as tf
from PIL import Image

cwd = os.getcwd()

# classes = ('bird', 'car', 'cat', 'dog')
classes = ('go_left', 'go_right', 'go_straight')  # 按照顺序遍历

filepath = './tfrecords/'
tfrecordfilename = ("traindata.tfrecords")

# 制作二进制数据

image_size = 128
writer = tf.python_io.TFRecordWriter(filepath + tfrecordfilename)
for index, name in enumerate(classes):
    print(index)
    print(name)
    # class_path = cwd + "\\test_set\\" + name + "\\"
    class_path = cwd + "\\train_set\\" + name + "\\"
    # class_path = "C:\\Users\\D201-RAN\\Desktop\\Paper\\hewei_dataset\\test\\" + name + "\\"
    list = os.listdir(class_path)
    for num in range(0, len(list)):
        img_path = os.path.join(class_path, list[num])
        print(num)
        print(img_path)
        img = Image.open(img_path)
        # img = img.convert("RGB")
        img = img.resize((image_size, image_size))
        size = img.size
        print(size)
        img_raw = img.tobytes()
        example = tf.train.Example(
            features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                # 'img_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[size[0]])),
                # 'img_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[size[1]]))
            }))
        writer.write(example.SerializeToString())
writer.close()
