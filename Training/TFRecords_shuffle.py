import os
import random
import tensorflow as tf
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
cwd = os.getcwd()

# classes = ('bird', 'car', 'cat', 'dog')
classes = ('go_left', 'go_right', 'go_straight')  # 按照顺序遍历

filepath = './tfrecords/'
tfrecordfilename = ("lab_traindata.tfrecords")

#制作二进制数据

image_size = 64
writer = tf.python_io.TFRecordWriter(filepath + tfrecordfilename)
num_0 = 0
num_1 = 0
num_2 = 0
while True:
    name = random.sample(classes, 1)
    if name[0] == 'go_left':
        index = 0
        class_path = cwd + "\\lab_train_set3\\" + classes[0] + "\\"
        print(class_path)
        list = os.listdir(class_path)
        if num_0 < len(list):
            print(num_0)
            img_path = os.path.join(class_path, list[num_0])
            img = Image.open(img_path)
            img = img.resize((image_size, image_size))
            size = img.size
            img_raw = img.tobytes()
            num_0 = num_0 + 1
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
            writer.write(example.SerializeToString())
    elif name[0] == 'go_right':
        index = 1
        class_path = cwd + "\\lab_train_set3\\" + classes[1] + "\\"
        print(class_path)
        list = os.listdir(class_path)
        if num_1 < len(list):
            print(num_1)
            img_path = os.path.join(class_path, list[num_1])
            img = Image.open(img_path)
            img = img.resize((image_size, image_size))
            size = img.size
            img_raw = img.tobytes()
            num_1 = num_1 + 1
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
            writer.write(example.SerializeToString())
    else:
        index = 2
        class_path = cwd + "\\lab_train_set3\\" + classes[2] + "\\"
        print(class_path)
        list = os.listdir(class_path)
        if num_2 < len(list):
            print(num_2)
            img_path = os.path.join(class_path, list[num_2])
            img = Image.open(img_path)
            img = img.resize((image_size, image_size))
            size = img.size
            img_raw = img.tobytes()
            num_2 = num_2 + 1
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
            writer.write(example.SerializeToString())
