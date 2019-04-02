import tensorflow as tf
import datetime
import matplotlib.pyplot as plt


graph_path = "D:\\DNN_algorithm\\my_graph"
# CNN模型文件 保存路径
cnn_model_save_path = "D:\\DNN_algorithm\\cnn_model\\DNN_algorithm_cnn_model_3_layers.ckpt"

class_num = 3


def read_and_decode_train():
    filepath = 'D:\DNN_algorithm\\tfrecords\\lab_traindata3.tfrecords'
    files = tf.gfile.Glob(filepath)

    filename_queue = tf.train.string_input_producer(files)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        }
    )

    image = tf.decode_raw(features['img_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)
    channel = 3
    image_shape = tf.stack([64, 64, channel])
    image = tf.reshape(image, image_shape)
    image = tf.cast(image, tf.float32) * (1. / 255.) - 0.5
    return image, label


image_train, label_train = read_and_decode_train()
img_batch, label_batch = tf.train.shuffle_batch([image_train, label_train],
                                                batch_size=2500,
                                                capacity=30000,
                                                min_after_dequeue=25000,
                                                num_threads=2)


def read_and_decode_test():
    filepath = 'D:\DNN_algorithm\\tfrecords\\lab_testdata2.tfrecords'
    files = tf.gfile.Glob(filepath)

    filename_queue = tf.train.string_input_producer(files)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        }
    )

    image = tf.decode_raw(features['img_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)
    channel = 3
    image_shape = tf.stack([64, 64, channel])
    image = tf.reshape(image, image_shape)
    image = tf.cast(image, tf.float32) * (1. / 255.) - 0.5
    return image, label


image_test, label_test = read_and_decode_test()
img_test_batch, label_test_batch = tf.train.shuffle_batch([image_test, label_test],
                                                          batch_size=1350,
                                                          capacity=1500,
                                                          min_after_dequeue=1300,
                                                          num_threads=2)


def weight_variable(shape, f_name):

    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=f_name)


def bias_variable(shape, f_name):

    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, f_name)


def conv2d(x, w):

    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_2x2(x):

    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def train():

    start_learning_rate = 0.001
    steps_per_decay = 10
    decay_factor = 0.95
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=start_learning_rate,
                                               global_step=global_step,
                                               decay_steps=steps_per_decay,
                                               decay_rate=decay_factor,
                                               staircase=True,
                                               # If `True` decay the learning rate at discrete intervals
                                               # staircase = False,change learning rate at every step
                                               )
    print("  scene recognition cnn train ---")
    x = tf.placeholder(tf.float32, [None, 64, 64, 3], name="images")
    y_ = tf.placeholder(tf.float32, [None, class_num], name="labels")
    #  第一层卷积池化
    # 卷积核5*5，3个channel，32个卷积核，形成32个feature map
    with tf.name_scope('Conv1'):
        w_conv1 = weight_variable([5, 5, 3, 32], 'W_conv1')
        b_conv1 = bias_variable([32], 'b_conv1')
        with tf.name_scope('h_conv1'):
            h_conv1 = tf.nn.relu(conv2d(x, w_conv1) + b_conv1)
    with tf.name_scope('Pool1'):
        h_pool1 = max_pool_2x2(h_conv1)
    #  第二层卷积池化
    with tf.name_scope('Conv2'):
        w_conv2 = weight_variable([4, 4, 32, 32], 'W_conv2')
        b_conv2 = bias_variable([32], 'b_conv2')
        with tf.name_scope('h_conv2'):
            h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    with tf.name_scope('Pool2'):
        h_pool2 = max_pool_2x2(h_conv2)
    #  第三层卷积池化
    with tf.name_scope('Conv3'):
        w_conv3 = weight_variable([3, 3, 32, 32], 'W_conv3')
        b_conv3 = bias_variable([32], 'b_conv3')
        with tf.name_scope('h_conv3'):
            h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3) + b_conv3)

    with tf.name_scope('Pool3'):
        h_pool3 = max_pool_2x2(h_conv3)
    #  第四层全连接层
    with tf.name_scope('Fc1'):
        w_fc1 = weight_variable([6 * 6 * 32, 240], 'W_fc1')
        b_fc1 = bias_variable([240], 'b_fc1')

        with tf.name_scope('Pool2_flat'):
            h_pool2_flat = tf.reshape(h_pool3, [-1, 6 * 6 * 32])

        with tf.name_scope('h_fc1'):
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

        keep_prob = tf.placeholder(tf.float32, name="my_keep_prob")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name="my_h_fc1_drop")
    #  第五层全连接层
    with tf.name_scope('Fc2'):
        w_fc2 = weight_variable([240, class_num], 'W_fc2')
        b_fc2 = bias_variable([class_num], 'b_fc2')

        with tf.name_scope('matmul'):
            y_conv = tf.add(tf.matmul(h_fc1_drop, w_fc2), b_fc2, name="add_pre")
        with tf.name_scope('softmax'):
            y_pre = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2, name="my_prediction")

    with tf.name_scope('Corss_Entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv), name="loss")
        tf.summary.scalar('corss_entropy', cross_entropy)

    with tf.name_scope('Train_step'):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step, name="train_step")

    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    print("cnn train start ---")
    with tf.Session() as session:
        # 启动Session
        session.run(init)
        train_writer = tf.summary.FileWriter(graph_path, session.graph)
        # 注意在训练开始之前，一定要调用start_queue_runners来开启各个队列的线程，否则队列的内容一直为空，训练的进程会一直挂着无法运行
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=session)
        saver = tf.train.Saver()  # 模型保存
        print("start .....")
        start = datetime.datetime.now()
        step_test = []
        accur_test = []
        loss_test = []
        step_train = []
        accur_train = []
        loss_train = []
        max_acc = 0
        for i in range(1000):
            img_batch_i, lab_batch_i = session.run([img_batch, tf.one_hot(label_batch, class_num, 1, 0)])
            print('learning rate:', session.run(learning_rate))
            train_loss, _, train_acc = session.run([cross_entropy, train_step, accuracy], feed_dict={x: img_batch_i, y_: lab_batch_i, keep_prob: 0.75})
            print("step%d loss:%f accuracy:%F" % (i, train_loss, train_acc))
            step_train.append(i)
            accur_train.append(train_acc)
            loss_train.append(train_loss)
            if (i % 10) == 0:
                if i > 1:
                    print("训练第", i, "次")
                    img_test_xs, label_test_xs = session.run([img_test_batch, tf.one_hot(label_test_batch, class_num, 1, 0)])  # 读取测试 batch
                    test_loss, test_acc = session.run([cross_entropy, accuracy], feed_dict={x: img_test_xs, y_: label_test_xs, keep_prob: 1.0})
                    print("Itsers = " + str(i) + " 测试损失值：" + str(test_loss) +"  测试准确率: " + str(test_acc))
                    step_test.append(i)
                    loss_test.append(test_loss)
                    accur_test.append(test_acc)
                    ###############################################
                    summay = session.run(merged, feed_dict={x: img_test_xs, y_: label_test_xs, keep_prob: 1.0})
                    # 每一次迭代中通过 add_summary 将测试得到的数据写入定义的 FileWriter
                    train_writer.add_summary(summay, i)
                    if max_acc < test_acc:  # 记录测试准确率最大时的模型
                        max_acc = test_acc
                        saver.save(session, save_path=cnn_model_save_path)
                    if test_acc > 0.998:
                        break
        font = {'size': 12}
        end = datetime.datetime.now()
        print("用时：" + str((end - start).seconds) + "秒")
        plt.figure()
        plt.subplot(2, 1, 1)
        # plt.title("loss")
        plt.plot(step_train, loss_train, 'r-', label="train_loss")
        plt.plot(step_test, loss_test, 'g-.', label="test_loss")
        plt.legend(loc=0, ncol=1, prop=font)
        # plt.xlabel("STEP")
        plt.grid()

        plt.subplot(2, 1, 2)
        # plt.title("accuracy")
        plt.plot(step_train, accur_train, 'r-', label="train_accuracy")
        plt.plot(step_test, accur_test, 'g-.', label="test_accuracy")
        plt.legend(loc=0, ncol=1, prop=font)
        plt.xlabel("STEP")
        plt.grid()
        plt.show()

        train_writer.close()
        coord.request_stop()
        coord.join(threads)
        session.close()
if __name__ == "__main__":

    train()
