import cv2
import tensorflow as tf
# import serial
# from velocity_decision import velocity_process, serial_trans
from camera import WebcamVideoStream, configs
# from receiveData import data_acquire
import datetime

PATH_TO_CKPT = 'D:\Scene_Recognition_V6\pb_model\CNN-8.pb'  # 训练好的网络模型参数文件路径


def main():
    my_list = []
    insert_count = 0
    # function : 加载训练好模型，进行预测测试
    load_start = datetime.datetime.now()
    load_graph = tf.Graph()
    with load_graph.as_default():  # 加载模型
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    load_end = datetime.datetime.now()
    print("加载模型用时：" + str((load_end - load_start).microseconds / 1000) + "毫秒")
    while True:
        # tf.reset_default_graph()
        start = datetime.datetime.now()

        frame_original = video_capture.read()
        end1 = datetime.datetime.now()
        # print("openCV读图用时" + str((end1 - start).microseconds / 1000) + "毫秒")
        frame = cv2.resize(frame_original, (64, 64))
        image = frame*(1./255) - 0.5
        with load_graph.as_default():
            with tf.Session(graph=load_graph) as sess:

                example_test = image.reshape((1, 64, 64, 3))  # reshape 为 4D 张量

                x = load_graph.get_tensor_by_name('images:0')
                keep_prob1 = load_graph.get_tensor_by_name('Fc1/my_keep_prob:0')
                keep_prob2 = load_graph.get_tensor_by_name('Fc2/my_keep_prob:0')
                prediction = load_graph.get_tensor_by_name('Fc3/softmax/my_prediction:0')

                end4 = datetime.datetime.now()
                # pre = sess.run(prediction, feed_dict={x: example_test, keep_prob: 1.0})
                pre = sess.run(prediction, feed_dict={x: example_test, keep_prob1: 1.0, keep_prob2: 1.0})
                end5 = datetime.datetime.now()
                print("sess.run(prediction): " + str((end5 - end4).microseconds / 1000) + "毫秒")
                # prob_left = pre[0, 0]
                # prob_right = pre[0, 1]
                # prob_straight = pre[0, 2]
                # print("prob_left：" + str(prob_left) + " prob_right：" + str(prob_right) + " prob_straight：" + str(prob_straight))
                # frame = cv2.resize(frame_original, (520, 520))
                #
                # cv2.putText(frame, 'go_left: '+str(round(prob_left, 5)), (10, 40), 0, 1, (0, 0, 255), 2)  # 显示预测的标签号在图片上
                # cv2.putText(frame, 'go_right: ' + str(round(prob_right, 5)), (10, 80), 0, 1, (255, 0, 0), 2)
                # cv2.putText(frame, 'go_straight: ' + str(round(prob_straight, 5)), (10, 120), 0, 1, (0, 255, 0), 2)
                #
                # line, angu = velocity_process(prob_left, prob_right, prob_straight)
                # cv2.putText(frame, 'cmd V: ' + str(round(line, 3)) + ' m/s', (10, 470), 0, 1, (0, 0, 0), 2)
                # cv2.putText(frame, 'cmd W: ' + str(round(angu, 3)) + ' rad/s', (10, 510), 0, 1, (0, 0, 0), 2)
                #
                # cv2.namedWindow("image", 1)  # 定义显示 窗口
                cv2.imshow("image", frame)
                cv2.waitKey(1)

                # serial_trans(ser, line, angu)
                end = datetime.datetime.now()
                classify_time = str((end - end1).microseconds / 1000)
                print("模型分类用时" + classify_time + "毫秒")
                frame_time = str((end - start).microseconds / 1000)
                print("processing time: " + frame_time + "毫秒\r\n")
                # classify_time = float(classify_time)
                # frame_time = float(frame_time)
                # val_temp = [classify_time, frame_time]
                # my_list.append(val_temp)  # 向数组中添加当前接收到的数据
                # array_length = len(my_list)  # 提取当前数组存储的数据量
                # if array_length >= 500:
                #     insert_count = insert_count + 1
                #     save_time(my_list, insert_count)
                #     my_list = []
                #     print("分类时间已保存")
            tf.get_default_graph().finalize()


if __name__ == '__main__':

    args = configs()
    video_capture = WebcamVideoStream(src=args.video_source).start()

    # ser = serial.Serial("COM6", 115200, timeout=5)
    # print("串口初始化完成！")

    # data_acquire(ser)
    main()
