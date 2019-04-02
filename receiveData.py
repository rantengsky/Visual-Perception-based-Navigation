import threading
import struct
import matplotlib.pyplot as plt
import mysql.connector

# 初始化数据库并返回db、cursor和sql对象
db = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="test_db"
)
cursor = db.cursor()
sql = "INSERT INTO odom_data (position_x, position_y, theta_data, vel_linear, vel_angule) VALUES (%s, %s, %s, %s, %s)"


db1 = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="test_db"
)
cursor1 = db1.cursor()
sql1 = "INSERT INTO time_record (classify_time, frame_time) VALUES (%s, %s)"


def data_process(ser_obj):
    data_count = 0  # 参数用于统计正确数据的数量
    insert_count = 0  # 参数用于统计数据插入数据库的次数
    # plt.ion()  # 开启interactive mode 成功的关键函数
    # t = [0]
    # m = [0]
    my_list = []  # 初始化数组用于存储串口数据
    while True:
        data = ser_obj.readline()
        # print(data)
        #  通过数据长度判断接收到的数据是否正确，若数据错误则跳过程序重新接收
        length = len(data)
        # print("里程计字符串长度：", length)
        if length == 21:
            data_count = data_count+1
            # print("当前正确的数据量有 ", data_count, " 条！")
            #  初始化五个空数组用于存储十六进制数据
            position_x = []
            position_y = []
            theta_data = []
            vel_linear = []
            vel_angule = []
            for i in range(4):
                #  用循环的方式将串口接收的数据分别存储到各自对应的五个数组中
                position_x.insert(i, data[i])
                position_y.insert(i, data[i + 4])
                theta_data.insert(i, data[i + 8])
                vel_linear.insert(i, data[i + 12])
                vel_angule.insert(i, data[i + 16])
            #  将数组中的数据转换为字节形式用于后续的变量转换
            position_x_bytes = bytes(position_x[:4])
            position_y_bytes = bytes(position_y[:4])
            theta_data_bytes = bytes(theta_data[:4])
            vel_linear_bytes = bytes(vel_linear[:4])
            vel_angule_bytes = bytes(vel_angule[:4])
            #  用struct.unpack（）方法将字节转换为float数
            position_x = struct.unpack('f', position_x_bytes)
            position_y = struct.unpack('f', position_y_bytes)
            theta_data = struct.unpack('f', theta_data_bytes)
            vel_linear = struct.unpack('f', vel_linear_bytes)
            vel_angule = struct.unpack('f', vel_angule_bytes)

            print("position_x: ", position_x[0])
            print("position_y: ", position_y[0])
            print("theta_data: ", theta_data[0])
            print("vel_linear: ", vel_linear[0])
            print("vel_angular: ", vel_angule[0])

            # 该部分代码用于数据的实时动态显示
            # t.append(position_x[0])  # 模拟数据增量流入，保存历史数据
            # m.append(position_y[0])  # 模拟数据增量流入，保存历史数据
            # plt.plot(t, m, c='b')
            # plt.draw()  # 注意此函数需要调用
            # plt.pause(0.0001)  # 添加plt.pause()才可以正常显示
            # 初始化一个数组用于存储数据，达到100条后就保存至数据库
            val_temp = [position_x[0], position_y[0], theta_data[0], vel_linear[0], vel_angule[0]]
            my_list.append(val_temp)  # 向数组中添加当前接收到的数据
            array_length = len(my_list)  # 提取当前数组存储的数据量
            if array_length >= 50:
                try:
                    insert_count = insert_count + 1
                    t_save = threading.Thread(target=save_data, name='save_data',
                                              args=(my_list, insert_count,))
                    t_save.start()
                    my_list = []  # 清空当前list
                    print("insert 50 odom_data")
                except:
                    print("Error: 无法启动存储数据子线程")


def save_data(my_list, insert_count):
    cursor.executemany(sql, my_list)
    db.commit()  # 数据表内容有更新，必须使用到该语句
    # print(cursor.rowcount, "记录插入成功。")
    # print("第 ", insert_count, " 次插入数据库！")


def save_time(my_list, insert_count):
    cursor1.executemany(sql1, my_list)
    db1.commit()  # 数据表内容有更新，必须使用到该语句
    # print(cursor.rowcount, "记录插入成功。")
    # print("第 ", insert_count, " 次插入数据库！")


def data_acquire(ser_obj):
    # 新建一个线程，不间断读取串口数据
    try:
        t = threading.Thread(target=data_process, name='data_process',
                             args=(ser_obj,))
        t.start()
    except:
        print("Error: 无法启动子线程")

