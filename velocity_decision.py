import struct

#  速度单位m/s
linear_velocity = 0.20
angular_velocity = 0.10

ratio = 1000.0  # 速度转换比例
D = 0.4450859  # 两轮间距


def velocity_process(p_left, p_right, p_straight):  # 传入图像和定位框的大小
    # 线速度角速度计算
    linear = p_straight * linear_velocity
    angular = (p_right - p_left) * angular_velocity

    return linear, angular


def serial_trans(ser_object, v, w):  # 将线速度角速度存入数组，数组作为数据通过串口发送给下位机
    # 线速度角速度转换为左右轮速度
    left_speed_data = v - 0.5 * w * D
    right_speed_data = v + 0.5 * w * D
    # 速度放大一千倍
    left_speed = left_speed_data * ratio
    right_speed = right_speed_data * ratio
    # 左右轮速度取小数点后三位
    left_speed = round(left_speed, 3)
    right_speed = round(right_speed, 3)
    # 将左右轮速度同时转换为字节
    vel = struct.pack('ff', left_speed, right_speed)
    # print("左轮下发速度：", left_speed, "右轮下发速度：", right_speed)
    # 将速度vel每一位单独取出放入list，vel为字符串
    a = vel[0]
    b = vel[1]
    c = vel[2]
    d = vel[3]
    e = vel[4]
    f = vel[5]
    g = vel[6]
    h = vel[7]
    # 创建空list，将速度分别放入list
    speed_data = []
    speed_data.append(a)
    speed_data.append(b)
    speed_data.append(c)
    speed_data.append(d)
    speed_data.append(e)
    speed_data.append(f)
    speed_data.append(g)
    speed_data.append(h)
    # 在list末尾添加换行回车符，否则串口无法接收数据
    # \r是0x0d \n是0x0a
    speed_data.append(0X0D)
    speed_data.append(0X0A)
    # 向串口发送list数据
    ser_object.write(speed_data)
    # print("command is sent !")
