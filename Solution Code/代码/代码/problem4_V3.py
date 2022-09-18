import numpy as np
import random
from scipy.optimize import root
from problem1 import get_angle,f_p2
from matplotlib import pyplot as plt


# 为了便于运算，中间的角度均采用弧度制
def get_angle_xy(x):  # 对point[0,1,2]，返回以1为顶点的角度
    # 得到直角坐标

    vec_10 = x[0, :] - x[1, :]
    vec_12 = x[2, :] - x[1, :]

    cos_angle = np.dot(vec_10, vec_12) / ((np.sqrt(np.dot(vec_10, vec_10)) * np.sqrt(np.dot(vec_12, vec_12))) + 1e-6)
    angle = np.arccos(cos_angle)

    return angle

#ideal = np.zeros((16,2)) # 为了方便，第0行存做空行
R = 50
r = 0.16 * R
base_vec = np.array([1,0])

# 得到理想位置
def get_ideal():
    dx = R * np.sqrt(3) / 2
    ideal = np.zeros((16, 2))  # 为了方便，第0行存做空行
    dy = R / 2
    ideal[13,:] = np.array([0,0])
    ideal[12,:] = np.array([0,R])
    ideal[11, :] = np.array([0, 2*R])
    ideal[14, :] = np.array([0, -R])
    ideal[15, :] = np.array([0, -2*R])
    ideal[8, :] = np.array([dx, dy])
    ideal[7, :] = np.array([ideal[12,0] + dx,ideal[12,1] + dy])
    ideal[9, :] = np.array([dx, -dy])
    ideal[10, :] = np.array([ideal[14, 0] + dx, ideal[14, 1] - dy])
    ideal[5,:] = np.array([2 * dx,0])
    ideal[4, :] = np.array([ideal[8, 0] + dx, ideal[8, 1] + dy])
    ideal[6, :] = np.array([ideal[9, 0] + dx, ideal[9, 1] - dy])
    ideal[2, :] = np.array([ideal[5, 0] + dx, ideal[5, 1] + dy])
    ideal[3, :] = np.array([ideal[5, 0] + dx, ideal[5, 1] - dy])
    ideal[1, :] = np.array([ideal[5, 0] + 2 * dx, 0])

    return ideal


# 得到临解矩阵
graph = np.zeros((16,16))
graph[1,2] = 1
graph[1,3] = 1
graph[2,3] = 1
graph[2,4]=1
graph[2,5]=1
graph[3,5]=1
graph[3,6]=1
graph[4,5]=1
graph[5,6]=1
graph[4,7]=1
graph[4,8]=1
graph[5,8]=1
graph[5,9]=1
graph[6,9]=1
graph[6,10]=1
graph[7,8]=1
graph[7,11]=1
graph[7,12]=1
graph[8,12]=1
graph[8,13]=1
graph[8,9]=1
graph[9,13]=1
graph[9,14]=1
graph[9,10]=1
graph[10,14]=1
graph[10,15]=1
graph[11,12]=1
graph[12,13]=1
graph[13,14]=1
graph[14,15]=1


ideal = get_ideal()
print('ideal',ideal)

# 添加噪声得到实际位置
actual = np.zeros((16,2))
for i in range(1,16):
    # 得到噪声
    # noise_rho = np.sqrt(random.random()) * r # 真实的
    noise_rho = random.random() * r
    noise_theta =  random.random() * 2 * np.pi
    actual[i,0] = ideal[i,0] + noise_rho * np.cos(noise_theta)
    actual[i, 1] = ideal[i,1] + noise_rho * np.sin(noise_theta)

print(actual)

# 设置预测位置
position_hat = actual


for i in range(1,16):
    # 得到噪声
    # noise_rho = np.sqrt(random.random()) * r # 真实的
    noise_rho = random.random() * r
    noise_theta =  random.random() * 2 * np.pi
    actual[i,0] = ideal[i,0] + noise_rho * np.cos(noise_theta)
    actual[i, 1] = ideal[i,1] + noise_rho * np.sin(noise_theta)

print(actual)

iter_max = 50
lr = 0.8
d = 50
t = 1
'''**********************************************************'''
# step1:确定以8为圆心的圆
index_launch = np.zeros((2,3))
index_launch[0,:] = np.array([4,9,12])
index_launch[1,:] = np.array([7,5,13])

index_measure = np.zeros((2,3))
index_measure[0,:] = np.array([7,5,13])
index_measure[1,:] = np.array([4,9,12])

launch = np.zeros(3)

for iter in range(iter_max):
    # 选择发送信号无人机
    launch = index_launch[iter % 2, :].tolist()

    # 带调整的无人机编号
    measured = index_measure[iter % 2, :].tolist()

    # 圆心
    center = 8

    # 对每个待调整无人机调整
    for i in measured:
        i = int(i)

        # 分三次测量
        alpha = np.zeros(3)
        # 角度center-i-launch[0]
        alpha[0] = get_angle_xy(np.array([actual[center, :], actual[i, :], actual[int(launch[0]), :]]))
        # 角度center-i-launch[1]
        alpha[1] = get_angle_xy(np.array([actual[center, :], actual[i, :], actual[int(launch[1]), :]]))
        # 角度center-i-launch[2]
        alpha[2] = get_angle_xy(np.array([actual[center, :], actual[i, :], actual[int(launch[2]), :]]))

        position_measure_xy_list = np.zeros((3, 2))

        for k in range(3):
            # 估算i的位置
            angle = np.zeros(4)  # angle1,angle2,beta1,beta2
            # 每次取launch_a,launch_b
            launch_a = k % 3
            launch_b = (k + 1) % 3

            # alpha
            angle[0] = alpha[launch_a]
            angle[1] = alpha[launch_b]

            print('发射点1', launch[launch_a])
            print('发射点2', launch[launch_b])
            print('被测点', i)
            print('alpha1', np.rad2deg(angle[0]))
            print('alpha2', np.rad2deg(angle[1]))
            # beta(理论值)
            # angle[2] = get_angle_xy(np.array([ideal[center,:] + base_vec, ideal[center,:], ideal[int(launch[launch_a]),:]]))
            # angle[3] = get_angle_xy(np.array([ideal[center,:] + base_vec, ideal[center,:], ideal[int(launch[launch_b]),:]]))
            relative_xy = ideal[int(launch[launch_a]), :] - ideal[center, :]

            # 得到角度
            if (relative_xy[0] >= 0) and (relative_xy[1] == 0):
                a = 0
            elif (relative_xy[0] < 0) and (relative_xy[1] == 0):
                a = np.pi
            elif (relative_xy[0] >= 0) and (relative_xy[1] > 0):  # 第一象限
                a = np.arctan(relative_xy[1] / relative_xy[0])
            elif (relative_xy[0] < 0) and (relative_xy[1] > 0):  # 第二象限
                a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
            elif (relative_xy[0] < 0) and (relative_xy[1] < 0):  # 第三象限
                a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
            else:  # 第四象限
                a = np.arctan(relative_xy[1] / relative_xy[0]) + 2 * np.pi

            angle[2] = a

            relative_xy = ideal[int(launch[launch_b]), :] - ideal[center, :]

            # 得到角度
            if (relative_xy[0] >= 0) and (relative_xy[1] == 0):
                a = 0
            elif (relative_xy[0] < 0) and (relative_xy[1] == 0):
                a = np.pi
            elif (relative_xy[0] >= 0) and (relative_xy[1] > 0):  # 第一象限
                a = np.arctan(relative_xy[1] / relative_xy[0])
            elif (relative_xy[0] < 0) and (relative_xy[1] > 0):  # 第二象限
                a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
            elif (relative_xy[0] < 0) and (relative_xy[1] < 0):  # 第三象限
                a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
            else:  # 第四象限
                a = np.arctan(relative_xy[1] / relative_xy[0]) + 2 * np.pi

            angle[3] = a

            print('beta1', np.rad2deg(angle[2]))
            print('beta2', np.rad2deg(angle[3]))

            relative_xy = ideal[int(i), :] - ideal[center, :]

            # 得到角度
            if (relative_xy[0] >= 0) and (relative_xy[1] == 0):
                a = 0
            elif (relative_xy[0] < 0) and (relative_xy[1] == 0):
                a = np.pi
            elif (relative_xy[0] >= 0) and (relative_xy[1] > 0):  # 第一象限
                a = np.arctan(relative_xy[1] / relative_xy[0])
            elif (relative_xy[0] < 0) and (relative_xy[1] > 0):  # 第二象限
                a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
            elif (relative_xy[0] < 0) and (relative_xy[1] < 0):  # 第三象限
                a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
            else:  # 第四象限
                a = np.arctan(relative_xy[1] / relative_xy[0]) + 2 * np.pi

            ideal_angle = a

            # ideal_angle = get_angle_xy(np.array([base_vec, ideal[center, :], ideal[int(i), :]]))

            # 求相对坐标
            # 注意相对坐标是指极径用相对值表示！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            position = root(f_p2, [1, ideal_angle], args=(angle)).x
            # 近似转化成绝对坐标
            position[0] = position[0] * R

            position_measure_xy_list[k, :] = np.array([ideal[center, 0] + position[0] * np.cos(position[1]),
                                                       ideal[center, 1] + position[0] * np.sin(position[1])])

        position_measure_xy = np.mean(position_measure_xy_list, axis=0)

        position_esimate_xy = position_measure_xy

        position_now_xy = position_hat[i, :]
        ideal_xy = ideal[i, :]

        w = ideal_xy - position_esimate_xy
        # 位置调整
        position_hat_temp = position_now_xy + lr * w

        position_hat[i, :] = position_hat_temp



'''*********************************************************************'''
# step2: 固定4,8,9，调整以5为圆心的圆
index_launch = np.zeros((2,3))
index_launch[0,:] = np.array([4,9,3])
index_launch[1,:] = np.array([2,6,8])

index_measure = np.zeros((2,3))
index_measure[0,:] = np.array([2,6,8])
index_measure[1,:] = np.array([4,9,3])

# 圆心
center = 5

launch = np.zeros(3)

for iter in range(iter_max):
    # 选择发送信号无人机
    launch = index_launch[iter % 2, :].tolist()

    # 带调整的无人机编号
    measured = index_measure[iter % 2, :].tolist()



    # 对每个待调整无人机调整
    for i in measured:
        i = int(i)

        if (i == 4) or (i == 8) or (i == 9):
            continue

        # 分三次测量
        alpha = np.zeros(3)
        # 角度center-i-launch[0]
        alpha[0] = get_angle_xy(np.array([actual[center, :], actual[i, :], actual[int(launch[0]), :]]))
        # 角度center-i-launch[1]
        alpha[1] = get_angle_xy(np.array([actual[center, :], actual[i, :], actual[int(launch[1]), :]]))
        # 角度center-i-launch[2]
        alpha[2] = get_angle_xy(np.array([actual[center, :], actual[i, :], actual[int(launch[2]), :]]))

        position_measure_xy_list = np.zeros((3, 2))

        for k in range(3):
            # 估算i的位置
            angle = np.zeros(4)  # angle1,angle2,beta1,beta2
            # 每次取launch_a,launch_b
            launch_a = k % 3
            launch_b = (k + 1) % 3

            # alpha
            angle[0] = alpha[launch_a]
            angle[1] = alpha[launch_b]

            print('发射点1', launch[launch_a])
            print('发射点2', launch[launch_b])
            print('被测点', i)
            print('alpha1', np.rad2deg(angle[0]))
            print('alpha2', np.rad2deg(angle[1]))
            # beta(理论值)
            # angle[2] = get_angle_xy(np.array([ideal[center,:] + base_vec, ideal[center,:], ideal[int(launch[launch_a]),:]]))
            # angle[3] = get_angle_xy(np.array([ideal[center,:] + base_vec, ideal[center,:], ideal[int(launch[launch_b]),:]]))
            relative_xy = ideal[int(launch[launch_a]), :] - ideal[center, :]

            # 得到角度
            if (relative_xy[0] >= 0) and (relative_xy[1] == 0):
                a = 0
            elif (relative_xy[0] < 0) and (relative_xy[1] == 0):
                a = np.pi
            elif (relative_xy[0] >= 0) and (relative_xy[1] > 0):  # 第一象限
                a = np.arctan(relative_xy[1] / relative_xy[0])
            elif (relative_xy[0] < 0) and (relative_xy[1] > 0):  # 第二象限
                a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
            elif (relative_xy[0] < 0) and (relative_xy[1] < 0):  # 第三象限
                a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
            else:  # 第四象限
                a = np.arctan(relative_xy[1] / relative_xy[0]) + 2 * np.pi

            angle[2] = a

            relative_xy = ideal[int(launch[launch_b]), :] - ideal[center, :]

            # 得到角度
            if (relative_xy[0] >= 0) and (relative_xy[1] == 0):
                a = 0
            elif (relative_xy[0] < 0) and (relative_xy[1] == 0):
                a = np.pi
            elif (relative_xy[0] >= 0) and (relative_xy[1] > 0):  # 第一象限
                a = np.arctan(relative_xy[1] / relative_xy[0])
            elif (relative_xy[0] < 0) and (relative_xy[1] > 0):  # 第二象限
                a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
            elif (relative_xy[0] < 0) and (relative_xy[1] < 0):  # 第三象限
                a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
            else:  # 第四象限
                a = np.arctan(relative_xy[1] / relative_xy[0]) + 2 * np.pi

            angle[3] = a

            print('beta1', np.rad2deg(angle[2]))
            print('beta2', np.rad2deg(angle[3]))

            relative_xy = ideal[int(i), :] - ideal[center, :]

            # 得到角度
            if (relative_xy[0] >= 0) and (relative_xy[1] == 0):
                a = 0
            elif (relative_xy[0] < 0) and (relative_xy[1] == 0):
                a = np.pi
            elif (relative_xy[0] >= 0) and (relative_xy[1] > 0):  # 第一象限
                a = np.arctan(relative_xy[1] / relative_xy[0])
            elif (relative_xy[0] < 0) and (relative_xy[1] > 0):  # 第二象限
                a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
            elif (relative_xy[0] < 0) and (relative_xy[1] < 0):  # 第三象限
                a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
            else:  # 第四象限
                a = np.arctan(relative_xy[1] / relative_xy[0]) + 2 * np.pi

            ideal_angle = a

            # ideal_angle = get_angle_xy(np.array([base_vec, ideal[center, :], ideal[int(i), :]]))

            # 求相对坐标
            # 注意相对坐标是指极径用相对值表示！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            position = root(f_p2, [1, ideal_angle], args=(angle)).x
            # 近似转化成绝对坐标
            position[0] = position[0] * R

            position_measure_xy_list[k, :] = np.array([ideal[center, 0] + position[0] * np.cos(position[1]),
                                                       ideal[center, 1] + position[0] * np.sin(position[1])])

        position_measure_xy = np.mean(position_measure_xy_list, axis=0)

        position_esimate_xy = position_measure_xy

        position_now_xy = position_hat[i, :]
        ideal_xy = ideal[i, :]

        w = ideal_xy - position_esimate_xy
        # 位置调整
        position_hat_temp = position_now_xy + lr * w

        position_hat[i, :] = position_hat_temp

'''***********************************************************************'''

# step3: 固定13,8,5，6,调整以9为圆心的圆
index_launch = np.zeros((2,3))
index_launch[0,:] = np.array([6,8,14])
index_launch[1,:] = np.array([5,10,13])

index_measure = np.zeros((2,3))
index_measure[0,:] = np.array([6,8,14])
index_measure[1,:] = np.array([5,10,13])

# 圆心
center = 9

launch = np.zeros(3)

for iter in range(iter_max):
    # 选择发送信号无人机
    launch = index_launch[iter % 2, :].tolist()

    # 带调整的无人机编号
    measured = index_measure[iter % 2, :].tolist()

    # 对每个待调整无人机调整
    for i in measured:
        i = int(i)

        if (i == 13) or (i == 8) or (i == 5) or (i == 6):
            continue

        # 分三次测量
        alpha = np.zeros(3)
        # 角度center-i-launch[0]
        alpha[0] = get_angle_xy(np.array([actual[center, :], actual[i, :], actual[int(launch[0]), :]]))
        # 角度center-i-launch[1]
        alpha[1] = get_angle_xy(np.array([actual[center, :], actual[i, :], actual[int(launch[1]), :]]))
        # 角度center-i-launch[2]
        alpha[2] = get_angle_xy(np.array([actual[center, :], actual[i, :], actual[int(launch[2]), :]]))

        position_measure_xy_list = np.zeros((3, 2))

        for k in range(3):
            # 估算i的位置
            angle = np.zeros(4)  # angle1,angle2,beta1,beta2
            # 每次取launch_a,launch_b
            launch_a = k % 3
            launch_b = (k + 1) % 3

            # alpha
            angle[0] = alpha[launch_a]
            angle[1] = alpha[launch_b]

            print('发射点1', launch[launch_a])
            print('发射点2', launch[launch_b])
            print('被测点', i)
            print('alpha1', np.rad2deg(angle[0]))
            print('alpha2', np.rad2deg(angle[1]))
            # beta(理论值)
            # angle[2] = get_angle_xy(np.array([ideal[center,:] + base_vec, ideal[center,:], ideal[int(launch[launch_a]),:]]))
            # angle[3] = get_angle_xy(np.array([ideal[center,:] + base_vec, ideal[center,:], ideal[int(launch[launch_b]),:]]))
            relative_xy = ideal[int(launch[launch_a]), :] - ideal[center, :]

            # 得到角度
            if (relative_xy[0] >= 0) and (relative_xy[1] == 0):
                a = 0
            elif (relative_xy[0] < 0) and (relative_xy[1] == 0):
                a = np.pi
            elif (relative_xy[0] >= 0) and (relative_xy[1] > 0):  # 第一象限
                a = np.arctan(relative_xy[1] / relative_xy[0])
            elif (relative_xy[0] < 0) and (relative_xy[1] > 0):  # 第二象限
                a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
            elif (relative_xy[0] < 0) and (relative_xy[1] < 0):  # 第三象限
                a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
            else:  # 第四象限
                a = np.arctan(relative_xy[1] / relative_xy[0]) + 2 * np.pi

            angle[2] = a

            relative_xy = ideal[int(launch[launch_b]), :] - ideal[center, :]

            # 得到角度
            if (relative_xy[0] >= 0) and (relative_xy[1] == 0):
                a = 0
            elif (relative_xy[0] < 0) and (relative_xy[1] == 0):
                a = np.pi
            elif (relative_xy[0] >= 0) and (relative_xy[1] > 0):  # 第一象限
                a = np.arctan(relative_xy[1] / relative_xy[0])
            elif (relative_xy[0] < 0) and (relative_xy[1] > 0):  # 第二象限
                a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
            elif (relative_xy[0] < 0) and (relative_xy[1] < 0):  # 第三象限
                a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
            else:  # 第四象限
                a = np.arctan(relative_xy[1] / relative_xy[0]) + 2 * np.pi

            angle[3] = a

            print('beta1', np.rad2deg(angle[2]))
            print('beta2', np.rad2deg(angle[3]))

            relative_xy = ideal[int(i), :] - ideal[center, :]

            # 得到角度
            if (relative_xy[0] >= 0) and (relative_xy[1] == 0):
                a = 0
            elif (relative_xy[0] < 0) and (relative_xy[1] == 0):
                a = np.pi
            elif (relative_xy[0] >= 0) and (relative_xy[1] > 0):  # 第一象限
                a = np.arctan(relative_xy[1] / relative_xy[0])
            elif (relative_xy[0] < 0) and (relative_xy[1] > 0):  # 第二象限
                a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
            elif (relative_xy[0] < 0) and (relative_xy[1] < 0):  # 第三象限
                a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
            else:  # 第四象限
                a = np.arctan(relative_xy[1] / relative_xy[0]) + 2 * np.pi

            ideal_angle = a

            # ideal_angle = get_angle_xy(np.array([base_vec, ideal[center, :], ideal[int(i), :]]))

            # 求相对坐标
            # 注意相对坐标是指极径用相对值表示！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            position = root(f_p2, [1, ideal_angle], args=(angle)).x
            # 近似转化成绝对坐标
            position[0] = position[0] * R

            position_measure_xy_list[k, :] = np.array([ideal[center, 0] + position[0] * np.cos(position[1]),
                                                       ideal[center, 1] + position[0] * np.sin(position[1])])

        position_measure_xy = np.mean(position_measure_xy_list, axis=0)

        position_esimate_xy = position_measure_xy

        position_now_xy = position_hat[i, :]
        ideal_xy = ideal[i, :]

        w = ideal_xy - position_esimate_xy
        # 位置调整
        position_hat_temp = position_now_xy + lr * w

        position_hat[i, :] = position_hat_temp

'''************************************************************'''
# step4:以7为圆心，调整11
index_launch = np.zeros((2,3))
index_launch[0,:] = np.array([12,8,4])
index_launch[1,:] = np.array([13,8,7])

# 圆心
index_center = np.array([7,12])

launch = np.zeros(3)

for iter in range(iter_max):
    # 选择发送信号无人机
    launch = index_launch[iter % 2, :].tolist()

    center = index_center[iter % 2]
    # 对待调整无人机11进行调整
    i = 11

    # 分三次测量
    alpha = np.zeros(3)
    # 角度center-i-launch[0]
    alpha[0] = get_angle_xy(np.array([actual[center, :], actual[i, :], actual[int(launch[0]), :]]))
    # 角度center-i-launch[1]
    alpha[1] = get_angle_xy(np.array([actual[center, :], actual[i, :], actual[int(launch[1]), :]]))
    # 角度center-i-launch[2]
    alpha[2] = get_angle_xy(np.array([actual[center, :], actual[i, :], actual[int(launch[2]), :]]))

    position_measure_xy_list = np.zeros((3, 2))

    for k in range(3):
        # 估算i的位置
        angle = np.zeros(4)  # angle1,angle2,beta1,beta2
        # 每次取launch_a,launch_b
        launch_a = k % 3
        launch_b = (k + 1) % 3

        # alpha
        angle[0] = alpha[launch_a]
        angle[1] = alpha[launch_b]

        print('发射点1', launch[launch_a])
        print('发射点2', launch[launch_b])
        print('被测点', i)
        print('alpha1', np.rad2deg(angle[0]))
        print('alpha2', np.rad2deg(angle[1]))
        # beta(理论值)
        # angle[2] = get_angle_xy(np.array([ideal[center,:] + base_vec, ideal[center,:], ideal[int(launch[launch_a]),:]]))
        # angle[3] = get_angle_xy(np.array([ideal[center,:] + base_vec, ideal[center,:], ideal[int(launch[launch_b]),:]]))
        relative_xy = ideal[int(launch[launch_a]), :] - ideal[center, :]

        # 得到角度
        if (relative_xy[0] >= 0) and (relative_xy[1] == 0):
            a = 0
        elif (relative_xy[0] < 0) and (relative_xy[1] == 0):
            a = np.pi
        elif (relative_xy[0] >= 0) and (relative_xy[1] > 0):  # 第一象限
            a = np.arctan(relative_xy[1] / relative_xy[0])
        elif (relative_xy[0] < 0) and (relative_xy[1] > 0):  # 第二象限
            a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
        elif (relative_xy[0] < 0) and (relative_xy[1] < 0):  # 第三象限
            a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
        else:  # 第四象限
            a = np.arctan(relative_xy[1] / relative_xy[0]) + 2 * np.pi

        angle[2] = a

        relative_xy = ideal[int(launch[launch_b]), :] - ideal[center, :]

        # 得到角度
        if (relative_xy[0] >= 0) and (relative_xy[1] == 0):
            a = 0
        elif (relative_xy[0] < 0) and (relative_xy[1] == 0):
            a = np.pi
        elif (relative_xy[0] >= 0) and (relative_xy[1] > 0):  # 第一象限
            a = np.arctan(relative_xy[1] / relative_xy[0])
        elif (relative_xy[0] < 0) and (relative_xy[1] > 0):  # 第二象限
            a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
        elif (relative_xy[0] < 0) and (relative_xy[1] < 0):  # 第三象限
            a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
        else:  # 第四象限
            a = np.arctan(relative_xy[1] / relative_xy[0]) + 2 * np.pi

        angle[3] = a

        print('beta1', np.rad2deg(angle[2]))
        print('beta2', np.rad2deg(angle[3]))

        relative_xy = ideal[int(i), :] - ideal[center, :]

        # 得到角度
        if (relative_xy[0] >= 0) and (relative_xy[1] == 0):
            a = 0
        elif (relative_xy[0] < 0) and (relative_xy[1] == 0):
            a = np.pi
        elif (relative_xy[0] >= 0) and (relative_xy[1] > 0):  # 第一象限
            a = np.arctan(relative_xy[1] / relative_xy[0])
        elif (relative_xy[0] < 0) and (relative_xy[1] > 0):  # 第二象限
            a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
        elif (relative_xy[0] < 0) and (relative_xy[1] < 0):  # 第三象限
            a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
        else:  # 第四象限
            a = np.arctan(relative_xy[1] / relative_xy[0]) + 2 * np.pi

        ideal_angle = a

        # ideal_angle = get_angle_xy(np.array([base_vec, ideal[center, :], ideal[int(i), :]]))

        # 求相对坐标
        # 注意相对坐标是指极径用相对值表示！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        position = root(f_p2, [1, ideal_angle], args=(angle)).x
        # 近似转化成绝对坐标
        position[0] = position[0] * R

        position_measure_xy_list[k, :] = np.array([ideal[center, 0] + position[0] * np.cos(position[1]),
                                                   ideal[center, 1] + position[0] * np.sin(position[1])])

    position_measure_xy = np.mean(position_measure_xy_list, axis=0)

    position_esimate_xy = position_measure_xy

    position_now_xy = position_hat[i, :]
    ideal_xy = ideal[i, :]

    w = ideal_xy - position_esimate_xy
    # 位置调整
    position_hat_temp = position_now_xy + lr * w

    position_hat[i, :] = position_hat_temp

'''************************************************************'''
# step5:以2为圆心，调整1
index_launch = np.zeros((2,3))
index_launch[0,:] = np.array([3,5,4])
index_launch[1,:] = np.array([2,5,6])

# 圆心
index_center = np.array([2,3])

launch = np.zeros(3)

for iter in range(iter_max):
    # 选择发送信号无人机
    launch = index_launch[iter % 2, :].tolist()

    center = index_center[iter % 2]
    # 对待调整无人机1进行调整
    i = 1

    # 分三次测量
    alpha = np.zeros(3)
    # 角度center-i-launch[0]
    alpha[0] = get_angle_xy(np.array([actual[center, :], actual[i, :], actual[int(launch[0]), :]]))
    # 角度center-i-launch[1]
    alpha[1] = get_angle_xy(np.array([actual[center, :], actual[i, :], actual[int(launch[1]), :]]))
    # 角度center-i-launch[2]
    alpha[2] = get_angle_xy(np.array([actual[center, :], actual[i, :], actual[int(launch[2]), :]]))

    position_measure_xy_list = np.zeros((3, 2))

    for k in range(3):
        # 估算i的位置
        angle = np.zeros(4)  # angle1,angle2,beta1,beta2
        # 每次取launch_a,launch_b
        launch_a = k % 3
        launch_b = (k + 1) % 3

        # alpha
        angle[0] = alpha[launch_a]
        angle[1] = alpha[launch_b]

        print('发射点1', launch[launch_a])
        print('发射点2', launch[launch_b])
        print('被测点', i)
        print('alpha1', np.rad2deg(angle[0]))
        print('alpha2', np.rad2deg(angle[1]))
        # beta(理论值)
        # angle[2] = get_angle_xy(np.array([ideal[center,:] + base_vec, ideal[center,:], ideal[int(launch[launch_a]),:]]))
        # angle[3] = get_angle_xy(np.array([ideal[center,:] + base_vec, ideal[center,:], ideal[int(launch[launch_b]),:]]))
        relative_xy = ideal[int(launch[launch_a]), :] - ideal[center, :]

        # 得到角度
        if (relative_xy[0] >= 0) and (relative_xy[1] == 0):
            a = 0
        elif (relative_xy[0] < 0) and (relative_xy[1] == 0):
            a = np.pi
        elif (relative_xy[0] >= 0) and (relative_xy[1] > 0):  # 第一象限
            a = np.arctan(relative_xy[1] / relative_xy[0])
        elif (relative_xy[0] < 0) and (relative_xy[1] > 0):  # 第二象限
            a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
        elif (relative_xy[0] < 0) and (relative_xy[1] < 0):  # 第三象限
            a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
        else:  # 第四象限
            a = np.arctan(relative_xy[1] / relative_xy[0]) + 2 * np.pi

        angle[2] = a

        relative_xy = ideal[int(launch[launch_b]), :] - ideal[center, :]

        # 得到角度
        if (relative_xy[0] >= 0) and (relative_xy[1] == 0):
            a = 0
        elif (relative_xy[0] < 0) and (relative_xy[1] == 0):
            a = np.pi
        elif (relative_xy[0] >= 0) and (relative_xy[1] > 0):  # 第一象限
            a = np.arctan(relative_xy[1] / relative_xy[0])
        elif (relative_xy[0] < 0) and (relative_xy[1] > 0):  # 第二象限
            a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
        elif (relative_xy[0] < 0) and (relative_xy[1] < 0):  # 第三象限
            a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
        else:  # 第四象限
            a = np.arctan(relative_xy[1] / relative_xy[0]) + 2 * np.pi

        angle[3] = a

        print('beta1', np.rad2deg(angle[2]))
        print('beta2', np.rad2deg(angle[3]))

        relative_xy = ideal[int(i), :] - ideal[center, :]

        # 得到角度
        if (relative_xy[0] >= 0) and (relative_xy[1] == 0):
            a = 0
        elif (relative_xy[0] < 0) and (relative_xy[1] == 0):
            a = np.pi
        elif (relative_xy[0] >= 0) and (relative_xy[1] > 0):  # 第一象限
            a = np.arctan(relative_xy[1] / relative_xy[0])
        elif (relative_xy[0] < 0) and (relative_xy[1] > 0):  # 第二象限
            a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
        elif (relative_xy[0] < 0) and (relative_xy[1] < 0):  # 第三象限
            a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
        else:  # 第四象限
            a = np.arctan(relative_xy[1] / relative_xy[0]) + 2 * np.pi

        ideal_angle = a

        # ideal_angle = get_angle_xy(np.array([base_vec, ideal[center, :], ideal[int(i), :]]))

        # 求相对坐标
        # 注意相对坐标是指极径用相对值表示！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        position = root(f_p2, [1, ideal_angle], args=(angle)).x
        # 近似转化成绝对坐标
        position[0] = position[0] * R

        position_measure_xy_list[k, :] = np.array([ideal[center, 0] + position[0] * np.cos(position[1]),
                                                   ideal[center, 1] + position[0] * np.sin(position[1])])

    position_measure_xy = np.mean(position_measure_xy_list, axis=0)

    position_esimate_xy = position_measure_xy

    position_now_xy = position_hat[i, :]
    ideal_xy = ideal[i, :]

    w = ideal_xy - position_esimate_xy
    # 位置调整
    position_hat_temp = position_now_xy + lr * w

    position_hat[i, :] = position_hat_temp

'''************************************************************'''
# step6:以10为圆心，调整15
index_launch = np.zeros((2,3))
index_launch[0,:] = np.array([6,9,14])
index_launch[1,:] = np.array([13,9,10])
# 圆心
index_center = np.array([10,14])

launch = np.zeros(3)

for iter in range(iter_max):
    # 选择发送信号无人机
    launch = index_launch[iter % 2, :].tolist()

    center = index_center[iter % 2]
    # 对待调整无人机1进行调整
    i = 15

    # 分三次测量
    alpha = np.zeros(3)
    # 角度center-i-launch[0]
    alpha[0] = get_angle_xy(np.array([actual[center, :], actual[i, :], actual[int(launch[0]), :]]))
    # 角度center-i-launch[1]
    alpha[1] = get_angle_xy(np.array([actual[center, :], actual[i, :], actual[int(launch[1]), :]]))
    # 角度center-i-launch[2]
    alpha[2] = get_angle_xy(np.array([actual[center, :], actual[i, :], actual[int(launch[2]), :]]))

    position_measure_xy_list = np.zeros((3, 2))

    for k in range(3):
        # 估算i的位置
        angle = np.zeros(4)  # angle1,angle2,beta1,beta2
        # 每次取launch_a,launch_b
        launch_a = k % 3
        launch_b = (k + 1) % 3

        # alpha
        angle[0] = alpha[launch_a]
        angle[1] = alpha[launch_b]

        print('发射点1', launch[launch_a])
        print('发射点2', launch[launch_b])
        print('被测点', i)
        print('alpha1', np.rad2deg(angle[0]))
        print('alpha2', np.rad2deg(angle[1]))
        # beta(理论值)
        # angle[2] = get_angle_xy(np.array([ideal[center,:] + base_vec, ideal[center,:], ideal[int(launch[launch_a]),:]]))
        # angle[3] = get_angle_xy(np.array([ideal[center,:] + base_vec, ideal[center,:], ideal[int(launch[launch_b]),:]]))
        relative_xy = ideal[int(launch[launch_a]), :] - ideal[center, :]

        # 得到角度
        if (relative_xy[0] >= 0) and (relative_xy[1] == 0):
            a = 0
        elif (relative_xy[0] < 0) and (relative_xy[1] == 0):
            a = np.pi
        elif (relative_xy[0] >= 0) and (relative_xy[1] > 0):  # 第一象限
            a = np.arctan(relative_xy[1] / relative_xy[0])
        elif (relative_xy[0] < 0) and (relative_xy[1] > 0):  # 第二象限
            a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
        elif (relative_xy[0] < 0) and (relative_xy[1] < 0):  # 第三象限
            a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
        else:  # 第四象限
            a = np.arctan(relative_xy[1] / relative_xy[0]) + 2 * np.pi

        angle[2] = a

        relative_xy = ideal[int(launch[launch_b]), :] - ideal[center, :]

        # 得到角度
        if (relative_xy[0] >= 0) and (relative_xy[1] == 0):
            a = 0
        elif (relative_xy[0] < 0) and (relative_xy[1] == 0):
            a = np.pi
        elif (relative_xy[0] >= 0) and (relative_xy[1] > 0):  # 第一象限
            a = np.arctan(relative_xy[1] / relative_xy[0])
        elif (relative_xy[0] < 0) and (relative_xy[1] > 0):  # 第二象限
            a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
        elif (relative_xy[0] < 0) and (relative_xy[1] < 0):  # 第三象限
            a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
        else:  # 第四象限
            a = np.arctan(relative_xy[1] / relative_xy[0]) + 2 * np.pi

        angle[3] = a

        print('beta1', np.rad2deg(angle[2]))
        print('beta2', np.rad2deg(angle[3]))

        relative_xy = ideal[int(i), :] - ideal[center, :]

        # 得到角度
        if (relative_xy[0] >= 0) and (relative_xy[1] == 0):
            a = 0
        elif (relative_xy[0] < 0) and (relative_xy[1] == 0):
            a = np.pi
        elif (relative_xy[0] >= 0) and (relative_xy[1] > 0):  # 第一象限
            a = np.arctan(relative_xy[1] / relative_xy[0])
        elif (relative_xy[0] < 0) and (relative_xy[1] > 0):  # 第二象限
            a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
        elif (relative_xy[0] < 0) and (relative_xy[1] < 0):  # 第三象限
            a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
        else:  # 第四象限
            a = np.arctan(relative_xy[1] / relative_xy[0]) + 2 * np.pi

        ideal_angle = a

        # ideal_angle = get_angle_xy(np.array([base_vec, ideal[center, :], ideal[int(i), :]]))

        # 求相对坐标
        # 注意相对坐标是指极径用相对值表示！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        position = root(f_p2, [1, ideal_angle], args=(angle)).x
        # 近似转化成绝对坐标
        position[0] = position[0] * R

        position_measure_xy_list[k, :] = np.array([ideal[center, 0] + position[0] * np.cos(position[1]),
                                                   ideal[center, 1] + position[0] * np.sin(position[1])])

    position_measure_xy = np.mean(position_measure_xy_list, axis=0)

    position_esimate_xy = position_measure_xy

    position_now_xy = position_hat[i, :]
    ideal_xy = ideal[i, :]

    w = ideal_xy - position_esimate_xy
    # 位置调整
    position_hat_temp = position_now_xy + lr * w

    position_hat[i, :] = position_hat_temp

length =[]
for i in range(1,16):
    for j in range(1,16):
        if graph[i,j] == 1:
            length.append(np.sqrt(np.dot(position_hat[i,:]-position_hat[j,:],position_hat[i,:]-position_hat[j,:])))

mean_length = sum(length) / len(length)
print('mean_length',mean_length)
print('length')
print(length)
'''
# 内圈以6轮为单位进行迭代
index_launch = np.zeros((6,3))
index_launch[0,:] = np.array([4,9,12])
index_launch[1,:] = np.array([7,5,13])
index_launch[2,:] = np.array([4,9,3])
index_launch[3,:] = np.array([2,6,8])
index_launch[4,:] = np.array([6,8,14])
index_launch[5,:] = np.array([5,10,13])

index_measure = np.zeros((6,3))
index_measure[0,:] = np.array([7,5,13])
index_measure[1,:] = np.array([4,9,12])
index_measure[2,:] = np.array([2,6,8])
index_measure[3,:] = np.array([4,9,3])
index_measure[4,:] = np.array([5,10,13])
index_measure[5,:] = np.array([6,8,14])

#index_center = np.array([8,5,9,8,5,9])
index_center = np.array([8,8,5,5,9,9])

#
iter_max = 50
lr = 0.5
d = 50
t = 1

launch = np.zeros(3)
for iter in range(iter_max):
    # 选择发送信号无人机
    launch = index_launch[iter % 6,:].tolist()

    # 带调整的无人机编号
    measured = index_measure[iter % 6, :].tolist()

    # 圆心坐标
    center = index_center[iter % 6]

    # 对每个待调整无人机调整
    for i in measured:
        i = int(i)
        
        if (i != 8) and (i != 5) and (i != 9):
            continue
        
        # 分三次测量
        alpha = np.zeros(3)
        # 角度center-i-launch[0]
        alpha[0] = get_angle_xy(np.array([actual[center, :], actual[i, :], actual[int(launch[0]), :]]))
        # 角度center-i-launch[1]
        alpha[1] = get_angle_xy(np.array([actual[center, :], actual[i, :], actual[int(launch[1]), :]]))
        # 角度center-i-launch[2]
        alpha[2] = get_angle_xy(np.array([actual[center, :], actual[i, :], actual[int(launch[2]), :]]))

        position_measure_xy_list = np.zeros((3, 2))

        for k in range(3):
            # 估算i的位置
            angle = np.zeros(4)  # angle1,angle2,beta1,beta2
            # 每次取launch_a,launch_b
            launch_a = k % 3
            launch_b = (k + 1) % 3

            # alpha
            angle[0] = alpha[launch_a]
            angle[1] = alpha[launch_b]

            print('发射点1',launch[launch_a])
            print('发射点2',launch[launch_b])
            print('被测点',i)
            print('alpha1',np.rad2deg(angle[0]))
            print('alpha2', np.rad2deg(angle[1]))
            # beta(理论值)
            # angle[2] = get_angle_xy(np.array([ideal[center,:] + base_vec, ideal[center,:], ideal[int(launch[launch_a]),:]]))
            # angle[3] = get_angle_xy(np.array([ideal[center,:] + base_vec, ideal[center,:], ideal[int(launch[launch_b]),:]]))
            relative_xy = ideal[int(launch[launch_a]),:] - ideal[center,:]

            # 得到角度
            if (relative_xy[0] >= 0) and (relative_xy[1] >= 0):  # 第一象限
                a = np.arctan(relative_xy[1] / relative_xy[0])
            elif (relative_xy[0] < 0) and (relative_xy[1] >= 0):  # 第二象限
                a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
            elif (relative_xy[0] < 0) and (relative_xy[1] < 0):  # 第三象限
                a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
            else:  # 第四象限
                a = np.arctan(relative_xy[1] / relative_xy[0]) + 2 * np.pi

            angle[2] = a

            relative_xy = ideal[int(launch[launch_b]), :] - ideal[center, :]

            # 得到角度
            if (relative_xy[0] >= 0) and (relative_xy[1] >= 0):  # 第一象限
                a = np.arctan(relative_xy[1] / relative_xy[0])
            elif (relative_xy[0] < 0) and (relative_xy[1] >= 0):  # 第二象限
                a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
            elif (relative_xy[0] < 0) and (relative_xy[1] < 0):  # 第三象限
                a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
            else:  # 第四象限
                a = np.arctan(relative_xy[1] / relative_xy[0]) + 2 * np.pi

            angle[3] = a



            print('beta1', np.rad2deg(angle[2]))
            print('beta2', np.rad2deg(angle[3]))

            relative_xy = ideal[int(i), :] - ideal[center, :]

            # 得到角度
            if (relative_xy[0] >= 0) and (relative_xy[1] >= 0):  # 第一象限
                a = np.arctan(relative_xy[1] / relative_xy[0])
            elif (relative_xy[0] < 0) and (relative_xy[1] >= 0):  # 第二象限
                a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
            elif (relative_xy[0] < 0) and (relative_xy[1] < 0):  # 第三象限
                a = np.arctan(relative_xy[1] / relative_xy[0]) + np.pi
            else:  # 第四象限
                a = np.arctan(relative_xy[1] / relative_xy[0]) + 2 * np.pi

            ideal_angle = a

            # ideal_angle = get_angle_xy(np.array([base_vec, ideal[center, :], ideal[int(i), :]]))

            # 求相对坐标
            # 注意相对坐标是指极径用相对值表示！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            position = root(f_p2, [1, ideal_angle], args=(angle)).x
            # 近似转化成绝对坐标
            position[0] = position[0] * R

            position_measure_xy_list[k,:] = np.array([ideal[center,0] + position[0] * np.cos(position[1]), ideal[center,1] + position[0] * np.sin(position[1])])

        position_measure_xy = np.mean(position_measure_xy_list, axis=0)

        position_esimate_xy = position_measure_xy

        position_now_xy = position_hat[i,:]
        ideal_xy = ideal[i,:]

        w = ideal_xy - position_esimate_xy
        # 位置调整
        position_hat_temp = position_now_xy + lr * w


        position_hat[i, :] = position_hat_temp
        
        



'''


print('position_hat',position_hat)

plt.scatter(position_hat[1:,0],position_hat[1:,1])
plt.show()



