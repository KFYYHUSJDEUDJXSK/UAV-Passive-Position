import numpy as np
import random
from scipy.optimize import root
from problem1 import get_angle,f_p2
from matplotlib import pyplot as plt

# 载入数据
plane = np.array([[0, np.deg2rad(0)], [100, np.deg2rad(0)], [98, np.deg2rad(40.10)],
                  [112, np.deg2rad(80.21)], [105, np.deg2rad(119.75)], [98, np.deg2rad(159.86)],
                  [112, np.deg2rad(199.96)], [105, np.deg2rad(240.07)], [98, np.deg2rad(280.17)],
                  [112, np.deg2rad(320.28)]])

plane_hat = plane
iter_max = 44
lr = 0.8
d = 100
fixed = 6
t = 0.5

position_remember_xy = np.zeros((10,2))

for iter in range(iter_max):
    # 在1-9中随机选择一个无人机（不能是fixed），参与发射信号
    while 1:
        x = random.randint(1, 9)
        if x != fixed:
            break
    # 发射信号,其余无人机得到角度
    for i in range(1,10):
        if (i == x) or (i == fixed):
            continue

        # 角度0-i-d=fixed
        alpha_1 = get_angle(np.array([plane[0,:],plane[i,:],plane[fixed,:]]))
        # 角度0ix
        alpha_2 = get_angle(np.array([plane[0, :], plane[i, :], plane[x, :]]))

        # 估算i的位置
        angle = np.zeros(4)  # angle1,angle2,beta1,beta2
        # alpha
        angle[0] = alpha_1
        angle[1] = alpha_2
        # beta
        angle[2] = (fixed - 1) * 2 / 9 * np.pi
        angle[3] = (x - 1) * 2 / 9 * np.pi# 注意这个角是不知道的！！！！！！！！想办法改
        ideal_angle = (i - 1) * 2 / 9 * np.pi
        # 求相对坐标
        # 注意相对坐标是指极径用相对值表示！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        position = root(f_p2, [1, ideal_angle], args=(angle)).x
        # 近似转化成绝对坐标
        position[0] = position[0] * d
        position_measure_xy = np.array([position[0] * np.cos(position[1]),position[0] * np.sin(position[1])])

        # 首次更新记忆坐标
        if not np.any(position_remember_xy[i,:]):
            position_remember_xy[i, :] = position_measure_xy

        position_esimate_xy = t * position_measure_xy + (1 - t) * position_remember_xy[i, :]

        position_now_xy = np.array([plane[i,0] * np.cos(plane[i,1]),plane[i,0] * np.sin(plane[i,1])])
        ideal_xy = np.array([d * np.cos(ideal_angle),d * np.sin(ideal_angle)])
        w = ideal_xy - position_esimate_xy
        # 位置调整
        position_new_xy = position_now_xy + lr * w

        # 更新记忆位置
        position_remember_xy[i, :] = position_measure_xy + lr * w

        # 得到角度
        if (position_new_xy[0] >= 0) and (position_new_xy[1] >= 0): # 第一象限
            a = np.arctan(position_new_xy[1]/position_new_xy[0])
        elif (position_new_xy[0] < 0) and (position_new_xy[1] >= 0): # 第二象限
            a = np.arctan(position_new_xy[1]/position_new_xy[0]) + np.pi
        elif (position_new_xy[0] < 0) and (position_new_xy[1] < 0): # 第三象限
            a = np.arctan(position_new_xy[1]/position_new_xy[0]) + np.pi
        else: # 第四象限
            a = np.arctan(position_new_xy[1]/position_new_xy[0]) + 2 * np.pi
        plane_hat[i,:] = np.array([np.sqrt(position_new_xy[0]**2 + position_new_xy[1]**2),a])

print('/******************************************************/')
print('plane_hat')
print(plane_hat)


ax1 = plt.subplot(121, projection='polar')  #极坐标轴
# ax1.scatter(plane[:,1],plane[:,0],color='r')
ax1.scatter(plane_hat[:,1],plane_hat[:,0],color='b')
plt.show()

