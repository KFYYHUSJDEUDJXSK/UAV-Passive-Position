import numpy as np
from scipy.optimize import root

from problem1 import get_angle

# 计算以某个发射信号的固定飞机与原点为弦长，测量的角度为半径的圆
# 输入:选择测量飞机的极角theta，被测飞机测量方位角alpha
# 输出:半径r，两种对称可能的圆心直角坐标[x0,y0]

def get_circle(theta,alpha):
    r = 1 / (2 * np.sin(alpha))
    x0_1 = np.cos(theta + alpha - np.pi / 2) / (2 * np.sin(alpha))
    y0_1 = np.sin(theta + alpha - np.pi / 2) / (2 * np.sin(alpha))
    x0_2 = np.cos(theta - alpha + np.pi / 2) / (2 * np.sin(alpha))
    y0_2 = np.sin(theta - alpha + np.pi / 2) / (2 * np.sin(alpha))

    return r,np.array([x0_1,y0_1]),np.array([x0_2,y0_2])

def get_intersection(x,center_1,center_2):
    x =center_1 + center_2
    return x
"""
Problem 2
"""
# 载入数据
R = 0.23

plane = np.array([[0, np.deg2rad(0)], [100, np.deg2rad(0)], [98, np.deg2rad(40.10)],
                  [112, np.deg2rad(80.21)], [105, np.deg2rad(119.75)], [98, np.deg2rad(159.86)],
                  [112, np.deg2rad(199.96)], [105, np.deg2rad(240.07)], [98, np.deg2rad(280.17)],
                  [112, np.deg2rad(320.28)]])
# 比例缩放
for i in range(10):
    plane[i,0] = plane[i,0] / 100
measured = 5
launch = 6
alpha_1 = get_angle(np.array([plane[0,:],plane[measured,:],plane[1,:]])) #角0-measured-1
alpha_2 = get_angle(np.array([plane[0,:],plane[measured,:],plane[launch,:]])) #角0-measured-launch
print('alpha',alpha_1,alpha_2)

# 得到主圆
r_main,center_main_1,center_main_2 = get_circle(plane[1,1],alpha_1)

# 确定主圆圆心
if plane[measured,1] < np.np: # 上方
    if center_main_1[1] > 0:
        center_main = center_main_1
    else:
        center_main = center_main_2

print("主圆：")
print(r_main,center_main_1,center_main_2)

# 遍历对除0,1号飞机和被测飞机
for i in range(2,10):
    if i == measured:
        continue
    # 得到副圆
    alpha_h = get_angle(np.array([plane[0,:],plane[measured,:],plane[i,:]])) #角0-measured-i
    r_h, center_h_1, center_h_2 = get_circle(plane[i, 1], alpha_h)

    # 求交点

'''
# 以第6个点尝试
point = np.array([1,np.deg2rad(5 * 40)])
alpha = np.deg2rad(40)
print(get_circle(point[1],alpha))
'''
