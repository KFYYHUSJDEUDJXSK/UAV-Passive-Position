import numpy as np
from scipy.optimize import root


# 为了便于运算，中间的角度均采用弧度制
def get_angle(point):  # 对point[0,1,2]，返回以1为顶点的角度
    # 得到直角坐标
    x = np.array([[point[0, 0] * np.cos(point[0, 1]), point[0, 0] * np.sin(point[0, 1])],
                  [point[1, 0] * np.cos(point[1, 1]), point[1, 0] * np.sin(point[1, 1])],
                  [point[2, 0] * np.cos(point[2, 1]), point[2, 0] * np.sin(point[2, 1])]])

    vec_10 = x[0, :] - x[1, :]
    vec_12 = x[2, :] - x[1, :]

    cos_angle = np.dot(vec_10, vec_12) / ((np.sqrt(np.dot(vec_10, vec_10)) * np.sqrt(np.dot(vec_12, vec_12))) + 1e-6)
    angle = np.arccos(cos_angle)

    return angle


def f_p2(x, rad):  # x:[ro,theta],rad:[alpha1,alpha2,beta1,beta2],角用弧度制表示
    eqs = []
    eqs.append(
        (1 - np.cos(rad[0]) ** 2) * x[0] ** 2 - 2 * x[0] * np.cos(x[1] - rad[2]) * (1 - np.cos(rad[0]) ** 2) + np.cos(
            x[1] - rad[2]) ** 2 - np.cos(rad[0]) ** 2)
    eqs.append(
        (1 - np.cos(rad[1]) ** 2) * x[0] ** 2 - 2 * x[0] * np.cos(x[1] - rad[3]) * (1 - np.cos(rad[1]) ** 2) + np.cos(
            x[1] - rad[3]) ** 2 - np.cos(rad[1]) ** 2)

    return eqs

# 求解
# 对问题二，尝试加一架编号未知的飞机，接触被测目标飞机的位置
# 输入：设发信号的编号未知飞机为?，被测飞机为x,三个角度信息theta[0x1,0x?,1x?]
# 求解变量: x[pho,theta,beta] (beta为编号未知的飞机的极角)

print(np.deg2rad(0))
point = np.array([[0, np.deg2rad(0)], [100, np.deg2rad(0)], [100, np.deg2rad(40.10)],
                  [112, np.deg2rad(80.21)], [105, np.deg2rad(119.75)], [98, np.deg2rad(159.86)],
                  [112, np.deg2rad(199.96)], [105, np.deg2rad(240.07)], [98, np.deg2rad(280.17)],
                  [112, np.deg2rad(320.28)]])

point_hat = np.zeros((10, 2))
point_hat[0, :] = point[0, :]
point_hat[1, :] = point[1, :]
point_hat[2, :] = point[2, :]

# 用 1，2解其他点的坐标
for i in range(3, 10):
    # 求夹角
    angle = np.zeros(4)  # angle1,angle2,beta1,beta2
    # alpha
    angle[0] = get_angle(np.array([point[0, :], point[i, :], point[1, :]]))  # 0i1
    angle[1] = get_angle(np.array([point[0, :], point[i, :], point[2, :]]))  # 0i2
    # beta
    angle[2] = point[1, 1]
    angle[3] = point[2, 1]

    print('i=', i, 'alpha1:', np.rad2deg(angle[0]), 'alpha2:', np.rad2deg(angle[1]), 'beta1:',
          np.rad2deg(angle[2]), 'beta2:', np.rad2deg(angle[3]))
    # 求相对坐标
    # 注意相对坐标是指极径用相对值表示！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    point_hat[i, :] = root(f_p2, [1, point[i, 1]], args=(angle)).x

    rad = angle
    x = point[i, :]


print('point')
print(point)

print('point_hat')
print(point_hat)



