import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 为了便于运算，中间的角度均采用弧度制
def vet_add(point1, point2):  # point(\rou,\theta)
    x1 = point1[0] * np.cos(point1[1])
    y1 = point1[0] * np.sin(point1[1])
    x2 = point2[0] * np.cos(point2[1])
    y2 = point2[0] * np.sin(point2[1])
    x = x1 + x2
    y = y1 + y2
    rou = np.linalg.norm([x, y])

    if (x >= 0) and (y >= 0):  # 第一象限
        theta = np.arctan(y / x)
    elif (x < 0) and (y >= 0):  # 第二象限
        theta = np.arctan(y / x) + np.pi
    elif (x < 0) and (y < 0):  # 第三象限
        theta = np.arctan(y / x) + np.pi
    else:  # 第四象限
        theta = np.arctan(y / x) + 2 * np.pi
    return np.array([rou, theta])


def vet_min(point1, point2):  # point(\rou,\theta)
    point2[1] += np.pi
    res = vet_add(point1, point2)
    return res


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

# print(np.deg2rad(0))
# point = np.array([[0, np.deg2rad(0)], [100, np.deg2rad(0)], [100, np.deg2rad(40.10)],
#                   [112, np.deg2rad(80.21)], [105, np.deg2rad(119.75)], [98, np.deg2rad(159.86)],
#                   [112, np.deg2rad(199.96)], [105, np.deg2rad(240.07)], [98, np.deg2rad(280.17)],
#                   [112, np.deg2rad(320.28)]])
#
# point_hat = np.zeros((10, 2))
# point_hat[0, :] = point[0, :]
# point_hat[1, :] = point[1, :]
# point_hat[2, :] = point[2, :]
#
# # 用 1，2解其他点的坐标
# for i in range(3, 10):
#     # 求夹角
#     angle = np.zeros(4)  # angle1,angle2,beta1,beta2
#     # alpha
#     angle[0] = get_angle(np.array([point[0, :], point[i, :], point[1, :]]))  # 0i1
#     angle[1] = get_angle(np.array([point[0, :], point[i, :], point[2, :]]))  # 0i2
#     # beta
#     angle[2] = point[1, 1]
#     angle[3] = point[2, 1]
#
#     print('i=', i, 'alpha1:', np.rad2deg(angle[0]), 'alpha2:', np.rad2deg(angle[1]), 'beta1:',
#           np.rad2deg(angle[2]), 'beta2:', np.rad2deg(angle[3]))
#     # 求相对坐标
#     # 注意相对坐标是指极径用相对值表示！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
#     point_hat[i, :] = root(f_p2, [1, point[i, 1]], args=(angle)).x
#
#     rad = angle
#     x = point[i, :]
#
#
# print('point')
# print(point)
#
# print('point_hat')
# print(point_hat)

if __name__ == "__main__":
    np.random.seed(41)
    drs_unbaised = {"dr0": [0, 0], 'dr2': [1, 40 / 180 * np.pi], 'dr3': [1, 80 / 180 * np.pi],
                    'dr4': [1, 120 / 180 * np.pi], 'dr5': [1, 160 / 180 * np.pi]
        , 'dr6': [1, 200 / 180 * np.pi], 'dr7': [1, 240 / 180 * np.pi], 'dr8': [1, 280 / 180 * np.pi],
                    'dr9': [1, 320 / 180 * np.pi], 'dr1': [1, 0]}
    epochs = 1000
    drtg = ["dr1","dr3","dr4","dr6","dr7","dr9"]
    dr1 = "dr2"
    dr2 = "dr5"
    dr3 = "dr8"
    diffs_sig = np.array([0, 1]).reshape(1, 2)
    diffs_mul = np.array([0, 1]).reshape(1, 2)
    mean_diffs_sig = np.array([])
    mean_diffs_mul = np.array([])
    max_noise = 0.16

    single_sim = True
    for i in range(6):
        dr = drtg[i]
        diffs_sig = np.array([0, 1]).reshape(1, 2)
        diffs_mul = np.array([0, 1]).reshape(1, 2)
        _mean_diffs_sig = np.array([])
        _mean_diffs_mul = np.array([])
        for epoch in range(epochs):
            _dr_actual = [np.random.uniform(0, 1) ** 0.5 * max_noise, np.random.uniform(0, 2 * np.pi)]
            _dr1_actual = [np.random.uniform(0, 1) ** 0.5 * max_noise, np.random.uniform(0, 2 * np.pi)]
            _dr2_actual = [np.random.uniform(0, 1) ** 0.5 * max_noise, np.random.uniform(0, 2 * np.pi)]
            dr_actual = vet_add(drs_unbaised[dr], _dr_actual)
            dr1_actual = vet_add(drs_unbaised[dr1], _dr1_actual)
            dr2_actual = vet_add(drs_unbaised[dr2], _dr2_actual)
            pt_buff_1 = np.array([dr1_actual, dr_actual, np.array([0, 0])])
            alpha1 = get_angle(pt_buff_1)
            pt_buff_2 = np.array([np.array([0, 0]), dr_actual, dr2_actual])
            alpha2 = get_angle(pt_buff_2)
            angle = [alpha1, alpha2, drs_unbaised[dr1][1], drs_unbaised[dr2][1]]
            dr_pre = root(f_p2, np.array(drs_unbaised[dr]), args=(angle)).x
            diff = vet_min(dr_pre, dr_actual)
            diff = np.reshape(diff, (1, 2))
            diffs_sig = np.append(diffs_sig, diff, axis=0)
            if epoch % 10 == 0:
                _mean_diffs_sig = np.append(_mean_diffs_sig, np.mean(diffs_sig[:, 0]))
        if i == 0:
            mean_diffs_sig = _mean_diffs_sig[:].reshape(1,len(_mean_diffs_sig))
        else:
            mean_diffs_sig = np.append(mean_diffs_sig, _mean_diffs_sig.reshape(1,-1), axis=0)


        print(dr + "sig={}".format(np.mean(diffs_sig[:, 0])))

        for epoch in range(epochs):
            _dr_actual = [np.random.uniform(0, 1) ** 0.5 * max_noise, np.random.uniform(0, 2 * np.pi)]
            _dr1_actual = [np.random.uniform(0, 1) ** 0.5 * max_noise, np.random.uniform(0, 2 * np.pi)]
            _dr2_actual = [np.random.uniform(0, 1) ** 0.5 * max_noise, np.random.uniform(0, 2 * np.pi)]
            _dr3_actual = [np.random.uniform(0, 1) ** 0.5 * max_noise, np.random.uniform(0, 2 * np.pi)]
            dr_actual = vet_add(drs_unbaised[dr], _dr_actual)
            dr1_actual = vet_add(drs_unbaised[dr1], _dr1_actual)
            dr2_actual = vet_add(drs_unbaised[dr2], _dr2_actual)
            dr3_actual = vet_add(drs_unbaised[dr3], _dr3_actual)

            pt_buff_1 = np.array([dr1_actual, dr_actual, np.array([0, 0])])
            alpha1 = get_angle(pt_buff_1)
            pt_buff_2 = np.array([np.array([0, 0]), dr_actual, dr2_actual])
            alpha2 = get_angle(pt_buff_2)
            angle = [alpha1, alpha2, drs_unbaised[dr1][1], drs_unbaised[dr2][1]]
            dr_pre_1 = root(f_p2, np.array(drs_unbaised[dr]), args=(angle)).x

            pt_buff_1 = np.array([dr1_actual, dr_actual, np.array([0, 0])])
            alpha1 = get_angle(pt_buff_1)
            pt_buff_2 = np.array([np.array([0, 0]), dr_actual, dr3_actual])
            alpha2 = get_angle(pt_buff_2)
            angle = [alpha1, alpha2, drs_unbaised[dr1][1], drs_unbaised[dr3][1]]
            dr_pre_2 = root(f_p2, np.array(drs_unbaised[dr]), args=(angle)).x

            pt_buff_1 = np.array([dr2_actual, dr_actual, np.array([0, 0])])
            alpha1 = get_angle(pt_buff_1)
            pt_buff_2 = np.array([np.array([0, 0]), dr_actual, dr3_actual])
            alpha2 = get_angle(pt_buff_2)
            angle = [alpha1, alpha2, drs_unbaised[dr2][1], drs_unbaised[dr3][1]]
            dr_pre_3 = root(f_p2, np.array(drs_unbaised[dr]), args=(angle)).x

            dr_pre = vet_add(dr_pre_1, dr_pre_2)
            dr_pre = vet_add(dr_pre, dr_pre_3)
            dr_pre[0] /= 3
            diff = vet_min(dr_pre, dr_actual)
            diff = np.reshape(diff, (1, 2))
            diffs_mul = np.append(diffs_mul, diff, axis=0)
            if epoch % 10 == 0:
                _mean_diffs_mul = np.append(_mean_diffs_mul, np.mean(diffs_mul[:, 0]))
        if i == 0:
            mean_diffs_mul = _mean_diffs_mul[:].reshape(1,len(_mean_diffs_mul))
        else:
            mean_diffs_mul = np.append(mean_diffs_mul,_mean_diffs_mul.reshape(1,-1), axis=0)

        print(dr + "mul={}".format(np.mean(diffs_mul[:, 0])))
    fig1 = plt.figure()
    _mean_sig = np.mean(mean_diffs_sig,axis=0)
    _std_sig = np.std(mean_diffs_sig,axis=0)
    x=np.arange(stop=len(_mean_sig)*10,step=10,dtype=int)
    plt.plot(x, _mean_sig, label="双机定位")
    plt.fill_between(x,_mean_sig-_std_sig,_mean_sig+_std_sig
                     ,color='b', alpha=0.1)
    _mean_mul = np.mean(mean_diffs_mul, axis=0)
    _std_mul = np.std(mean_diffs_mul, axis=0)
    plt.plot(x, _mean_mul, label="三机定位")
    plt.fill_between(x, _mean_mul - _std_mul, _mean_mul + _std_mul
                     , color='r', alpha=0.1)
    plt.title("多次采样下单机定位与多机定位的平均误差")
    plt.xlabel("采样数x10")
    plt.ylabel("相对误差")
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig("./fig.png")
    plt.show()


