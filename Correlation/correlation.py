from scipy.stats import kendalltau
import numpy as np
import xlrd

def row2array(row, length):
    row = np.array(row)
    arr = np.array([])

    for i in range(1, length, 1):
        arr = np.append(arr, float(row[i]))

    return arr, row[0]


def func(a, b):
    Lens = len(a)

    ties_onlyin_x = 0
    ties_onlyin_y = 0
    con_pair = 0
    dis_pair = 0
    for i in range(Lens - 1):
        for j in range(i + 1, Lens):
            test_tying_x = np.sign(a[i] - a[j])
            test_tying_y = np.sign(b[i] - b[j])
            panduan = test_tying_x * test_tying_y
            if panduan == 1:
                con_pair += 1
            elif panduan == -1:
                dis_pair += 1

            if test_tying_y == 0 and test_tying_x != 0:
                ties_onlyin_y += 1
            elif test_tying_x == 0 and test_tying_y != 0:
                ties_onlyin_x += 1

    if (con_pair + dis_pair + ties_onlyin_x) * (dis_pair + con_pair + ties_onlyin_y) == 0:
        k = 10**-1
    else:
        k = (con_pair + dis_pair + ties_onlyin_x) * (dis_pair + con_pair + ties_onlyin_y)

    Kendallta1 = (con_pair - dis_pair) / np.sqrt(k)

    return Kendallta1

if __name__ == '__main__':
    excel = xlrd.open_workbook('correlation.xlsx')

    ## MNIST's related values are in sheet 0
    mnist_data = excel.sheet_by_index(0)

    print(mnist_data.name, mnist_data.nrows, mnist_data.ncols)

    ## get every row's value and put them in a dictionary
    mnist_dic = {}

    for j in range(2, mnist_data.nrows):
        row_excel = mnist_data.row_values(j)
        row_value, name = row2array(row_excel, mnist_data.ncols)
        mnist_dic[name] = row_value

    print(mnist_dic)

    # calculate Kendallta
    for metric_1 in np.array(mnist_data.col_values(0))[2:13]:
        for metric_2 in np.array(mnist_data.col_values(0))[13:23]:
            Kendallta, p_value = kendalltau(mnist_dic[metric_1], mnist_dic[metric_2])
            v = func(mnist_dic[metric_2], mnist_dic[metric_1])
        #     print(v)
        # print('---------------------------------------------')
            print('The Kendallta between {} and {} is {}'.format(metric_2, metric_1, v))






