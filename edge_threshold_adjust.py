import numpy as np


def x_write(list_name, list_to_file_name):
    with open(list_to_file_name, 'a+') as f:
        for line_value in list_name:
            f.write(str(line_value) + "\n")


def read_to_list_threshold(file_name, threshold, left_num):
    list_threshold = []
    with open("./data/split_test/" + file_name + "_predict_" + str(left_num) + ".txt", "r") as f:
        without_round = f.readlines()
        for value in without_round:
            if float(value) > threshold:
                list_threshold.append(1)
            else:
                list_threshold.append(0)
    return list_threshold


def round_with_threshold(file_name, threshold, left_num):
    round_list = read_to_list_threshold(file_name, threshold, left_num)
    x_write(round_list, "./data/split_test/" + file_name + "_predict_round_" + str(threshold) + "_" + str(left_num) + ".txt")


def adjust_threshold(file_dire, threshold, adjust_num):
    """
    if the predict value > threshold, value = 1
    :param file_dire:
    :param threshold:
    :param adjust_num:
    :return: the predict_round and right_sum
    """
    predict = np.loadtxt("./data/split_test/" + file_dire + "_predict_" + str(adjust_num) + ".txt")
    label = np.loadtxt("./data/split_test/" + file_dire + "_truth_" + str(adjust_num) + ".txt", dtype=int)
    right_sum = 0
    predict_round = []
    for i in range(predict.shape[0]):
        if predict[i, ] > threshold:
            after_round = 1
        else:
            after_round = 0
        predict_round.append(after_round)
        if after_round == label[i, ]:
            right_sum += 1
    return predict_round, right_sum/predict.shape[0]


def optimal_search(file_name, adjust_num):
    """
    find the maximum right_sum and the corresponding threshold, predict result
    :param file_name:
    :param adjust_num:
    :return:
    """
    threshold = 0.5
    predict_list = []
    accuracy_list = []
    threshold_list = []
    for index in range(50):
        threshold += 0.01
        predict_round, accuracy = adjust_threshold(file_name, threshold, adjust_num)
        predict_list.append(predict_round)
        accuracy_list.append(accuracy)
        threshold_list.append(threshold)
    max_index = accuracy_list.index(np.max(accuracy_list))
    return predict_list[max_index], accuracy_list[max_index], round(threshold_list[max_index], 2)


def return_file_list(edge_list, total_num, adjust_num):
    """
    return the file name after optimization
    :param edge_list:
    :param total_num:
    :param adjust_num:
    :return: a list of optimizated file name
    """
    left_num = total_num - adjust_num
    file_list = []
    for edge in edge_list:
        # get the optimal
        edge_name = "l" + str(edge[0]) + "_" + "l" + str(edge[1])
        opt_predict, opt_accracy, opt_threshold = optimal_search(edge_name, adjust_num)
        # use the optimal threshold to deal the left data
        round_with_threshold(edge_name, opt_threshold, left_num)
        # for each edge, we define a blank file name
        file_name = edge_name + "_" + "predict_round_" + str(opt_threshold) + "_" + str(left_num) + ".txt"
        file_list.append(file_name)
    return file_list


# def main():
#
#
#
# if __name__ == '__main__':
#     main()


