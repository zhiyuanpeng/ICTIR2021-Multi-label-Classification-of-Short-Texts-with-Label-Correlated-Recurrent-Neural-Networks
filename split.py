import numpy as np
import random
import os

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  OK  ---")
    else:
        print("---  There is this folder!  ---")


def x_write(list_name, list_to_file_name):
    """
    write a list of test value to file
    :param list_name:
    :param list_to_file_name:
    :return: no
    """
    with open(list_to_file_name, 'w') as f:
        for line_value in list_name:
            f.write(str(line_value) + "\n")


def y_write(list_name, list_to_file_name):
    """
    write the test result matrix to file
    :param list_name:
    :param list_to_file_name:
    :return: no
    """
    with open(list_to_file_name, 'w') as f:
        for line_value in list_name:
            s = ""
            for i in range(len(line_value)):
                if i != len(line_value) - 1:
                    s += str(int(line_value[i]))
                    s += " "
                else:
                    s += str(int(line_value[i]))
                    s += "\n"
            f.write(s)


def y_write_float(list_name, list_to_file_name):
    """
    wirte the float test result matrix to file
    :param list_name:
    :param list_to_file_name:
    :return: no
    """
    with open(list_to_file_name, 'w') as f:
        for line_value in list_name:
            s = ""
            for i in range(len(line_value)):
                if i != len(line_value) - 1:
                    s += str(line_value[i])
                    s += " "
                else:
                    s += str(line_value[i])
                    s += "\n"
            f.write(s)


def load_data(edge_list):
    """
    load text
    :param edge_list: a list of the edges
    :return:
    edge_predict a list of matrix according to the edge_list
    """
    # convert the input list to string list
    edge_str = ["l" + str(i[0]) + "_" + "l" + str(i[1]) for i in edge_list]
    edge_predict = []
    edge_predict_round = []
    edge_truth = []
    # for edge
    for edge in edge_str:
        edge_path = "data/store/" + edge + "/" + edge
        edge_predict.append(np.loadtxt(edge_path + "_predict.txt"))
        edge_predict_round.append(np.loadtxt(edge_path + "_predict_round.txt", dtype=int))
        # for edge truth
        edge_truth_path = "data/split_test/" + edge + "_truth.txt"
        edge_truth.append(np.loadtxt(edge_truth_path, dtype=int))
    return edge_predict, edge_predict_round, edge_truth


def write_edge_truth(edge_list, y_test):
    """
    write the true edge info to split
    :param edge_list: a list of edge
    :param y_test: the result of the test
    :return: no
    """
    for edge in edge_list:
        edge_name = "l" + str(edge[0]) + "_" + "l" + str(edge[1])
        true_edge = [y_test[i, edge[0]] * y_test[i, edge[1]] for i in range(y_test.shape[0])]
        true_edge_path = "data/split_test/" + edge_name + "_truth.txt"
        with open(true_edge_path, "a+") as f:
            for value in true_edge:
                f.write(str(value) + "\n")


def split_data(edge_list, edge_predict, edge_predict_round, edge_truth,
               y_test, original_predict, original_predict_round, total_num, adjust_num):
    """
    split data to adjust and left randomly
    :param edge_list:
    :param edge_predict:
    :param edge_predict_round:
    :param edge_truth:
    :param y_test:
    :param original_predict:
    :param original_predict_round:
    :param total_num:
    :param adjust_num:
    :return: the split result
    """
    random.seed(0)
    sample_index = random.sample(range(total_num), adjust_num)
    left_num = total_num - adjust_num
    #
    edge_str = ["l" + str(i[0]) + "_" + "l" + str(i[1]) for i in edge_list]
    # for edge
    for edge_index in range(len(edge_list)):
        edge_predict_adjust = []
        edge_predict_adjust_round = []
        edge_predict_left = []
        edge_predict_left_round = []
        edge_truth_adjust = []
        edge_truth_left = []
        for index in range(y_test.shape[0]):
            if index in sample_index:
                edge_predict_adjust.append(edge_predict[edge_index][index])
                edge_predict_adjust_round.append(edge_predict_round[edge_index][index])
                edge_truth_adjust.append(edge_truth[edge_index][index])
            else:
                edge_predict_left.append(edge_predict[edge_index][index])
                edge_predict_left_round.append(edge_predict_round[edge_index][index])
                edge_truth_left.append(edge_truth[edge_index][index])
        x_write(edge_predict_adjust, "data/split_test/" + edge_str[edge_index] + "_predict_" + str(adjust_num) + ".txt")
        x_write(edge_predict_adjust_round, "data/split_test/" + edge_str[edge_index] + "_predict_" + str(adjust_num) + "_round.txt")
        x_write(edge_predict_left, "data/split_test/" + edge_str[edge_index] + "_predict_" + str(left_num) + ".txt")
        x_write(edge_predict_left_round, "data/split_test/" + edge_str[edge_index] + "_predict_" + str(left_num) + "_round.txt")
        x_write(edge_truth_adjust, "data/split_test/" + edge_str[edge_index] + "_truth_" + str(adjust_num) + ".txt")
        x_write(edge_truth_left, "data/split_test/" + edge_str[edge_index] + "_truth_" + str(left_num) + ".txt")
    # split y_test, original_predict, original_predict_round
    y_test_adjust = []
    original_predict_adjust = []
    original_predict_adjust_round = []
    y_test_left = []
    original_predict_left = []
    original_predict_left_round = []
    for index in range(y_test.shape[0]):
        if index in sample_index:
            y_test_adjust.append(y_test[index])
            original_predict_adjust.append(original_predict[index])
            original_predict_adjust_round.append(original_predict_round[index])
        else:
            y_test_left.append(y_test[index])
            original_predict_left.append(original_predict[index])
            original_predict_left_round.append(original_predict_round[index])
    y_write(y_test_adjust, "data/split_test/y_test_" + str(adjust_num) + ".txt")
    y_write(y_test_left, "data/split_test/y_test_" + str(left_num) + ".txt")
    y_write_float(original_predict_adjust, "data/split_test/original_predict_" + str(adjust_num) + ".txt")
    y_write_float(original_predict_left, "data/split_test/original_predict_" + str(left_num) + ".txt")
    y_write(original_predict_adjust_round, "data/split_test/original_predict_" + str(adjust_num) + "_round.txt")
    y_write(original_predict_left_round, "data/split_test/original_predict_" + str(left_num) + "_round.txt")


def prepare_test_data(edge_list, total_num, adjust_num):
    mkdir("data/split_test")
    y_test = np.loadtxt("data/processed/y_test.txt", dtype=int)
    original_predict = np.loadtxt("data/store/original/original_predict.txt")
    original_predict_round = np.loadtxt("data/store/original/original_predict_round.txt")
    write_edge_truth(edge_list, y_test)
    edge_predict, edge_predict_round, edge_truth = load_data(edge_list)
    split_data(edge_list, edge_predict, edge_predict_round, edge_truth,
               y_test, original_predict, original_predict_round, total_num, adjust_num)



