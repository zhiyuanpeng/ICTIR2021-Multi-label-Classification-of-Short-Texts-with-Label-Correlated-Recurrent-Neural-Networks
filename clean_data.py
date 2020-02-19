import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import get_tree as gt
import igraph as ig
import os
from sklearn.feature_extraction.text import CountVectorizer
import random
from tqdm import tqdm
import argparse

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  OK  ---")
    else:
        print("---  There is this folder!  ---")


# clear text
def preprocess_text(sen):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', str(sen))
    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence.lower()


def tag_process(tag):
    # Remove punctuations and numbers
    # sentence = re.sub('[^a-zA-Z]', ' ', str(tag))
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', str(tag))
    return sentence.lower()


def x_write(list_name, list_to_file_name):
    with open(list_to_file_name, 'a+') as f:
        for line_value in list_name:
            f.write(str(line_value) + "\n")


def y_write(list_name, list_to_file_name):
    with open(list_to_file_name, 'a+') as f:
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


def show_X_train_info(X_train):
    """
    show the min max and avg length of the text
    :param y_train:
    :return:
    """
    length = []
    for tweet in X_train:
        length.append(len(tweet.split(" ")))
    max_len = np.max(length)
    min_len = np.min(length)
    mean_len = np.mean(length)
    print("the min length of train text is: ")
    print(min_len)
    print("\n")

    print("the mean length of train text is: ")
    print(mean_len)
    print("\n")

    print("the max length of train text is: ")
    print(max_len)
    print("\n")


def split_data(meta_data, label_name, min_label_num):
    """
    drop the labels not in label_name and drop the data with len(label) less than 2
    :param meta_data: the input the pd data
    :param label_name: the top k label name
    :param min_label_num: minimum number of labels per instance
    :return:
    """
    #
    random.seed(0)
    threshold = 0.1
    #
    X_test = []
    X_train = []
    y_train = []
    y_test = []
    y_test_text = []
    y_train_text = []
    for index, row in tqdm(meta_data.iterrows()):
        # drop the data with num of labels less than two
        tags = str(row["Tags_clean"]).split(",")
        tags_clean = []
        tags_clean_binary = [0 for i in range(len(label_name))]
        for tag in tags:
            if tag in label_name:
                tags_clean.append(tag)
                tags_clean_binary[label_name.index(tag)] = 1
        if len(tags_clean) >= min_label_num:
            if random.random() > threshold:
                X_train.append(row["Title_clean"])
                y_train.append(tags_clean_binary)
                y_train_text.append(tags_clean)
            else:
                X_test.append(row["Title_clean"])
                y_test.append(tags_clean_binary)
                y_test_text.append(tags_clean)
    # delete the all 0s columns
    new_label_name = []
    label_sum = list(np.sum(y_train, axis=0))
    for i in range(len(label_sum)):
        if label_sum[i] == 0:
            for row in y_train:
                del row[i]
            for row in y_test:
                del row[i]
        else:
            new_label_name.append(label_name[i])
    # print the info of train text
    show_X_train_info(X_train)
    # write to processed/archive
    x_write(X_test, "data/processed/X_test" + ".txt")
    x_write(X_train, "data/processed/X_train" + ".txt")
    y_write(y_train, "data/processed/y_train" + ".txt")
    y_write(y_test, "data/processed/y_test" + ".txt")
    return y_train, new_label_name


def remove_duplicate(file_name):
    """
    drop the duplicates and process the text
    :param file_name:
    :return: no repeat clean file
    """
    # remove duplicates
    df = pd.read_csv(file_name + ".csv", sep="\t", names=["Title", "Tags"])
    duplicate_pairs = df.duplicated('Title')
    print("Total number of duplicate questions : ", duplicate_pairs.sum())
    df = df[~duplicate_pairs]
    print("Dataframe shape after duplicate removal : ", df.shape)
    # process the tags
    df["Title_clean"] = df["Title"].apply(lambda x: preprocess_text(x))
    df["Tags_clean"] = df["Tags"].apply(lambda x: tag_process(x))
    df_clean = df.drop(columns=["Title", "Tags"])
    df_clean.to_csv(file_name + "_no_repeat" + ".csv")
    print("write done")
    return df_clean


def get_frequent_labels(df, topk):
    """
    return the top frequent labels
    :param df: clean no repeat data
    :param topk: top k frequent labels
    :return: top 20 frequent labels
    """
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(","))
    tag_bow = vectorizer.fit_transform(df['Tags_clean'].values.astype(str))
    # show info
    print("Number of questions :", tag_bow.shape[0])
    print("Number of unique tags :", tag_bow.shape[1])
    tags = vectorizer.get_feature_names()
    freq = tag_bow.sum(axis=0).A1
    tag_to_count_map = dict(zip(tags, freq))
    list = []
    for key, value in tag_to_count_map.items():
        list.append([key, value])
    tag_df = pd.DataFrame(list, columns=['Tags', 'Counts'])
    tag_df_sorted = tag_df.sort_values(['Counts'], ascending=False)
    i = np.arange(topk)
    tag_df_sorted.head(topk).plot(kind='bar')
    plt.title('Frequency of top k tags')
    plt.xticks(i, tag_df_sorted['Tags'])
    plt.xlabel('Tags')
    plt.ylabel('Counts')
    plt.savefig("data/img/topk.png")
    return tag_df_sorted["Tags"][:topk].values.tolist()


def show_tree(y_train, label_name, threshold, steps, score_method):
    """
    show the tree structure
    :param y_train: the y training data
    :param label_name: the label name to be shown in the tree img
    :param threshold: edge weight bigger than the threshold will be kept
    :param steps: the steps max-spanning tree algorithm working
    :param score_method: method to calculate the correlations
    :return: no
    """
    # use the true name of the labels
    tree_node, tree_graph, tree_edge = gt.get_tree(y_train, label_name, threshold, steps, score_method)
    layout = tree_graph.layout_lgl()
    ig.drawing.plot(tree_graph, "data/tree_img/tree_" + str(score_method) + ".png", layout=layout, bbox=(850, 850),
                    margin=(80, 80, 80, 80))
    # use the int number to represent the labels
    label_name_digit = [str(i) for i in range(len(label_name))]
    tree_node_digit, tree_graph_digit, tree_edge_digit = gt.get_tree(y_train, label_name_digit, threshold, steps, score_method)
    layout_digit = tree_graph_digit.layout_lgl()
    ig.drawing.plot(tree_graph_digit, "data/tree_img/tree_" + str(score_method) + "_digit.png", layout=layout_digit, bbox=(850, 850),
                    margin=(80, 80, 80, 80))
    # mkdir data/store/edges
    for edge in tree_edge_digit:
        edge_dir = "data/store/" + "l" + str(edge[0]) + "_l" + str(edge[1])
        mkdir(edge_dir)


def mkdir_necessary():
    """
    mkdir the necessary folder
    :return: no
    """
    # mkdir necessary directories
    mkdir("data")
    mkdir("data/processed")
    mkdir("data/tree_img")
    mkdir("data/img")
    mkdir("data/store")


def show_y_train_info(y_train):
    row_distribution = []
    for i in range(len(y_train)):
        row_distribution.append(np.sum(y_train[i]))
    row_mean = np.mean(row_distribution)
    print("the avg number of the labels per instance in the train is: ")
    print(row_mean)
    print("\n")


def data_clean(file_name, topk, min_avg_labels, correlation_method, threshold):
    """
    clean the data and generate the tree structure
    :param file_name:
    :param topk:
    :param min_avg_labels:
    :param correlation_method:
    :param threshold:
    :return:
    """
    # mkdir the necessary folders
    mkdir_necessary()
    # remove the duplicate text
    df_no_repeat = remove_duplicate(file_name)
    # get the topk frequent labels
    top_labels = get_frequent_labels(df_no_repeat, topk)
    # the len(new_label_name) may be less than topk because all 0s column will be delected
    y_train, new_label_name = split_data(df_no_repeat, top_labels, min_avg_labels)
    show_y_train_info(y_train)
    show_tree(y_train, new_label_name, threshold, len(new_label_name), correlation_method)


def main():

    # command-line parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", type=str, help="the file name of the csv file, no .csv is needed.")
    parser.add_argument("topk", type=int,
                        help="the number of the top frequent labels needed to be kept")
    parser.add_argument("min_avg_labels", type=int,
                        help="the minimum number of labels per instance should have. " +
                             "instance with less than minimum num of labels will be delete")
    parser.add_argument("--correlation_method", type=str, default="default", choices=["default", "cosine", "pearson"],
                        help="three ways to calculate the correlation: default, cosine, pearson")
    parser.add_argument("--threshold", type=int, default=0,
                        help="three ways to calculate the correlation: default, cosine, pearson")
    args = parser.parse_args()

    data_clean(args.file_name, args.topk, args.min_avg_labels, args.correlation_method, args.threshold)


if __name__ == "__main__":
    main()
