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


def split_data(file_name, label_name, label_num):
    """
    drop the labels not in label_name and drop the data with len(label) less than 2
    :param file_name: the input the file name
    :param label_name: the top k label name
    :return:
    """
    #
    random.seed(0)
    threshold = 0.1
    #
    meta_data = pd.read_csv(file_name)
    X_test = []
    X_train = []
    y_train = []
    y_test = []
    y_test_text = []
    y_train_text = []
    for index, row in tqdm(meta_data.iterrows()):
        # drop the data with num of labels less than two
        tags = str(row["Tags_clean"]).split(" ")
        tags_clean = []
        tags_clean_binary = [0 for i in range(len(label_name))]
        for tag in tags:
            if tag in label_name:
                tags_clean.append(tag)
                tags_clean_binary[label_name.index(tag)] = 1
        if len(tags_clean) >= label_num:
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
    # write to processed/archive
    x_write(X_test, "data/processed/archive/X_test_" + str(label_num) + ".txt")
    x_write(X_train, "data/processed/archive/X_train_" + str(label_num) + ".txt")
    y_write(y_train, "data/processed/archive/y_train_" + str(label_num) + ".txt")
    y_write(y_test, "data/processed/archive/y_test_" + str(label_num) + ".txt")
    return y_train, new_label_name


def remove_duplicate(train_file, test_file):
    """
    drop the duplicates and process the text
    :param train_file:
    :param test_file:
    :return: no repeat clean file
    """
    # remove duplicates
    train_meta = pd.read_csv(train_file)
    test_meta = pd.read_csv(test_file)
    df = train_meta.append(test_meta, ignore_index=True)
    # shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    duplicate_pairs = df.duplicated('Title')
    print("Total number of duplicate questions : ", duplicate_pairs.sum())
    df = df[~duplicate_pairs]
    print("Dataframe shape after duplicate removal : ", df.shape)
    # process the tags
    df["Tags_clean"] = df["Tags"].apply(lambda x: tag_process(x))
    df["Title_clean"] = df["Title"].apply(lambda x: preprocess_text(x))
    df_clean = df.drop(columns=["Title", "Tags", "Body"])
    df_clean.to_csv("../Datasets/Stack/Total.csv")
    print("write done")
    return df_clean


def get_frequent_labels(df, topk):
    """
    return the top frequent labels
    :param df: clean no repeat data
    :param topk: top k frequent labels
    :return: top 20 frequent labels
    """
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split())
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
    plt.show()
    return tag_df_sorted["Tags"][:topk].values.tolist()


# mkdir necessary directories
mkdir("data/enumeration")
mkdir("data/img")
mkdir("data/processed")
mkdir("data/store")
mkdir("data/store/original")
mkdir("data/treeimg")
#
pd.set_option('display.max_colwidth', 300)
label_index = [str(i) for i in range(6)]
y_train = np.loadtxt("data/processed/y_train.txt")
tree_node_2, tree_graph_2, tree_edge_2 = gt.get_tree(y_train, label_index, 0, 6, "pearson")
layout_2 = tree_graph_2.layout_lgl()
ig.drawing.plot(tree_graph_2, "data/treeimg/tree_pearson.png", layout=layout_2, bbox=(850, 850), margin=(80, 80, 80, 80))
# mkdir data/store/edges
for edge in tree_edge_2:
    edge_dir = "data/store/" + "l" + str(edge[0]) + "_l" + str(edge[1])
    mkdir(edge_dir)
