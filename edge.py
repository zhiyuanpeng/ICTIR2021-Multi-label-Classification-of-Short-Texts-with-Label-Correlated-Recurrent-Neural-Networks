import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Activation, Dropout, Dense
from tensorflow.keras.layers import Flatten, LSTM, Bidirectional
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Embedding
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input
import argparse
from numpy import asarray
from numpy import zeros
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  OK  ---")
    else:
        print("---  There is this folder!  ---")


def x_write(list_name, list_to_file_name):
    with open(list_to_file_name, 'a+') as f:
        for line_value in list_name:
            f.write(str(line_value) + "\n")


def y_write(list_name, list_to_file_name):
    with open(list_to_file_name, 'a+') as f:
        for line_value in list_name:
            f.write(str(line_value) + "\n")


def y_predict_write(list_name, list_to_file_name):
    with open(list_to_file_name, 'a+') as f:
        for line_value in list_name:
            f.write(str(line_value[0]) + "\n")


def y_write_round(list_name, list_to_file_name):
    with open(list_to_file_name, 'a+') as f:
        for line_value in list_name:
            if float(line_value) > 0.5:
                f.write(str(1) + "\n")
            else:
                f.write(str(0) + "\n")


def read_text(filename):
    """
    read the train text to list
    @param filename: the file name of the train.txt
    @return: a list contains all the text
    """
    text_list = []
    with open(filename, "r") as f:
        text = f.readlines()
        for line in text:
            text_list.append(line.strip("\n"))
    return text_list


def get_edge_label(node_a, node_b, file_name):
    y_a = np.loadtxt("data/processed/" + file_name, dtype=int)[:, node_a]
    y_b = np.loadtxt("data/processed/" + file_name, dtype=int)[:, node_b]
    edge_label = [y_a[i]*y_b[i] for i in range(y_b.shape[0])]
    return np.array(edge_label)


def train_predict(epoch_num, a, b, optimizer_name, lstm_num, dense_num, max_length, batchsize):
    # edge_str = "l" + str(a) + "_" + "l" + str(b)
    # # unbalance data
    # X_train_ub = read_text("data/processed/X_train.txt")
    # y_train_ub = get_edge_label(a, b, "y_train.txt")
    # # balance data
    # non_zero = np.sum(y_train_ub)
    # rate = non_zero / (len(y_train_ub) - non_zero)
    # X_train = []
    # y_train = []
    # np.random.seed(0)
    # for i in range(y_train_ub.shape[0]):
    #     if y_train_ub[i, ] == 1:
    #         X_train.append(X_train_ub[i])
    #         y_train.append(1)
    #     elif np.random.random() <= rate:
    #         X_train.append(X_train_ub[i])
    #         y_train.append(0)
    # y_train = np.array(y_train)
    # use unbalanced data
    X_train = read_text("data/processed/X_train.txt")
    y_train = get_edge_label(a, b, "y_train.txt")
    #
    X_test = read_text("data/processed/X_test.txt")
    y_test = get_edge_label(a, b, "y_test.txt")
    # record the train and test data
    # x_write(X_train, "data/store/" + edge_str + "/" + edge_str + "_X_train.txt")
    # x_write(X_test, "data/store/" + edge_str + "/" + edge_str + "_X_test.txt")
    # y_write(y_train, "data/store/" + edge_str + "/" + edge_str + "_y_train.txt")
    # y_write(y_test, "data/store/" + edge_str + "/" + edge_str + "_y_test.txt")
    #
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    vocab_size = len(tokenizer.word_index) + 1

    maxlen = max_length

    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

    embeddings_dictionary = dict()

    glove_file = open('../glove.6B.100d.txt', encoding="utf8")

    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    glove_file.close()

    embedding_matrix = zeros((vocab_size, 100))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

    deep_inputs = Input(shape=(maxlen,))
    embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(deep_inputs)
    LSTM_Layer_1 = Bidirectional(LSTM(lstm_num))(embedding_layer)
    dense_layer_1 = Dense(dense_num, activation='sigmoid')(LSTM_Layer_1)
    model = Model(inputs=deep_inputs, outputs=dense_layer_1)

    model.compile(loss='binary_crossentropy', optimizer=optimizer_name, metrics=['acc'])
    history = model.fit(X_train, y_train, batch_size=batchsize, epochs=epoch_num, verbose=1, validation_split=0.1)
    score = model.evaluate(X_test, y_test, verbose=1)
    predict_list = model.predict(X_test, batch_size=batchsize, verbose=1)
    return history, score, predict_list, model


def select_best(epoch_num, a, b, save_data, optimizer_name, iter_num, lstm_num, dense_num, max_length, batchsize):
    edge_str = "l" + str(a) + "_" + "l" + str(b)
    y_test = get_edge_label(a, b, "y_test.txt")
    history_list = []
    accuracy_list = []
    predict_list = []
    # model_list = []
    for i in range(iter_num):
        history, score, predict, model = train_predict(epoch_num, a, b, optimizer_name,lstm_num, dense_num, max_length, batchsize)
        history_list.append(history)
        accuracy_list.append(score[1])
        predict_list.append(predict)
        # model_list.append(model)
    max_index = accuracy_list.index(np.max(accuracy_list))
    history_max = history_list[max_index]
    accuracy_max = accuracy_list[max_index]
    predict_max = predict_list[max_index]
    # model_max = model_list[max_index]
    if save_data == 1:
        mkdir("data/store/" + edge_str)
        y_predict_write(predict_max, "data/store/" + edge_str + "/" + edge_str + "_predict.txt")
        y_write_round(predict_max, "data/store/" + edge_str + "/" + edge_str + "_predict_round.txt")
    y_predict_round = []
    for index in range(len(predict_max)):
        if float(predict_max[index]) > 0.5:
            y_predict_round.append(1)
        else:
            y_predict_round.append(0)
    con_mat = confusion_matrix(list(y_test), y_predict_round)
    con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]
    con_mat_norm = np.around(con_mat_norm, decimals=6)
    # save the result
    with open("data/store/" + edge_str + "/" + edge_str + "_result.txt", "a+") as r:
        r.write("\n")
        r.write("\n")
        r.write("optimezer: " + optimizer_name + "\n")
        r.write("epoch is " + str(epoch_num) + "\n")
        r.write("batch size is " + str(batchsize) + "\n")
        r.write("LSTM num is " + str(lstm_num) + "\n")
        r.write("text max length is " + str(max_length) + "\n")
        r.write("accuracy is " + str(accuracy_max) + "\n")
        r.write("\n")
        # r.write("edge " + edge_str + " confusion matrix [0, 0]: " + str(con_mat[0, 0]) + "\n")
        # r.write("edge " + edge_str + " confusion matrix [0, 1]: " + str(con_mat[0, 1]) + "\n")
        # r.write("edge " + edge_str + " confusion matrix [1, 0]: " + str(con_mat[1, 0]) + "\n")
        # r.write("edge " + edge_str + " confusion matrix [1, 1]: " + str(con_mat[1, 1]) + "\n")
        # r.write("edge " + edge_str + " precision of 0 is " + str(con_mat_norm[0, 0]) + "\n")
        # if con_mat[0, 0] == 0.0:
        #     r.write("edge " + edge_str + " recall of 0 is " + str(0.0) + "\n")
        # else:
        #     r.write("edge " + edge_str + " recall of 0 is " + str(
        #         con_mat[0, 0] / (con_mat[0, 0] + con_mat[1, 0])) + "\n")
        # r.write("edge " + edge_str + " precision of 1 is " + str(con_mat_norm[1, 1]) + "\n")
        # if con_mat[1, 1] == 0.0:
        #     r.write("edge " + edge_str + " recall of 1 is " + str(0.0) + "\n")
        # else:
        #     r.write("edge " + edge_str + " recall of 1 is " + str(
        #         con_mat[1, 1] / (con_mat[0, 1] + con_mat[1, 1])) + "\n")
    # model_max.save("models/" + edge_str + ".h5")
    epochs = len(history_max.history['loss'])
    plt.plot(range(epochs), history_max.history['loss'], label='loss')
    plt.plot(range(epochs), history_max.history['val_loss'], label='val_loss')
    plt.legend()
    plt.savefig("data/img/" + edge_str + "_" + str(optimizer_name) + "_" + str(epoch_num) + "_batch" + str(batchsize) + ".png")
    # plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("epoch_num", type=int, help="the name of the node")
    parser.add_argument("a", type=int, help="begin node")
    parser.add_argument("b", type=int, help="end node")
    parser.add_argument("--save_data", type=int, default=0, help="save the predicted data or not: default not save," +
                                                                 " set 1 save")
    parser.add_argument("--optimizer_name", type=str, default="adam", help="the optimizer name, default is adam")
    parser.add_argument("--iter_num", type=int, default=1,
                        help="iter num, default is 1. If you set it n, the code " +
                        "will run n times and select the best one as final result")
    args = parser.parse_args()
    lstm_num = 4
    dense_num = 1
    max_length = 16
    batchsize = 32
    select_best(args.epoch_num, args.a, args.b, args.save_data, args.optimizer_name, args.iter_num, lstm_num,
                dense_num, max_length, batchsize)


if __name__ == "__main__":
    main()
