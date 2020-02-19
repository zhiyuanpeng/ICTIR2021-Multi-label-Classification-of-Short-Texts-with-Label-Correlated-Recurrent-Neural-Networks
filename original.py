import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Activation, Dropout, Dense
from tensorflow.keras.layers import Flatten, LSTM
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Embedding
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input

import argparse
from numpy import asarray
from numpy import zeros
import numpy as np

import os

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# use gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  OK  ---")
    else:
        print("---  There is this folder!  ---")


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


def x_write(list_name, list_to_file_name):
    with open(list_to_file_name, 'a+') as f:
        for line_value in list_name:
            f.write(str(line_value) + "\n")


def y_write(list_name, list_to_file_name):
    with open(list_to_file_name, 'a+') as f:
        for row in range(list_name.shape[0]):
            s = ""
            for column in range(list_name.shape[1]):
                s += str(float(list_name[row, column]))
                if column != (list_name.shape[1] - 1):
                    s += " "
            f.write(s + "\n")


def y_write_round(list_name, list_to_file_name):
    with open(list_to_file_name, 'a+') as f:
        for row in range(list_name.shape[0]):
            s = ""
            for column in range(list_name.shape[1]):
                if float(list_name[row, column]) > 0.5:
                    value = "1"
                else:
                    value = "0"
                s += value
                if column != (list_name.shape[1] - 1):
                    s += " "
            f.write(s + "\n")


def train_predict(epoch_num, save_data, optimizer_name, lstm_num, dense_num, max_length, batchsize):
    X_train = read_text("data/processed/X_train.txt")
    X_test = read_text("data/processed/X_test.txt")
    y_train = np.loadtxt("data/processed/y_train.txt", dtype=int)
    y_test = np.loadtxt("data/processed/y_test.txt", dtype=int)
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    check_max_len = 0
    for i in range(len(X_train)):
        if len(X_train[i]) > check_max_len:
            check_max_len = len(X_train[i])

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
    LSTM_Layer_1 = LSTM(lstm_num)(embedding_layer)
    dense_layer_1 = Dense(dense_num, activation='sigmoid')(LSTM_Layer_1)
    model = Model(inputs=deep_inputs, outputs=dense_layer_1)

    model.compile(loss='binary_crossentropy', optimizer=optimizer_name, metrics=['acc'])

    history = model.fit(X_train, y_train, batch_size=batchsize, epochs=epoch_num, verbose=1, validation_split=0.2)
    score = model.evaluate(X_test, y_test, verbose=1)
    predict_list = model.predict(X_test, batch_size=batchsize, verbose=1)
    if save_data == 1:
        y_write(predict_list, "data/store/original/original_predict.txt")
        y_write_round(predict_list, "data/store/original/original_predict_round.txt")
    right_sum_1 = 0
    right_sum_2 = 0
    y_predict_round = np.zeros_like(predict_list, dtype=int)
    for i in range(predict_list.shape[0]):
        for j in range(predict_list.shape[1]):
            if float(predict_list[i, j]) > 0.5:
                y_predict_round[i, j] = 1
            else:
                y_predict_round[i, j] = 0
            if (y_predict_round[i, j]) == int(y_test[i, j]):
                right_sum_1 += 1
        if (y_predict_round[i, :] == [int(y_test[i, index]) for index in range(predict_list.shape[1])]).all():
            right_sum_2 += 1
    accuracy_row = (right_sum_2 / predict_list.shape[0])
    accuracy_total = (right_sum_1 / (predict_list.shape[0] * predict_list.shape[1]))

    # save the result
    mkdir("data/store/original")
    with open("data/store/original/original_result.txt", "a+") as r:
        r.write("epoch is " + str(epoch_num) + "\n")
        r.write("optimizer is: " + optimizer_name + "\n")
        r.write("batch size is " + str(batchsize) + "\n")
        r.write("LSTM num is " + str(lstm_num) + "\n")
        r.write("text max length is " + str(max_length) + "\n")
        r.write("self total accuracy is " + str(accuracy_total) + "\n")
        r.write("self row accuracy is " + str(accuracy_row) + "\n")
        r.write("score is " + str(score[0]) + "\n")
        r.write("accuracy by model_predict is " + str(score[1]) + "\n")
        r.write("total number of training data is " + str(y_train.shape[0]) + "\n")
        for i in range(y_test.shape[1]):
            # get confusion matrix
            con_mat = confusion_matrix(list(y_test[:, i]), list(y_predict_round[:, i]))
            con_mat_norm = np.round(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], 4)
            r.write("\n")
            r.write("node l" + str(i) + " confusion matrix [0, 0]: " + str(con_mat[0, 0]) + "\n")
            r.write("node l" + str(i) + " confusion matrix [0, 1]: " + str(con_mat[0, 1]) + "\n")
            r.write("node l" + str(i) + " confusion matrix [1, 0]: " + str(con_mat[1, 0]) + "\n")
            r.write("node l" + str(i) + " confusion matrix [1, 1]: " + str(con_mat[1, 1]) + "\n")
            r.write("node l" + str(i) + " precision of 0 is " + str(con_mat_norm[0, 0]) + "\n")
            if con_mat[0, 0] == 0.0:
                r.write("node l" + str(i) + " recall of 0 is " + str(0.0) + "\n")
            else:
                r.write("node l" + str(i) + " recall of 0 is " + str(
                    con_mat[0, 0] / (con_mat[0, 0] + con_mat[1, 0])) + "\n")
            r.write("node l" + str(i) + " precision of 1 is " + str(con_mat_norm[1, 1]) + "\n")
            if con_mat[1, 1] == 0.0:
                r.write("node l" + str(i) + " recall of 1 is " + str(0.0) + "\n")
            else:
                r.write("node l" + str(i) + " recall of 1 is " + str(
                    con_mat[1, 1] / (con_mat[0, 1] + con_mat[1, 1])) + "\n")

    # model.save("models/original.h5")
    epochs = len(history.history['loss'])
    plt.plot(range(epochs), history.history['loss'], label='loss')
    plt.plot(range(epochs), history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.savefig("data/img/original_" + optimizer_name + "_" + str(epoch_num) + "_batch" + str(batchsize) + ".png")
    # plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("epoch_num", type=int, help="the number of the epochs")
    parser.add_argument("--save_data", type=int, default=0, help="save the predicted data or not: default not save," +
                                                                 " set 1 save")
    parser.add_argument("--optimizer_name", type=str, default="adam", help="the optimizer name, default is adam")
    args = parser.parse_args()
    lstm_num = 16
    dense_num = 10
    max_length = 32
    batchsize = 32
    train_predict(args.epoch_num, args.save_data, args.optimizer_name, lstm_num, dense_num, max_length, batchsize)


if __name__ == "__main__":
    main()

