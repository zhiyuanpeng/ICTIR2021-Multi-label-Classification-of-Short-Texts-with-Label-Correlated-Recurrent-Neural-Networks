import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.pipeline import Pipeline
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import argparse

def get_text(file_name):
    """
    read the text and return a list
    :param file_name:
    :return: a list of text
    """
    text = []
    with open(file_name, "r") as f:
        for line in f.readlines():
            text.append(line.lower())
    return text


def get_result(max_length):
    x_test = get_text("data/processed" + "/X_test.txt")
    x_train = get_text("data/processed" + "/X_train.txt")
    y_test = np.loadtxt("data/processed" + "/y_test.txt", dtype=int)
    y_train = np.loadtxt("data/processed" + "/y_train.txt", dtype=int)
    # word index from 1
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(x_train)
    x_train_cut_num = tokenizer.texts_to_sequences(x_train)
    x_test_cut_num = tokenizer.texts_to_sequences(x_test)
    x_train_cut_num_pad = pad_sequences(x_train_cut_num, padding="post", maxlen=max_length, value=4)
    x_test_cut_num_pad = pad_sequences(x_test_cut_num, padding="post", maxlen=max_length, value=4)
    x_train_cut_text = tokenizer.sequences_to_texts(x_train_cut_num_pad)
    x_test_cut_text = tokenizer.sequences_to_texts(x_test_cut_num_pad)

    nb_pipeline = Pipeline([('tfidf', TfidfVectorizer()),
                            ('clf', MultinomialNB(fit_prior=True, class_prior=None))])
    nb_total = 0
    for i in range(y_test.shape[1]):
        nb_pipeline.fit(x_train_cut_text, y_train[:, i])
        nb_predict = nb_pipeline.predict(x_test_cut_text)
        nb_total += np.sum([y_test[j, i] == nb_predict[j] for j in range(y_test.shape[0])])
    total_num = y_test.shape[0]*y_test.shape[1]
    print("navie bayes:")
    print(nb_total/total_num)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("max_length", type=int, help="the truncation length of the text or query, usually longer " +
                                                 "than the average length of the text")
    args = parser.parse_args()
    get_result(args.max_length)


if __name__ == "__main__":
    main()




