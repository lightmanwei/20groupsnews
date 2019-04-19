import os
import numpy as np
import sys
import re
import math
import simpleMLP

BASE_DIR = '.'
GLOVE_DIR = BASE_DIR + '/glove.6B/'
TEXT_DATA_DIR = BASE_DIR + '/20newsbydate/20news-bydate-train/'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
batch_size = 32
embeddings_index = {}
DOC_NUM = 2

train_matrix = []
wordList = []
texts = []
labels_index = {}
labels = []
resultLabels = []


def putOutrRsult(content):
    f = open(os.path.join("output.txt"), 'w', encoding='utf-8')  # 文件操作
    f.write(content)


# 都传index

def tf(word, document, matrix):
    return matrix[document][word] / sum(matrix[document])


def n_containing(word, matrix):
    return sum(1 for document in matrix if document[word] > 0)


def idf(word, matrix):
    return math.log(len(matrix)) / (1 + n_containing(word, matrix))


def tfidf(word, document, matrix):  # word 文件 大字典
    return tf(word, document, matrix) * idf(word, matrix)


print('Processing text dataset')

if __name__ == '__main__':
    for name in sorted(os.listdir(TEXT_DATA_DIR)):
        path = os.path.join(TEXT_DATA_DIR, name)
        label_id = 0
        if os.path.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id  # 每个文件夹给一个ID
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = os.path.join(path, fname)
                    if sys.version_info < (3,):
                        f = open(fpath)
                    else:
                        f = open(fpath, encoding='latin-1')
                    texts.append(f.read().lower())
                    f.close()
                    resultLabels.append(label_id)
                    labels.append(fname)

    print('Found %s texts.' % len(texts))

    reg = re.compile('\\W*')
    bigText = []
    for item in texts:
        try:
            bigText.append(reg.split(item))
        except:
            print("bigText is empty!")
            exit()

    for document in bigText:
        for word in document:
            if not word in wordList:
                wordList.append(word)

    train_matrix = np.zeros((len(labels), len(wordList)), dtype=np.double)
    for document in bigText:
        for word in document:
            train_matrix[bigText.index(document), wordList.index(word)] += 1

    for document in range(len(labels)):
        for word in range(len(wordList)):
            train_matrix[document][word] = tfidf(word, document, train_matrix)

    putOutrRsult(str(train_matrix))

    x = train_matrix
    y = np.zeros((len(x), DOC_NUM), dtype=np.int)
    for resut, y0 in zip(resultLabels, y):
        y0[resut] = 1

    # label = np.argmax(y, axis=0)
    nn = simpleMLP.NaiveNN()
    losses = nn.fit(nn, x, y, 100, 1e-5)
    # pred=nn.predict(x)
    print("准确率：{:8.6} %".format((nn.predict(x) == y).mean() * 100))
