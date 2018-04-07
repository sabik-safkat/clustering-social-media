# This file create and Vectorize and out result with training

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import time
start_time = time.time()



def TrainX(vec):
    fin = open("dataset18000.txt", encoding="utf-8")
    corpus = []
    text = fin.readline()

    while text:
        corpus.append(text.strip())
        text = fin.readline()
    fin.close()
    return vec.fit_transform(corpus)


def TrainY():
    trainY = []
    fin = open("result18000.txt")

    yval = fin.readline()
    while yval:
        trainY.append(int(yval))
        yval = fin.readline()

    return np.asarray(trainY)


def SVM(trainX,trainY,testX):
    model = svm.SVC(kernel="linear")
    model.fit(trainX, trainY)
    result = []
    for i in testX:
        result.append(model.predict(i))

    return np.asarray(result)

def LR(trainX, trainY, testX, testY):
    clf = LogisticRegression(fit_intercept=True, C = 1e15)
    clf.fit(trainX, trainY)

    print ('Accuracy from logistic regression: {0}'.format(clf.score(testX, testY)))

    print (clf.intercept_, clf.coef_)
    # print (weights)

def KNC(trainX, trainY, testX, testY):
    knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=3, p=2,
           weights='distance')
    knn.fit(trainX, trainY)
    # print(knn.predict(testX))
    # print(knn.predict_proba(testX))
    print(knn.predict_proba(data)[:, 1])
    print('accuracy for KNN:{0}'.format(knn.score(testX, testY)))

def MLP(trainX, trainY, testX, testY):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(7,), random_state=1)
    clf.fit(trainX, trainY)
    clf.predict(testX)
    print(clf.predict_proba(testX))
    print('accuracy for MLP:{0}'.format(clf.score(testX, testY)))




if __name__ == '__main__':
    vec = CountVectorizer(tokenizer=lambda x: x.split(),ngram_range=(1,4) )
    dataX = TrainX(vec)
    dataY = TrainY()

    print(dataX.shape)

    x = dataX[:13000]
    y = dataY[:13000]
    tx = dataX[13000:18000]
    ty = dataY[13000:18000]

    predictedResult = SVM(x,y,tx)

    # print(f1_score(ty,predictedResult))

    cm = confusion_matrix(ty,predictedResult)
    accu = accuracy_score(ty,predictedResult)

    print(cm)
    print("Accuracy = ", accu)
    print("--- %s seconds ---" % (time.time() - start_time))

    predict2= LR(x, y, tx, ty)

    print("--- %s seconds ---" % (time.time() - start_time))

    KNC(x, y, tx, ty)
    print("--- %s seconds ---" % (time.time() - start_time))

    MLP(x, y, tx, ty)
    print("--- %s seconds ---" % (time.time() - start_time))
