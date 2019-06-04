import matplotlib.pyplot as plt
import gensim
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import math
import operator


def prepareData():
    # read datasets
    Mozilla = pd.read_csv("DataSets/Mozilla_total.csv")
    Eclipse = pd.read_csv("DataSets/Eclipse_total.csv")
    BothDataSets = pd.read_csv("DataSets/Both_DataSets.csv")

    # creating numpy arrays for labels
    MozillaLabel = Mozilla['severity'].values
    EclipseLabel = Eclipse['severity'].values
    BothLabel = BothDataSets['severity'].values

    # read the doc2vec models
    MozillaDoc2VecModel = gensim.models.doc2vec.Doc2Vec.load('Doc2VecModels/MozillaDoc2Vec.model')
    EclipseDoc2VecModel = gensim.models.doc2vec.Doc2Vec.load('Doc2VecModels/EclipseDoc2Vec.model')
    BothDoc2VecModel = gensim.models.doc2vec.Doc2Vec.load('Doc2VecModels/BothDoc2Vec.model')

    MozillaVectors = np.zeros((len(MozillaLabel), 500))
    EclipseVectors = np.zeros((len(EclipseLabel), 500))
    BothVectors = np.zeros((len(BothLabel), 500))

    # get documents vectors from the doc2vec model
    for i in range(len(MozillaLabel)):
        MozillaVectors[i] = MozillaDoc2VecModel.docvecs[MozillaLabel[i] + str(i)]

    for i in range(len(EclipseLabel)):
        EclipseVectors[i] = EclipseDoc2VecModel.docvecs[EclipseLabel[i] + str(i)]

    for i in range(len(BothLabel)):
        BothVectors[i] = BothDoc2VecModel.docvecs[BothLabel[i] + str(i)]

    # split datasets
    MozillaVectorsTrain, MozillaVectorsTest, MozillaLabelTrain, MozillaLabelTest = train_test_split(MozillaVectors, MozillaLabel, test_size=0.3, stratify=MozillaLabel)
    EclipseVectorsTrain, EclipseVectorsTest, EclipseLabelTrain, EclipseLabelTest = train_test_split(EclipseVectors, EclipseLabel, test_size=0.3, stratify=EclipseLabel)
    return MozillaVectorsTrain, MozillaVectorsTest, MozillaLabelTrain, MozillaLabelTest, EclipseVectorsTrain, EclipseVectorsTest, EclipseLabelTrain, EclipseLabelTest


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def getNeighbors(trainingSet, testInstance, trainingLabel, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingLabel[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
        # print(neighbors[x])
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


# create KNN classifier
def KNN_classifier(MozillaVectorsTrain, MozillaVectorsTest, MozillaLabelTrain, k):
    predictions = []
    for x in range(len(MozillaVectorsTest)):
        neighbors = getNeighbors(MozillaVectorsTrain, MozillaVectorsTest[x], MozillaLabelTrain, 1)
        result = getResponse(neighbors)
        predictions.append(result)
    return predictions


# apply KNN on Mozilla dataset
def applyKNNonMozilla(MozillaVectorsTrain, MozillaVectorsTest, MozillaLabelTrain, MozillaLabelTest):
    error = []
    accuracy = []
    print('Applying KNN classifier on Mozilla Dataset with K=' + str(1))
    predictions = KNN_classifier(MozillaVectorsTrain, MozillaVectorsTest, MozillaLabelTrain, 1)
    error.append(np.mean(predictions != MozillaLabelTest))
    print('Accuracy: {}'.format(accuracy_score(MozillaLabelTest, predictions)))
    accuracy.append(accuracy_score(MozillaLabelTest, predictions))
    print('F1 score: {}'.format(f1_score(MozillaLabelTest, predictions, average='weighted')))
    print('Confusion Matrix: {}'.format(confusion_matrix(MozillaLabelTest, predictions)))
    print(classification_report(MozillaLabelTest, predictions))
    plot_confusion_matrix(confusion_matrix(MozillaLabelTest, predictions), 'Mozilla', str(1))
    plot_classification_report(classification_report(MozillaLabelTest, predictions), 'Mozilla', str(1))
    for i in range(5, 30, 5):
        print('Applying KNN classifier on Mozilla Dataset with K='+str(i))
        predictions = KNN_classifier(MozillaVectorsTrain, MozillaVectorsTest, MozillaLabelTrain, i)
        error.append(np.mean(predictions != MozillaLabelTest))
        print('Accuracy: {}'.format(accuracy_score(MozillaLabelTest, predictions)))
        accuracy.append(accuracy_score(MozillaLabelTest, predictions))
        print('F1 score: {}'.format(f1_score(MozillaLabelTest, predictions, average='weighted')))
        print('Confusion Matrix: {}'.format(confusion_matrix(MozillaLabelTest, predictions)))
        print(classification_report(MozillaLabelTest, predictions))
        plot_confusion_matrix(confusion_matrix(MozillaLabelTest, predictions), 'Mozilla', str(i))
        plot_classification_report(classification_report(MozillaLabelTest, predictions), 'Mozilla', str(i))

    plot_error_rate(error, 'Mozilla')
    plot_accuracy_rate(accuracy, 'Mozilla')


# apply KNN on Eclipse dataset
def applyKNNonEclipse(EclipseVectorsTrain, EclipseVectorsTest, EclipseLabelTrain, EclipseLabelTest):
    error = []
    accuracy = []
    print('Applying KNN classifier on Eclipse Dataset with K=' + str(1))
    predictions = KNN_classifier(EclipseVectorsTrain, EclipseVectorsTest, EclipseLabelTrain, 1)
    error.append(np.mean(predictions != EclipseLabelTest))
    print('Accuracy: {}'.format(accuracy_score(EclipseLabelTest, predictions)))
    accuracy.append(accuracy_score(EclipseLabelTest, predictions))
    print('F1 score: {}'.format(f1_score(EclipseLabelTest, predictions, average='weighted')))
    print('Confusion Matrix: {}'.format(confusion_matrix(EclipseLabelTest, predictions)))
    print(classification_report(EclipseLabelTest, predictions))
    plot_confusion_matrix(confusion_matrix(EclipseLabelTest, predictions), 'Eclipse', str(1))
    plot_classification_report(classification_report(EclipseLabelTest, predictions), 'Eclipse', str(1))
    for i in range(5, 30, 5):
        print('Applying KNN classifier on Eclipse Dataset with K='+str(i))
        predictions = KNN_classifier(EclipseVectorsTrain, EclipseVectorsTest, EclipseLabelTrain, i)
        error.append(np.mean(predictions != EclipseLabelTest))
        print('Accuracy: {}'.format(accuracy_score(EclipseLabelTest, predictions)))
        accuracy.append(accuracy_score(EclipseLabelTest, predictions))
        print('F1 score: {}'.format(f1_score(EclipseLabelTest, predictions, average='weighted')))
        print('Confusion Matrix: {}'.format(confusion_matrix(EclipseLabelTest, predictions)))
        print(classification_report(EclipseLabelTest, predictions))
        plot_confusion_matrix(confusion_matrix(EclipseLabelTest, predictions), 'Eclipse', str(i))
        plot_classification_report(classification_report(EclipseLabelTest, predictions), 'Eclipse', str(i))

    plot_error_rate(error, 'Eclipse')
    plot_accuracy_rate(accuracy, 'Eclipse')


def plot_error_rate(error, dataset):
    plt.figure(figsize=(12, 6))
    plt.plot(range(0, 30, 5), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value In '+dataset+' Dataset')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    plt.show()


def plot_accuracy_rate(accuracy, dataset):
    plt.figure(figsize=(12, 6))
    plt.plot(range(0, 30, 5), accuracy, color='red', linestyle='dashed', marker='o', markerfacecolor='blue',
             markersize=10)
    plt.title('Accuracy Rate K Value In '+dataset+' Dataset')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.show()


def plot_classification_report(classificationReport, dataset, k, cmap='RdBu'):
    lines = classificationReport.split('\n')
    classes = []
    plotMat = []
    for line in lines[2:7]:
        t = line.strip().split()
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        plotMat.append(v)

    plt.imshow(plotMat, interpolation='nearest', cmap=cmap)
    plt.title('Classification Report of '+dataset+' Dataset With K='+k)
    plt.colorbar()
    x_tick_marks = np.arange(3)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('Classes')
    plt.xlabel('Measures')
    plt.show()


def plot_confusion_matrix(cm, dataset,k):
    labels = ['Blocker', 'Critical', 'Major', 'Minor', 'Trivial']
    fig, ax = plt.subplots()
    h = ax.matshow(cm)
    fig.colorbar(h)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    ax.set_xlabel('Confusion Matrix of '+dataset+' Dataset With K='+k)
    ax.set_ylabel('Severity classes')
    plt.show()


if __name__ == '__main__':

    MozillaVectorsTrain, MozillaVectorsTest, MozillaLabelTrain, MozillaLabelTest, EclipseVectorsTrain, EclipseVectorsTest, EclipseLabelTrain, EclipseLabelTest = prepareData()
    applyKNNonMozilla(MozillaVectorsTrain, MozillaVectorsTest, MozillaLabelTrain, MozillaLabelTest)
    applyKNNonEclipse(EclipseVectorsTrain, EclipseVectorsTest, EclipseLabelTrain, EclipseLabelTest)
