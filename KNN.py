import pickle
import matplotlib.pyplot as plt
import gensim
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


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
        MozillaVectors[i] = MozillaDoc2VecModel.docvecs[MozillaLabel[i]+str(i)]

    for i in range(len(EclipseLabel)):
        EclipseVectors[i] = EclipseDoc2VecModel.docvecs[EclipseLabel[i]+str(i)]

    for i in range(len(BothLabel)):
        BothVectors[i] = BothDoc2VecModel.docvecs[BothLabel[i]+str(i)]

    # split datasets
    MozillaVectorsTrain, MozillaVectorsTest, MozillaLabelTrain, MozillaLabelTest = train_test_split(MozillaVectors, MozillaLabel, test_size=0.3, stratify=MozillaLabel)
    EclipseVectorsTrain, EclipseVectorsTest, EclipseLabelTrain, EclipseLabelTest = train_test_split(EclipseVectors, EclipseLabel, test_size=0.3, stratify=EclipseLabel)
    return MozillaVectorsTrain, MozillaVectorsTest, MozillaLabelTrain, MozillaLabelTest, EclipseVectorsTrain, EclipseVectorsTest, EclipseLabelTrain, EclipseLabelTest, BothVectors, BothLabel


# create KNN classifier
def KNN_classifier(k):
    model = KNeighborsClassifier(algorithm='auto', leaf_size=200, metric='minkowski', metric_params=None, n_jobs=1,
                                 n_neighbors=k, p=2, weights='uniform')
    return model


# apply KNN on Mozilla dataset
def applyKNNonMozilla(MozillaVectorsTrain, MozillaVectorsTest, MozillaLabelTrain, MozillaLabelTest):
    error = []
    classifiers = []
    accuracy = []
    c = 0
    print('Applying KNN classifier on Mozilla Dataset with K=' + str(1))
    classifiers.append(KNN_classifier(1))
    classifiers[c].fit(MozillaVectorsTrain, MozillaLabelTrain)
    predictions = classifiers[c].predict(MozillaVectorsTest)
    error.append(np.mean(predictions != MozillaLabelTest))
    print('Accuracy: {}'.format(accuracy_score(MozillaLabelTest, predictions)))
    accuracy.append(accuracy_score(MozillaLabelTest, predictions))
    print('F1 score: {}'.format(f1_score(MozillaLabelTest, predictions, average='weighted')))
    print('Confusion Matrix: {}'.format(confusion_matrix(MozillaLabelTest, predictions)))
    print(classification_report(MozillaLabelTest, predictions))
    plot_confusion_matrix(confusion_matrix(MozillaLabelTest, predictions), 'Mozilla', str(1))
    plot_classification_report(classification_report(MozillaLabelTest, predictions), 'Mozilla', str(1))

    for i in range(5, 30, 5):
        c = c+1
        print('Applying KNN classifier on Mozilla Dataset with K='+str(i))
        classifiers.append(KNN_classifier(i))
        classifiers[c].fit(MozillaVectorsTrain, MozillaLabelTrain)
        predictions = classifiers[c].predict(MozillaVectorsTest)
        # MozillaKNNModel = KNN_classifier(i)
        # MozillaKNNModel.fit(MozillaVectorsTrain, MozillaLabelTrain)
        # predictions = MozillaKNNModel.predict(MozillaVectorsTest)
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
    pickle.dump(classifiers[3], open('ClassifiersModels/KNN_builtin_mozilla_model.sav', 'wb'))


# apply KNN on Eclipse dataset
def applyKNNonEclipse(EclipseVectorsTrain, EclipseVectorsTest, EclipseLabelTrain, EclipseLabelTest):
    error = []
    classifiers = []
    accuracy = []
    c = 0
    print('Applying KNN classifier on Eclipse Dataset with K=' + str(1))
    classifiers.append(KNN_classifier(1))
    classifiers[c].fit(EclipseVectorsTrain, EclipseLabelTrain)
    predictions = classifiers[c].predict(EclipseVectorsTest)
    error.append(np.mean(predictions != EclipseLabelTest))
    print('Accuracy: {}'.format(accuracy_score(EclipseLabelTest, predictions)))
    accuracy.append(accuracy_score(EclipseLabelTest, predictions))
    print('F1 score: {}'.format(f1_score(EclipseLabelTest, predictions, average='weighted')))
    print('Confusion Matrix: {}'.format(confusion_matrix(EclipseLabelTest, predictions)))
    print(classification_report(EclipseLabelTest, predictions))
    plot_confusion_matrix(confusion_matrix(EclipseLabelTest, predictions), 'Eclipse', str(1))
    plot_classification_report(classification_report(EclipseLabelTest, predictions), 'Eclipse', str(1))

    for i in range(5, 30, 5):
        c = c+1
        print('Applying KNN classifier on Mozilla Dataset with K=' + str(i))
        classifiers.append(KNN_classifier(i))
        classifiers[c].fit(EclipseVectorsTrain, EclipseLabelTrain)
        predictions = classifiers[c].predict(EclipseVectorsTest)
        # EclipseKNNModel = KNN_classifier(i)
        # EclipseKNNModel.fit(EclipseVectorsTrain, EclipseLabelTrain)
        # predictions = EclipseKNNModel.predict(EclipseVectorsTest)
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
    pickle.dump(classifiers[3], open('ClassifiersModels/KNN_builtin_eclipse_model.sav', 'wb'))


# apply KNN on Both dataset
def applyKNNonBoth(BothVectors, BothLabel):
    classifiers = []
    c = 0
    classifiers.append(KNN_classifier(1))
    classifiers[c].fit(BothVectors, BothLabel)

    for i in range(5, 30, 5):
        c = c+1
        classifiers.append(KNN_classifier(i))
        classifiers[c].fit(BothVectors, BothLabel)

    pickle.dump(classifiers[3], open('ClassifiersModels/KNN_builtin_both_model.sav', 'wb'))


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
    MozillaVectorsTrain, MozillaVectorsTest, MozillaLabelTrain, MozillaLabelTest, EclipseVectorsTrain, EclipseVectorsTest, EclipseLabelTrain, EclipseLabelTest, BothVectors, BothLabel = prepareData()
    applyKNNonMozilla(MozillaVectorsTrain, MozillaVectorsTest, MozillaLabelTrain, MozillaLabelTest)
    applyKNNonEclipse(EclipseVectorsTrain, EclipseVectorsTest, EclipseLabelTrain, EclipseLabelTest)
    applyKNNonBoth(BothVectors, BothLabel)
