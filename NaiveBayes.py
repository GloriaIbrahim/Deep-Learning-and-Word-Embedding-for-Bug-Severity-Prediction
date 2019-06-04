import pickle
import gensim
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB


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


# create Naive Bayes classifier
def NB_classifier():
    model = GaussianNB()
    return model


# apply Naive Bayes on Mozilla dataset
def applyNBonMozilla(MozillaVectorsTrain, MozillaVectorsTest, MozillaLabelTrain, MozillaLabelTest):
    print('Applying Naive Bayes classifier on Mozilla Dataset')
    MozillaNBModel = NB_classifier()
    MozillaNBModel.fit(MozillaVectorsTrain, MozillaLabelTrain)
    predictions = MozillaNBModel.predict(MozillaVectorsTest)
    print('Accuracy: {}'.format(accuracy_score(MozillaLabelTest, predictions)))
    print('F1 score: {}'.format(f1_score(MozillaLabelTest, predictions, average='weighted')))
    print('Confusion Matrix: {}'.format(confusion_matrix(MozillaLabelTest, predictions)))
    print(classification_report(MozillaLabelTest, predictions))
    plot_confusion_matrix(confusion_matrix(MozillaLabelTest, predictions), 'Mozilla')
    plot_classification_report(classification_report(MozillaLabelTest, predictions), 'Mozilla')
    pickle.dump(MozillaNBModel, open('ClassifiersModels/NaiveBayes_mozilla_model.sav', 'wb'))


# apply Naive Bayes on Eclipse dataset
def applyNBonEclipse(EclipseVectorsTrain, EclipseVectorsTest, EclipseLabelTrain, EclipseLabelTest):
    print('Applying Naive Bayes classifier on Eclipse Dataset')
    EclipseNBModel = NB_classifier()
    EclipseNBModel.fit(EclipseVectorsTrain, EclipseLabelTrain)
    predictions = EclipseNBModel.predict(EclipseVectorsTest)
    print('Accuracy: {}'.format(accuracy_score(EclipseLabelTest, predictions)))
    print('F1 score: {}'.format(f1_score(EclipseLabelTest, predictions, average='weighted')))
    print('Confusion Matrix: {}'.format(confusion_matrix(EclipseLabelTest, predictions)))
    print(classification_report(EclipseLabelTest, predictions))
    plot_confusion_matrix(confusion_matrix(EclipseLabelTest, predictions), 'Eclipse')
    plot_classification_report(classification_report(EclipseLabelTest, predictions), 'Eclipse')
    pickle.dump(EclipseNBModel, open('ClassifiersModels/NaiveBayes_eclipse_model.sav', 'wb'))


# apply Naive Bayes on Both dataset
def applyNBonBoth(BothVectors, BothLabel):
    BothNBModel = NB_classifier()
    BothNBModel.fit(BothVectors, BothLabel)
    pickle.dump(BothNBModel, open('ClassifiersModels/NaiveBayes_both_model.sav', 'wb'))


def plot_classification_report(classificationReport, dataset, cmap='RdBu'):
    lines = classificationReport.split('\n')

    classes = []
    plotMat = []
    for line in lines[2:7]:
        t = line.strip().split()
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        plotMat.append(v)

    plt.imshow(plotMat, interpolation='nearest', cmap=cmap)
    plt.title('Classification Report of '+dataset+' Dataset')
    plt.colorbar()
    x_tick_marks = np.arange(3)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('Classes')
    plt.xlabel('Measures')
    plt.show()


def plot_confusion_matrix(cm, dataset):
    labels = ['Blocker', 'Critical', 'Major', 'Minor', 'Trivial']
    fig, ax = plt.subplots()
    h = ax.matshow(cm)
    fig.colorbar(h)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    ax.set_xlabel('Confusion Matrix of '+dataset+' Dataset')
    ax.set_ylabel('Severity classes')
    plt.show()


if __name__ == '__main__':
    MozillaVectorsTrain, MozillaVectorsTest, MozillaLabelTrain, MozillaLabelTest, EclipseVectorsTrain, EclipseVectorsTest, EclipseLabelTrain, EclipseLabelTest, BothVectors, BothLabel = prepareData()
    applyNBonMozilla(MozillaVectorsTrain, MozillaVectorsTest, MozillaLabelTrain, MozillaLabelTest)
    applyNBonEclipse(EclipseVectorsTrain, EclipseVectorsTest, EclipseLabelTrain, EclipseLabelTest)
    applyNBonBoth(BothVectors, BothLabel)
