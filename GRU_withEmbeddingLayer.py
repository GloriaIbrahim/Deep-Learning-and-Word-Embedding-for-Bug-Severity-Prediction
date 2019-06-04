import gensim
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, GRU, Embedding, Flatten
from keras.optimizers import Adam
import matplotlib.pyplot as plt


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

    MozillaLabelEncoded = encodingClass(MozillaLabel)
    EclipseLabelEncoded = encodingClass(EclipseLabel)
    BothLabelEncoded = encodingClass(BothLabel)

    # split datasets
    MozillaVectorsTrain, MozillaVectorsTest, MozillaLabelTrain, MozillaLabelTest = train_test_split(MozillaVectors,
                                                                                                    MozillaLabelEncoded,
                                                                                                    test_size=0.3,
                                                                                                    stratify=MozillaLabelEncoded)
    EclipseVectorsTrain, EclipseVectorsTest, EclipseLabelTrain, EclipseLabelTest = train_test_split(EclipseVectors,
                                                                                                    EclipseLabelEncoded,
                                                                                                    test_size=0.3,
                                                                                                    stratify=EclipseLabelEncoded)

    return MozillaVectorsTrain, MozillaVectorsTest, MozillaLabelTrain, MozillaLabelTest, EclipseVectorsTrain, EclipseVectorsTest, EclipseLabelTrain, \
           EclipseLabelTest, BothVectors, BothLabelEncoded, MozillaDoc2VecModel, EclipseDoc2VecModel, BothDoc2VecModel


# apply label encoder on labels
def encodingClass(targetModel):
    labelEncoder = LabelEncoder()
    # LabelEncoder to encode class into categorical integer values
    labelEncoder.fit(targetModel)
    encodedTarget = labelEncoder.transform(targetModel)
    print("Label classes: " + labelEncoder.classes_)
    return encodedTarget


# create GRU model
def GRUmodel(doc2vecmodel):
    model = Sequential()
    model.add(Embedding(doc2vecmodel.corpus_count, 64, input_length=500))
    model.add(GRU(64, dropout=0.4, recurrent_dropout=0.4, return_sequences=True))
    model.add(GRU(32, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(5, activation='softmax'))

    # Define the optimizer and compile the model
    # optimizer = optimizers.SGD(lr=0.03, clipnorm=5.)
    model.compile(optimizer=Adam(lr=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print("model defined")
    print(model.summary())
    return model


# apply GRU model on mozilla dataset
def applyGRUonMozilla(MozillaVectorsTrain, MozillaVectorsTest, MozillaLabelTrain, MozillaLabelTest, MozillaDoc2VecModel):
    print('Applying GRU classifier on Mozilla Dataset')
    MozillaGRUModel = GRUmodel(MozillaDoc2VecModel)
    MozillaGRUModel.fit(MozillaVectorsTrain, MozillaLabelTrain, batch_size=200, epochs=5)

    predictions = MozillaGRUModel.predict_classes(MozillaVectorsTest)
    print('Accuracy: {}'.format(accuracy_score(MozillaLabelTest, predictions)))
    print('F1 score: {}'.format(f1_score(MozillaLabelTest, predictions, average='weighted')))
    print('Confusion Matrix: {}'.format(confusion_matrix(MozillaLabelTest, predictions)))
    print(classification_report(MozillaLabelTest, predictions))
    plot_confusion_matrix(confusion_matrix(MozillaLabelTest, predictions), 'Mozilla')
    plot_classification_report(classification_report(MozillaLabelTest, predictions), 'Mozilla')
    MozillaGRUModel.save('ClassifiersModels/GRU_wthEmbedding_mozilla_model_keras.h5')


# apply GRU model on eclipse dataset
def applyGRUonEclipse(EclipseVectorsTrain, EclipseVectorsTest, EclipseLabelTrain, EclipseLabelTest, EclipseDoc2VecModel):
    print('Applying GRU classifier on Eclipse Dataset')
    EclipseGRUModel = GRUmodel(EclipseDoc2VecModel)
    EclipseGRUModel.fit(EclipseVectorsTrain, EclipseLabelTrain, batch_size=200, epochs=5)

    predictions = EclipseGRUModel.predict_classes(EclipseVectorsTest)
    print('Accuracy: {}'.format(accuracy_score(EclipseLabelTest, predictions)))
    print('F1 score: {}'.format(f1_score(EclipseLabelTest, predictions, average='weighted')))
    print('Confusion Matrix: {}'.format(confusion_matrix(EclipseLabelTest, predictions)))
    print(classification_report(EclipseLabelTest, predictions))
    plot_confusion_matrix(confusion_matrix(EclipseLabelTest, predictions), 'Eclipse')
    plot_classification_report(classification_report(EclipseLabelTest, predictions), 'Eclipse')
    EclipseGRUModel.save('ClassifiersModels/GRU_wthEmbedding_eclipse_model_keras.h5')


# apply GRU model on both datasets
def applyGRUonBoth(BothVectors, BothLabelEncoded, BothDoc2VecModel):
    BothGRUModel = GRUmodel(BothDoc2VecModel)
    BothGRUModel.fit(BothVectors, BothLabelEncoded, batch_size=200, epochs=5)

    BothGRUModel.save('ClassifiersModels/GRU_wthEmbedding_both_model_keras.h5')


def plot_classification_report(classificationReport, dataset, cmap='RdBu'):
    lines = classificationReport.split('\n')
    classes = ['Blocker', 'Critical', 'Major', 'Minor', 'Trivial']
    plotMat = []
    for line in lines[2:7]:
        t = line.strip().split()
        # classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        plotMat.append(v)

    plt.imshow(plotMat, interpolation='nearest', cmap=cmap)
    plt.title('Classification Report of '+dataset+' Dataset With GRU')
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
    ax.set_xlabel('Confusion Matrix of '+dataset+' Dataset With GRU')
    ax.set_ylabel('Severity classes')
    plt.show()


if __name__ == '__main__':
    MozillaVectorsTrain, MozillaVectorsTest, MozillaLabelTrain, MozillaLabelTest, EclipseVectorsTrain, EclipseVectorsTest, \
    EclipseLabelTrain, EclipseLabelTest, BothVectors, BothLabelEncoded, MozillaDoc2VecModel, EclipseDoc2VecModel, BothDoc2VecModel = prepareData()
    applyGRUonMozilla(MozillaVectorsTrain, MozillaVectorsTest, MozillaLabelTrain, MozillaLabelTest, MozillaDoc2VecModel)
    applyGRUonEclipse(EclipseVectorsTrain, EclipseVectorsTest, EclipseLabelTrain, EclipseLabelTest, EclipseDoc2VecModel)
    applyGRUonBoth(BothVectors, BothLabelEncoded, BothDoc2VecModel)
