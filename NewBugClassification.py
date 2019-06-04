# import the necessary packages
from keras.models import load_model
import pickle
import gensim
import pandas as pd
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QFrame, QFileDialog, QPushButton, QHBoxLayout, QGroupBox, \
    QVBoxLayout, QGridLayout, QLineEdit
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PyQt5.QtCore import QDir


class App(QWidget):

    def __init__(self):
        super().__init__()

        self.title = 'Classify New Bug Report'
        self.left = 30
        self.top = 30
        self.width = 500
        self.height = 400
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setFixedSize(self.width, self.height)

        Loadbutton = QPushButton("Load New Bug Report (CSV file)", self)
        Loadbutton.clicked.connect(self.Load)
        Loadbutton.setFont(QFont("Times", 10))
        Loadbutton.move(220, 330)

        clssifier = QLabel(self)
        clssifier.setText("Select Classifier")
        clssifier.setFont(QFont("Times", 10, QFont.Bold))
        clssifier.move(50, 30)

        current = QLabel(self)
        current.setText("Current Severity")
        current.setFont(QFont("Times", 10, QFont.Bold))
        current.move(220, 70)

        self.current_severity = QLineEdit(self)
        self.current_severity.setFont(QFont("Times", 10))
        self.current_severity.move(220, 120)

        predicted = QLabel(self)
        predicted.setText("Predicted Severity")
        predicted.setFont(QFont("Times", 10, QFont.Bold))
        predicted.move(220, 170)

        self.predicted_severity = QLineEdit(self)
        self.predicted_severity.setFont(QFont("Times", 10))
        self.predicted_severity.move(220, 220)

        KNNbutton = QPushButton('KNN', self)
        KNNbutton.clicked.connect(self.KNN)
        KNNbutton.setFont(QFont("Times", 10))
        KNNbutton.move(50, 70)

        SVMbutton = QPushButton('SVM', self)
        SVMbutton.clicked.connect(self.SVM)
        SVMbutton.setFont(QFont("Times", 10))
        SVMbutton.move(50, 120)

        NBbutton = QPushButton('Naive Bayes', self)
        NBbutton.clicked.connect(self.NB)
        NBbutton.setFont(QFont("Times", 10))
        NBbutton.move(50, 170)

        GRUbutton = QPushButton('GRU', self)
        GRUbutton.clicked.connect(self.GRU)
        GRUbutton.setFont(QFont("Times", 10))
        GRUbutton.move(50, 220)

        LSTMbutton = QPushButton('LSTM', self)
        LSTMbutton.clicked.connect(self.LSTM)
        LSTMbutton.setFont(QFont("Times", 10))
        LSTMbutton.move(50, 270)
        # self.differentClassifiers()
        self.show()

    # def differentClassifiers(self):

    def Load(self):
        fileName, _ = QFileDialog.getOpenFileName(self, 'Single File', './DataSets/Single Bug Reports', '*.csv')
        if fileName:
            print(fileName)
            bug_report = pd.read_csv(fileName)
            bug_features = bug_report.drop(['severity'], axis=1).values
            if bug_report['severity'].values is not None:
                self.bug_label = bug_report['severity'].values
            print(self.bug_label)
            print(bug_features)
            self.bug_doc2vec = np.zeros((1, 500))
            self.bug_doc2vec[0] = self.applyDoc2Vec(bug_features, self.bug_label)
            print(self.bug_doc2vec)

    def TaggedDocument(self, features, label):
        docs = []
        for index, row in enumerate(features):
            docs.append(gensim.models.doc2vec.TaggedDocument(words=row, tags=[label[index]]))
        return docs

    def applyDoc2Vec(self, features, label):
        data = self.TaggedDocument(features, label)
        print(data)
        model = gensim.models.doc2vec.Doc2Vec(dm=0, vector_size=500, negative=5, window=6, hs=1, min_count=0,
                                              sample=1e-5, workers=3, alpha=0.05, min_alpha=0.001)
        model.build_vocab(data, update=False)
        model.init_sims(replace=True)
        model.train(data, total_examples=model.corpus_count, epochs=200)
        return model.docvecs[self.bug_label[0]]

    def KNN(self):
        print("KNN button")
        knn_model = pickle.load(open('ClassifiersModels/KNN_builtin_both_model.sav', 'rb'))
        predict = knn_model.predict(self.bug_doc2vec)
        print(predict)
        print(self.bug_label)
        self.predicted_severity.setText(str(predict))
        self.current_severity.setText(str(self.bug_label))

    def SVM(self):
        print("SVM button")
        svm_model = pickle.load(open('ClassifiersModels/SVM_both_model.sav', 'rb'))
        predict = svm_model.predict(self.bug_doc2vec)
        print(predict)
        print(self.bug_label)
        self.predicted_severity.setText(str(predict))
        self.current_severity.setText(str(self.bug_label))

    def NB(self):
        print("NB button")
        nb_model = pickle.load(open('ClassifiersModels/NaiveBayes_both_model.sav', 'rb'))
        predict = nb_model.predict(self.bug_doc2vec)
        print(predict)
        print(self.bug_label)
        self.predicted_severity.setText(str(predict))
        self.current_severity.setText(str(self.bug_label))

    def GRU(self):
        print("GRU button")
        gru_model = load_model('ClassifiersModels/GRU_both_model_keras.h5')
        predict = gru_model.predict_classes(self.bug_doc2vec.reshape(self.bug_doc2vec.shape + (1,)))
        if predict == 0:
            self.predicted_severity.setText("['blocker']")
            print("['blocker']")
        elif predict == 1:
            self.predicted_severity.setText("['critical']")
            print("['critical']")
        elif predict == 2:
            self.predicted_severity.setText("['major']")
            print("['major']")
        elif predict == 3:
            self.predicted_severity.setText("['minor']")
            print("['minor']")
        elif predict == 4:
            self.predicted_severity.setText("['trivial']")
            print("['trivial']")

        print(self.bug_label)
        self.current_severity.setText(str(self.bug_label))

    def LSTM(self):
        print("LSTM button")
        lstm_model = load_model('ClassifiersModels/LSTM_both_model_keras.h5')
        predict = lstm_model.predict_classes(self.bug_doc2vec.reshape(self.bug_doc2vec.shape + (1,)))
        if predict == 0:
            self.predicted_severity.setText("['blocker']")
            print("['blocker']")
        elif predict == 1:
            self.predicted_severity.setText("['critical']")
            print("['critical']")
        elif predict == 2:
            self.predicted_severity.setText("['major']")
            print("['major']")
        elif predict == 3:
            self.predicted_severity.setText("['minor']")
            print("['minor']")
        elif predict == 4:
            self.predicted_severity.setText("['trivial']")
            print("['trivial']")

        print(self.bug_label)
        self.current_severity.setText(str(self.bug_label))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
