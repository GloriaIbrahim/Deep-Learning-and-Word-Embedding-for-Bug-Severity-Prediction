import gensim
import pandas as pd
from xlwt import Workbook
from sklearn.preprocessing import LabelEncoder

# read datasets
Mozilla = pd.read_csv("DataSets/Mozilla_total.csv")
Eclipse = pd.read_csv("DataSets/Eclipse_total.csv")
BothDataSets = pd.read_csv("DataSets/Both_DataSets.csv")

# print the first 5 rows of the dataframe
print('\nMozilla top 5 rows')
print(Mozilla.head())
print('\nEclipse top 5 rows')
print(Eclipse.head())

# observe the shape of the dataframe
print('\nMozilla shape: ')
print(Mozilla.shape)
print('\nEclipse shape: ')
print(Eclipse.shape)

# creating numpy arrays for features and target
MozillaFeature = Mozilla.drop(['severity'], axis=1).values
MozillaLabel = Mozilla['severity'].values
MozillaIDs = Mozilla['bug ID'].values
EclipseFeature = Eclipse.drop(['severity'], axis=1).values
EclipseLabel = Eclipse['severity'].values
EclipseIDs = Eclipse['bug ID'].values
BothFeature = BothDataSets.drop(['severity'], axis=1).values
BothLabel = BothDataSets['severity'].values
BothIDs = BothDataSets['bug ID'].values


# split documents into taggedDocument function
def TaggedDocument(features, label):
    docs = []
    for index, row in enumerate(features):
        docs.append(gensim.models.doc2vec.TaggedDocument(words=row, tags=[label[index]+str(index)]))
    return docs


# train the model using doc2vec function
def doc2vec(features, labels):
    """
    dm = 0, distributed bag of words(DBOW) is used.
    vector_size = 500, 500 vector dimensional feature vectors.
    negative = 5, specifies how many “noise words” should be drawn.
    min_count = 3, ignores all words with total frequency lower than this.
    alpha = 0.05, the initial learning rate.
    """
    data = TaggedDocument(features, labels)
    model = gensim.models.doc2vec.Doc2Vec(dm=0, vector_size=500, negative=5, window=6, hs=1, min_count=3, sample=1e-5, workers=3, alpha=0.05, min_alpha=0.001)
    model.build_vocab(data)
    model.init_sims(replace=True)
    model.train(data, total_examples=model.corpus_count, epochs=200)
    return model


def encodingClass(targetModel):
    labelEncoder = LabelEncoder()
    # LabelEncoder to encode class into categorical integer values
    labelEncoder.fit(targetModel)
    encodedTarget = labelEncoder.transform(targetModel)
    print("Label classes: " + labelEncoder.classes_)
    return encodedTarget


MozillaLabelEncoded = encodingClass(MozillaLabel)
EclipseLabelEncoded = encodingClass(EclipseLabel)
BothLabelEncoded = encodingClass(BothLabel)

# train the model on Mozilla dataset
MozillaDoc2vecModel = doc2vec(MozillaFeature, MozillaLabel)
MozillaDoc2vecModel.save('Doc2VecModels/MozillaDoc2Vec.model')
print('Mozilla Dataset Doc2vec Model')
print(MozillaDoc2vecModel.corpus_count)


# train the model on Eclipse dataset
EclipseDoc2vecModel = doc2vec(EclipseFeature, EclipseLabel)
EclipseDoc2vecModel.save('Doc2VecModels/EclipseDoc2Vec.model')
print('Eclipse Dataset Doc2vec Model')
print(EclipseDoc2vecModel.corpus_count)


# train the model on Both datasets together
BothDoc2vecModel = doc2vec(BothFeature, BothLabel)
BothDoc2vecModel.save('Doc2VecModels/BothDoc2vec.model')
print('Both Datasets Doc2vec Model')
print(BothDoc2vecModel.corpus_count)

'''
mozilla_workbook = Workbook()
eclipse_workbook = Workbook()
mozilla_sheet = mozilla_workbook.add_sheet('Mozilla_Doc2Vec_Model')
eclipse_sheet = eclipse_workbook.add_sheet('Eclipse_Doc2Vec_Model')


mozilla_sheet.write(0, 0, 'ID')
mozilla_sheet.write(0, 1, 'Document Vector')
mozilla_sheet.write(0, 2, 'Severity')
mozilla_sheet.write(0, 3, 'Severity Number')
for i in range(len(MozillaFeature)):
    mozilla_sheet.write(i+1, 0, str(MozillaIDs[i]))
    mozilla_sheet.write(i+1, 1, str(MozillaDoc2vecModel.docvecs[MozillaLabel[i]+str(i)]))
    mozilla_sheet.write(i+1, 2, str(MozillaLabel[i]))
    mozilla_sheet.write(i+1, 3, str(MozillaLabelEncoded[i]))

mozilla_workbook.save('Doc2VecModelsExcel/Mozilla_Doc2Vec_Model_Vectors.xls')

eclipse_sheet.write(0, 0, 'ID')
eclipse_sheet.write(0, 1, 'Document Vector')
eclipse_sheet.write(0, 2, 'Severity')
eclipse_sheet.write(0, 3, 'Severity Number')
for i in range(len(EclipseFeature)):
    eclipse_sheet.write(i+1, 0, str(EclipseIDs[i]))
    eclipse_sheet.write(i+1, 1, str(EclipseDoc2vecModel.docvecs[EclipseLabel[i]+str(i)]))
    eclipse_sheet.write(i+1, 2, str(EclipseLabel[i]))
    eclipse_sheet.write(i+1, 3, str(EclipseLabelEncoded[i]))

eclipse_workbook.save('Doc2VecModelsExcel/Eclipse_Doc2Vec_Model_Vectors.xls')
'''
