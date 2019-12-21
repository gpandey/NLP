'''This Code Written in Pycharm'''
# import sys
# sys.path.append('C:\\Users\\gitaa\\Desktop\\gita\\')
import numpy as np
import pandas as pd
#ignore warning
import warnings
warnings.filterwarnings('ignore')

from sklearn import model_selection, preprocessing, linear_model,naive_bayes,metrics,svm
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,TfidfTransformer
from sklearn import decomposition,ensemble
import xgboost,string, textblob
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

print('Upto here it is working fine')

#reading the data from data path
data_path = r"C:\Users\gitaa\Desktop\DataScienceProject\NLP_project\Multi_class_classification\train\train.csv"
df = pd.read_csv(data_path)
df_te = pd.read_csv( r"C:\Users\gitaa\Desktop\DataScienceProject\NLP_project\Multi_class_classification\test\test.csv")

#understanding the data
print(df.head())
print("\n shape of train: \n",df.shape)
print(df.info())
print('\nPrinting the value: \n')
print(df.author.value_counts(dropna=False))

#making a tfidf for the text:
'''tf = term frequency = no_of_word/totol_no_word_inDoc
for eg: in document 3: I have gold occuring 5 times and total words in that doc =41
        tf=5/41
 idf = inverse document frequency = log(totol_no_of_doc/no_of_doc_with_that_word)
 for eg: Here I have total number of documents =19579 and lets say gold word is in 234 documents  than
         idf= log(19579/234)
 Note: tf-idf lower the weight of highly occuring word and boosts the weight of low-occuring words'''

tfidf = TfidfVectorizer(stop_words='english')

features = tfidf.fit_transform(df.text).toarray()
author = df.author
print(features.shape)
#making X data and Y data
X = df['text']
Y = df['author']
#Train test split here text_size i divided into 80/20 ratio random_state we can provide 0 to any number
x_train, x_test,y_train, y_test = train_test_split(X,Y, test_size=0.2,random_state= 10)

#conutvectorizer
count_vect = CountVectorizer()
x_train_counts = count_vect.fit_transform(x_train)
x_test_counts = count_vect.fit_transform(x_test)
print('Printing the shape of countvectorizer:\n')
print(x_train_counts.shape)
print(x_test_counts.shape)

#creating a tfidf
tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
x_test_tfidf = tfidf_transformer.fit_transform(x_test_counts)
print('Printing the shape of tfidf:\n')
print(x_train_tfidf.shape)
print(x_test_tfidf.shape)

#modeling on tfidf data
print('Naive Bayes Model Fitting on TFIDF')
NB_tf = MultinomialNB()
NB_tf.fit(x_train_tfidf,y_train)

#model prediction
test_pred= NB_tf.predict(x_test_tfidf)
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
report=classification_report(y_test, test_pred)
print(report)

#modeling on tfidf data
print('Naive Bayes Model Fitting on Count vectorizer')
NB_co = MultinomialNB()
NB_co.fit(x_train_counts,y_train)

#model prediction
test_pred= NB_co.predict(x_test_counts)
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, precision_recall_curve,roc_auc_score,auc
report=classification_report(y_test, test_pred)
acc=accuracy_score(y_test, test_pred)
precision=precision_score(y_test,test_pred, pos_label=1)
print(report)
print('Accuracy: ', acc)
print('Precision score for 1 :', precision)
