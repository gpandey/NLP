import numpy as np
import pandas as pd
#ignore warning
import warnings
warnings.filterwarnings('ignore')

from sklearn import model_selection, preprocessing, linear_model,naive_bayes,metrics,svm
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,TfidfTransformer
from sklearn import decomposition,ensemble
from sklearn.feature_selection import chi2
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

import sys
#sys.exit(0)  # to run upt to this and skip code below from running

#Label encoding for target variable
df['author_label'] = df['author'].factorize()[0]

from io import StringIO
# Firstly giving the label and actual categories of target
author_label_df = df[['author','author_label']].drop_duplicates().sort_values('author_label')
author_to_label = dict(author_label_df.values)
label_to_author = dict(author_label_df[['author_label','author']].values)
print('\ text: \n',df.head())

tfidf = TfidfVectorizer(stop_words= 'english')

#converting the into tfidf vector
features = tfidf.fit_transform(df.text).toarray()
labels = df.author_label
print(features.shape)

# looking for a correlation between two text
'''N = 2
for author, author_label in sorted(author_to_label.items()):
    features_chi2 = chi2(features, labels ==author_label)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' '))==1]
    bigrams = [v for v in feature_names if len(v.split(' '))==2]
    print("# '{}':" .format(author))
    print("  . Most Correlated unigrams:\n     .{}".format('\n           . '.join(unigrams[-N:])))
    print("  . Most Correlated bigrams:\n     .{}".format('\n           . ' .join(bigrams[-N:])))
 '''
 #Modeling
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
model = LinearSVC()
x_train, x_test,y_train, y_test,indices_train,indices_test = train_test_split(features, labels, df.index,
                                                                              test_size=0.33,random_state=5)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred, target_names= df['author'].unique()))

#saving model for API use: serialization(or pickling)
from sklearn.externals import joblib
import pickle
model.save('LinearSVC.h5')
joblib.dump(model,'LinearSVC.pkl')
pickle.dump(model,open('linSVC.pkl','wb'))

#loading the saved model
LinSVC = joblib.load('LinearSVC.pkl')













