import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.svm import SVC , LinearSVC
import os
from sklearn.decomposition import PCA
import pickle
from sklearn import *

df = pd.read_csv('AppleStore_u_description(Optional_Bonus).csv')
df2 = pd.read_csv("AppleStore_training_classification.csv")


def prepareData(df):
    # Remove package name as it's not relevant
    df = df.drop('size_bytes', axis=1)
    df = df.drop('track_name', axis=1)
    df = df.drop('id', axis=1)

    df['app_desc'] = df['app_desc'].str.strip().str.lower()
    H = 'High';
    I = 'Intermediate';
    L = 'Low'
    df2["rate"] = df2["rate"].map(lambda x: 2 if x == H else 1 if x == I else 0 if x == L else 2)
    df.insert(df.shape[1], 'rate', df2['rate'], True)
    df = df.drop(df.index[range(1500, 7197)])

    #val = preprocessing.MinMaxScaler(feature_range=(0, 1))
    #val.fit(df['rate'].values.reshape(-1, 1))
    #df = val.transform(df['rate'].values.reshape(-1, 1))

    #df = df.drop(df.index[range(4798,7197)])
    #print(df)
    return df

df = prepareData(df)

x = df['app_desc']
y = df['rate']
x, x_test, y, y_test = train_test_split(x,y, stratify=y, test_size=0.25, random_state=42)

# Vectorize text reviews to numbers ----------
vec = CountVectorizer(stop_words='english')
x = vec.fit_transform(x).toarray()
x_test = vec.transform(x_test).toarray()
# --------------------------------------------
#print(x)
min1 = 0; min2 = 0; min3 = 0

model = MultinomialNB()
model.fit(x, y)
print(model.score(x_test, y_test))


pca = PCA(n_components=30)
pca.fit(x)
xx = pca.transform(x)
x = xx
# ratio= pca.explained_variance_ratio_
xtest = pca.transform(x_test)
x_test = xtest

for i in range(0,50):
    C = 9
    model1 = OneVsRestClassifier(LinearSVC(C=C)).fit(x, y)
    acc1 = model1.score(x_test, y_test)
    acc = str(acc1 * 100)
    #print('One VS One SVC accuracy: ' + acc)
    if acc1 > min1:
        min1 = acc1
        filename = 'linearSVM1.pkl'
        pickle.dump(model1, open(filename, 'wb'))


    C = .000001
    # svm_model_linear_ovr = OneVsRestClassifier(SVC(kernel='rbf', gamma=0.4, C=1).fit(X_train, y_train))
    model2 = OneVsOneClassifier(SVC(kernel='rbf', gamma=0.4, C=C)).fit(x, y)
    acc2 = model2.score(x_test, y_test)
    #print('One VS One SVM accuracy Kernel == rbf Gaussian : ' + str(accuracy * 100))
    if acc2 > min2:
        min2 = acc2
        filename = 'rbfSVM1.pkl'
        pickle.dump(model2, open(filename, 'wb'))

    C = 1
    model3 = OneVsRestClassifier(SVC(kernel='poly', degree=3, C=C)).fit(x, y)
    acc3 = model3.score(x_test, y_test)
    #print('One VS Rest SVM accuracy Kernel == Poly: ' + str(acc3 * 100))
    if acc3 > min3:
        min3 = acc3
        filename = 'polySVM1.pkl'
        pickle.dump(model3, open(filename, 'wb'))

print('One VS One SVC accuracy: ' + str(min1))
print('One VS One SVM accuracy Kernel: rbf Gaussian : ' + str(min2))
print('One VS Rest SVM accuracy Kernel: Poly: ' + str(min3))

# Save model
#joblib.dump(model, 'model.pkl')

#sentence = 'Seems interesting to me'
#x = vec.transform(sentence).toarray()
#np.reshape(x, (1,20))
##X = x[:, :20]

#model1 = pickle.load(open("linearSVM.pkl", 'rb'))
#print(model1.score(x, '1'))
##print(model1.predict(x.shape(20)))

#model2 = pickle.load(open("rbfSVM.pkl", 'rb'))
#print(model2.predict(x))

#model3 = pickle.load(open("polySVM.pkl", 'rb'))
#print(model3.predict(x))

filename = 'model.pkl'
pickle.dump(model, open(filename, 'wb'))