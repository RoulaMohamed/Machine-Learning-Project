import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import Classify

# --------------------------------------------------------------------------
#Loading data
data = pd.read_csv('AppleStore_training_classification.csv')
#pre proccessing
data.dropna(how='any', inplace=True)
data['cont_rating'] = [i.replace('+','') for i in data['cont_rating']]
data['cont_rating'] = data['cont_rating'].astype(int)



# --------------------------------------------------------------------------

# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')
# passing bridge-types-cat column (label encoded values of bridge_types)
enc_array = enc.fit_transform(data[['prime_genre']]).toarray()

X=data.iloc[:,2:15] #Features
X.pop('prime_genre')
X.pop('currency')
X.pop('ver')
Y=data['user_rating'] #Label
Xnp = X.to_numpy()
#feature scaling
for c in range(10) :
    m1 = np.amin(Xnp[:,c])
    m2 = np.amax(Xnp[:,c])
    Xnp[:,c] = Xnp[:,c] - m1
    Xnp[:,c] = Xnp[:,c] / (m2 - m1)

Xnp = np.concatenate((Xnp,enc_array),1)

X_train, X_test, y_train, y_test = train_test_split(Xnp, Y, test_size = 0.20,shuffle=True)

# Train Models:
# -------------------------------------------------------------
classify = Classify(X_train, y_train, X_test, y_test) #--------
# -------------------------------------------------------------

data = data.iloc[:,:]
#Get the correlation between the features
corr = data.corr()
#Top 50% Correlation training features with the Value
top_feature = corr.index[abs(corr['user_rating']>0)]
#Correlation plot
plt.subplots(figsize=(12, 8))
top_corr = data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()


cls = linear_model.LinearRegression()
cls.fit(X_train,y_train)
prediction= cls.predict(X_test)

print('Co-efficient of linear regression',cls.coef_)
print('Intercept of linear regression model',cls.intercept_)
print('Mean Square Error', metrics.mean_squared_error(np.asarray(y_test), prediction))

true_app_value=np.asarray(y_test)[0]
predicted_app_value=prediction[0]
print('True value for the first app in the test set is : ' + str(true_app_value))
print('Predicted value for the first app in the test set is : ' + str(predicted_app_value))