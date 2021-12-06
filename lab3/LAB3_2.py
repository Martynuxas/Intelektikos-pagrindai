from imblearn.over_sampling import SMOTE
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as ex
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.neural_network import MLPClassifier
import numpy as np
# Nuskaitom duomenis
data = pd.read_csv("Video_Games_Sales_as_at_22_Dec_2016.csv", error_bad_lines=False)

# Nustatom, kad i konsole rasytu visus irasus ir isvesties plotis butu 200
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print(data.head())

# Surandam visas kategorines reiksmes
categorical = data.select_dtypes(exclude=['int64',
'float64']).columns.to_numpy()
print('\nKategoriniai kintamieji: {0}'.format(categorical))

# Surandam visas tolydines reiksmes
continous = data.select_dtypes(include=['int64',
'float64']).columns.to_numpy()
print('\nTolydiniai kintamieji: {0}'.format(continous))
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#POKYCIA
data.drop(categorical)
data.drop(columns =['NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales','User_Count'],inplace=True)

print("POKYCIAI PO POKYCIU {0}".format(data['Critic_Score']))
oversample = SMOTE()
X, y = oversample.fit_resample(data[data.columns[1:]], data[data.columns[0]])
usampled_df = X.assign(Churn = y)
usampled_df = usampled_df.drop(columns=usampled_df.columns[15:-1])

fig = plt.figure(figsize =(10, 7))

fig = ex.pie(usampled_df,names='Churn',title='Išlikusių ir iškritusių klientųskaičius',hole=0.33).show()

X_features = ['User_Score','Critic_Count']
X = usampled_df[X_features]
y = usampled_df['Churn']

train_x,test_x,train_y,test_y = train_test_split(X,y,random_state=42)
clf = MLPClassifier(random_state=42, max_iter=300, activation='logistic',
solver='adam', learning_rate_init=0.001).fit(train_x, train_y)
predictions = clf.predict(test_x)
score = clf.score(test_x, test_y)
print("Predictions {0}".format(predictions))
print("Score {0}".format(score))

splitX = np.array_split(test_x, 10)
splitY = np.array_split(test_y, 10)

for i in range(len(splitX)):
    predictions = clf.predict(splitX[i])
    score = clf.score(splitX[i], splitY[i])
    print("Predictions {0}".format(predictions))
    print("Score {0}".format(score))