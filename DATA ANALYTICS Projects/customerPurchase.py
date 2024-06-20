import pandas as pd

purchase = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Customer%20Purchase.csv')



print(purchase.head())

print(purchase.info())

print(purchase.describe())

# step 3

print(purchase.columns)

y = purchase['Purchased']

X = purchase.drop(['Customer ID', 'Purchased' ], axis=1)

X.replace({'Review':{'Poor':0,'Average':1,'Good':2}},inplace=True)
X.replace({'Education':{'School':0,'UG':1,'PG':2}},inplace=True)
X.replace({'Gender':{'Male': 0,'Female':1}},inplace=True)

print(X.head())

# step 4

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=2529)

# step 5

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

print(RandomForestClassifier())

# step6

model.fit(X_train, y_train)

print(model.fit(X_train, y_train))

# step 7

y_pred = model.predict(X_test)

print(y_pred)

# step 8

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

print(confusion_matrix(y_test,y_pred))

print(accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))




