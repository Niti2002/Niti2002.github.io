import pandas as pd

cancer = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/Cancer.csv')

print(cancer.head())
print(cancer.info())
print(cancer.describe())


print(cancer.columns)

y = cancer['diagnosis']

X = cancer.drop(['id','diagnosis','Unnamed: 32'],axis=1)


# step 4

from sklearn.model_selection import train_test_split

X_train, X_test, y_train,  y_test = train_test_split(X,y, train_size=0.7,  random_state=2529)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

 
# step5

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=5000)

print(LogisticRegression())

# step6

model.fit(X_train, y_train)

print(model.fit(X_train, y_train))

model.intercept_
print(model.intercept_)

model.coef_
print(model.coef_)


# step 7 predict model

y_pred = model.predict(X_test)

print(y_pred)

# STEP 8

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

print(confusion_matrix(y_test, y_pred))

print(accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))