import pandas as pd

credit = pd.read_csv("https://github.com/YBIFoundation/Dataset/raw/main/Credit%20Default.csv")

print(credit.head())
print(credit.info())
print(credit.describe())

# step 3

print(credit.columns)

y = credit['Default']

X = credit[['Income', 'Age', 'Loan', 'Loan to Income']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=2529)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

print(X_train)


# step 5

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)

print(LogisticRegression())

# step 6

model.fit(X_train, y_train)

model.intercept_

print(model.intercept_)

model.coef_

print(model.coef_)

# step 7

y_pred = model.predict(X_test)

print(y_pred)

# step 8

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

print(confusion_matrix(y_test,y_pred))

print(accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))





