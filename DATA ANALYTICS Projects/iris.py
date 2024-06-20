# step1
import pandas as pd

# step 2
iris = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/IRIS.csv')



print(iris.head())

print(iris.info())

print(iris.describe())

# step 3 define x and y

print(iris.columns)

y = iris['species']

X = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

# step 4 split data

from sklearn.model_selection import train_test_split

# by default this model divide data to 75% and 25% both train ans split.
X_train, X_test, y_train, y_test = train_test_split(X,y)

# for row column see

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# (112, 4) (38, 4) (112,) (38,) output X

# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

# but  we are change the data % instead of 75 to 80% by using (train_size=0.8) also change the by default change the number of X_train modle by using(random state=2529)

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.8, random_state=2529)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape) # (120, 4) (30, 4) (120,) (30,) output 

print(X_train)

# step 5 model selecction

from sklearn.linear_model import LogisticRegression
model =LogisticRegression(max_iter=500)
print(LogisticRegression())

# step 6 train X and y by fit model

model.fit(X_train,y_train)

model.intercept_
print(model.intercept_)

model.coef_
print(model.coef_)

# Step 7 : predict model
y_pred = model.predict(X_test)
print(y_pred)
# print(model.predict(X_test))

# step 8 accuracy of data

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test,y_pred))

from sklearn.metrics import accuracy_score, classification_report

print(accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))





