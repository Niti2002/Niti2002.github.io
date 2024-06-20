# step1
import pandas as pd

# step2

fish = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Fish.csv')

print(fish.head())

print(fish.info())

print(fish.describe())

# step 3

print(fish.columns)

y= fish['Weight']

X = fish.drop(['Species', 'Weight'], axis=1)

# step 4

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, random_state=2529)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

print(X_train)

# step5

from sklearn.linear_model import LinearRegression
model = LinearRegression()

print(LinearRegression())

# step6

model.fit(X_train,y_train)

print(model.fit(X_train,y_train))

model.intercept_

print(model.intercept_)

print(model.coef_)

# print(model.rank_)

# step 7

y_pred = model.predict(X_test)

print(y_pred)

# step 8

from sklearn.metrics import mean_absolute_error, r2_score

print(mean_absolute_error(y_test, y_pred))

print(r2_score(y_pred, y_test))

