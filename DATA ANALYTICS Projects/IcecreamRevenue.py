import pandas as pd

icecream = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/Ice%20Cream.csv')

print(icecream.head())
print(icecream.info())
print(icecream.describe())


# step 3
print(icecream.columns)



y = icecream['Revenue']

X = icecream[['Temperature']]

# step4

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2529)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# step 5

from sklearn.linear_model import LinearRegression
model = LinearRegression()

print(LinearRegression())




# step 6

model.fit(X_train, y_train)

print(model.intercept_)
print(model.coef_)

# print(model.__new__)

# step 7 predict



y_pred = model.predict(X_test)

print(y_pred)

# step 8

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
print(mean_absolute_percentage_error(y_test,y_pred))

print(mean_squared_error(y_test, y_pred))


