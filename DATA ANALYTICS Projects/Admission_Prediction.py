
# # STEP 1
# import pandas as pd

# # STEP 2
# admission = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Admission%20Chance.csv')

# print(admission.head())
# print(admission.info())

# # print(admission.info(verbose=True))     # Full summary print karega 
# # print(admission.info(verbose=False))    # Sirf column count aur dtype ki summary print karega

# print(admission.describe())


# # STEP 3
# print(admission.columns)

# y = admission['Chance of Admit']

# X = admission.drop(['Serial No', 'Chance of Admit '], axis=1)

# # step4 split data

# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, random_state=2529)

# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# print(X_train.shape)

# step 5

# from sklearn.linear_model import LinearRegression
# model = LinearRegression()

# print(model = LinearRegression())


# # step6

# model.fit(X_train, y_train,)

# print(model.intercept_)
# print(model.coef_)

# # step 7 predict model

# y_pred = model.predict(X_test)

# print(y_pred)


# # step 8 accuracy of model

# from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

# print(mean_absolute_error(y_test, y_pred))
# print(mean_absolute_percentage_error(y_test, y_pred))
# print(mean_squared_error(y_test, y_pred))


import pandas as pd

admission = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/Admission%20Chance.csv')

print(admission.head())
print(admission.info())
print(admission.describe())


# step 3

print(admission.columns)

y = admission['Chance of Admit ']

X = admission.drop(['Serial No','Chance of Admit ' ], axis=1)

# step 4

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, random_state=2529)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

print(X_train)

# step 5 model selection

from sklearn.linear_model import LinearRegression
model = LinearRegression()

print(LinearRegression())

# step 6
model.fit(X_train, y_train)
print(model.fit(X_train, y_train))

model.intercept_
print(model.intercept_)


model.coef_
print(model.coef_)

# step 7 predict model

y_pred = model.predict(X_test)
print(type(y_pred))

# step 8 accuracy

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

mean_absolute_error(y_test, y_pred)
print(mean_absolute_error(y_test, y_pred))

print(mean_absolute_percentage_error(y_test, y_pred))   # data is 92.5% accurate


print(mean_squared_error(y_test, y_pred))




