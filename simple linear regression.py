#step 1
import pandas as pd
#for step 2
from sklearn.model_selection import train_test_split
#for step 5
from sklearn.linear_model import LinearRegression
# for step 8
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
#step 2
salary = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/Salary%20Data.csv')
#print(salary.info())
#step 3 : define (y) and (X)
#step 4 : trai test split
print(salary.columns)
y = salary['Salary']
X = salary[['Experience Years']]
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=2529)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
# step 5 : select model
model = LinearRegression()
# step 6 : train or fit model
model.fit(X_train,y_train)
print(model)
print("intercept is :",model.intercept_)
print("coefficent is :",model.coef_)
# step 7: predict model
y_pred = model.predict(X_test)
print(y_pred)
# step 8 : model accuracy
print("mean abs error is :",mean_absolute_error(y_test,y_pred))
print("mean abs % error is :",mean_absolute_percentage_error(y_test,y_pred))
print("mean square error is : ",mean_squared_error(y_test,y_pred))




