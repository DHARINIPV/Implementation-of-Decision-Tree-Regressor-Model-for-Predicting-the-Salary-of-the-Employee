# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2.
5. Print the obtained values.

## Program:
```
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Dharini PV
RegisterNumber:  212222240024
```
```python
import pandas as pd
data = pd.read_csv("/content/Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()

x = data[["Position","Level"]]
y = data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse

r2 = metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```
## Output:

## Initial dataset:

![image](https://github.com/DHARINIPV/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119400845/612e98f3-6d70-421c-92df-77f3afeb97ef)

## Data Info:

![image](https://github.com/DHARINIPV/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119400845/a3aae6c3-a44a-4c3f-991c-adabbc57d73e)

## Optimiztion of null values:

![image](https://github.com/DHARINIPV/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119400845/c63d3b42-b804-49b2-a2f9-62ccb9d37574)

## Converting string literals to numerical values using label encoder:

![image](https://github.com/DHARINIPV/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119400845/f0f3e744-d71b-441c-8763-a3561597d454)

## Mean Squared Error:

![image](https://github.com/DHARINIPV/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119400845/6126edb0-f3f0-4a72-8c5c-3adbd02cf7c0)

## R2 (variance):

![image](https://github.com/DHARINIPV/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119400845/f28c7c33-47a1-4c85-bdf3-b1fee31dbd74)

## Prediction:

![image](https://github.com/DHARINIPV/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119400845/c5c526c4-fb03-49d9-95cc-1372a81e7ad3)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
