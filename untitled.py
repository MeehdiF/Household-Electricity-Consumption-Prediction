# STEPS FOR MACHINE LEARNING
## Step 0: Import the necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn import linear_model
import numpy as np


## Step 1: Import Data
# readinag given csv file 
# and creating dataframe 
try:
    file = pd.read_csv('household_power_consumption.csv')
except FileNotFoundError as e:
    print(f"File Not Found! Error is [{e}]")
except NameError as e:
    print(f"File Not Found! Error is [{e}]")

file.head()



## Step 2: Clean the Data
file = file.drop(labels=["Date", "Time"] ,axis=1)
file.head(5)
print(file.isnull().sum())
file.dtypes
for col in file.columns:
    print(f"1)Before Changing the types: {file[col].describe()}")
    file[col] = file[col].replace('?', pd.NA).fillna(0).astype(float)
    print(f"2)After Changing the types: {file[col].describe()}")




## Step 3: Split the Data into Training/testing
X = np.asanyarray(file[["Global_reactive_power", "Voltage", "Global_intensity","Sub_metering_1","Sub_metering_2","Sub_metering_3"]])
y = np.asanyarray(file[["Global_active_power"]])

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=38)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


## Step 4: Create a Model
regr = linear_model.LinearRegression()


## Step 5: Train the Model
regr.fit(X_train, y_train)
# The coefficients
print ('Coefficients: ', regr.coef_)


## Step 6: Make Predictions
y_hat= regr.predict(X_test)



## Step 7: Evaluation and Improve
print("Residual sum of squares: %.2f"
      % np.mean((y_hat - y_test) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test, y_test))