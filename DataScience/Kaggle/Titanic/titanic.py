'''
October 2024
Kaggle Titanic
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df_train.sample(10)
df_train.shape
df_train.info()
#Duplicate Values
df_train.duplicated().sum()
#Null values
df_train.isnull().sum().sort_values(ascending = False)
#Unique values
df_train.nunique()
#Age values
age = df_train["Age"]
df_age = pd.DataFrame(age)
df_age.sort_values(by="Age")

categoric_columns = df_train.select_dtypes(include=['object']).columns
for column in categoric_columns:
    if df_train[column].nunique() <= 10:
        print(f"{column}:{df_train[column].unique()}")

numerical_columns = df_train.select_dtypes(include=['int64', 'float64']).columns
for column in numerical_columns:
    if df_train[column].nunique() <= 10:
        print(f"{column}:{df_train[column].unique()}")

'''
Here we can see some interesting things:
The data set has the Passenger data like its ID, name, Pclass, sex, and if the passenger survived.
Values:
- PassengerID: Unique for each passenger
- Name: Unique for each passenger
- Sex: Male/Female
- Survived: 1/0 (Yes or Not boolean value)
- PClass: 1-3 (Class 1, 2 or 3)
- Age: (19-88)
- Embarked: S, C or Q.
.
.
.
Passengers: 891
Items like PassengerID, Name, Ticket, Fare and Cabin doesn't  doesn't contribute much to the prediction,
so it will be better to omit them from the model.
'''
#Graphic Data Exploration
#'Survived'
sns.countplot(x='Survived', data=df_train)
#'Sex'
sns.barplot(x='Sex', y='Survived', data = df_train)
'''
Analysis:
Approximately 19% of men survived, while more than 70% of women survived.
'''
#Removing not important to the model
df_train=df_train.drop(columns=["PassengerID", "Name", "Ticket", "Fare", "Cabin"])

#Predicters and target separation
X = df_train.drop(columns=["Survived"], axis=1)
Y = df_train.Survived
#
s = (X.dtypes == 'object')
object_column=list(s[s].index)
ordinal_encoder = OrdinalEncoder
X[object_column] = ordinal_encoder.fit_transform(X[object_column])

imputer=SimpleImputer()
x_transformer=pd.DataFrame(imputer.fit_transform(X))
x_transformer.columns=X.columns
x_transformer.isnull().sum()

#Model
model = RandomForestClassifier()
model.fit(x_transformer, Y)

df_test[object_column]=ordinal_encoder.fit_transform(df_test[object_column])
df_test_transformer=pd.DataFrame(imputer.transform(df_test))
df_test_transformer.columns=df_test_columns

predictions = model.predict(df_test_transformer)
