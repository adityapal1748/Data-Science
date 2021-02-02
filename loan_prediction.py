import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


loc = r'C:\Users\dell\Desktop\aditya\loan_train.csv'
df = pd.read_csv(loc)
# print(df.head())
# print(df.describe())
# print(df.isnull().any())
# print(df.info())
# print(df.loc[0:,['LoanAmount']])
# print(df.isnull().sum())

df_loan = df.dropna()
# df_loan.info()
# print(df.isnull().sum())
# df.info()

df['Dependents'].fillna(1,inplace=True)
# df.info()
df['LoanAmount'].fillna(df.LoanAmount.mean(),inplace = True)
# df.info()
df['Loan_Amount_Term'].fillna(df.Loan_Amount_Term.mean(),inplace=True)
# df.info()
df['Credit_History'].fillna(df.Credit_History.mean(),inplace = True)
# df.info()

value_mapping = {'Male':1,'Female':0}
df['Gender'] = df['Gender'].map(value_mapping)
df['Gender'].fillna(df.Gender.mean(),inplace = True)
# print(df.isnull().sum())

df['Self_Employed'] = df['Self_Employed'].map({'Yes':1,'No':0})
df['Self_Employed'].fillna(df.Self_Employed.mean(),inplace = True)
# df.info()

df['Married'] = df['Married'].map({'Yes':1,'No':0})
df['Married'].fillna(df.Married.mean(),inplace = True)
# print(df.head(9))
# df.info()

df['Education'] = df['Education'].map({'Graduate':1,'Not Graduate':0})
# print(df.head())

Value_Mapping4 = {'Y' : 1, 'N' : 0}
df['Loan_Status'] = df['Loan_Status'].map(Value_Mapping4)

# print(df.Property_Area.unique())

df['Property_Area'] = df.Property_Area.map({'Rural':0,'Semiurban':1,'Urban':2})
# print(df.head())

X=df[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Married',
        'Gender','Education','Self_Employed','Property_Area']].values
y=df[["Loan_Status"]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = RandomForestClassifier()
model.fit(X_test,y_test)
# print(model.score(X_train,y_train))

y_pred = model.predict(X_test)
# print(y_pred)
# print(X_train)


