# Load packages.
import os, sys
from numpy import mean
from math import sqrt
import pandas as pd
df1=pd.read_csv('LengthOfStay.csv')
df1cols=list(df1.columns)
print(df1cols)
df3=pd.read_csv('MetaData_Facilities.csv')

df13=pd.merge(df1,df3,left_on='facid', right_on='Id')
df13.head()
del df13['Id']
del df13['Name']
del df13['Capacity']
## check classes balance
df13['vdate']=pd.to_datetime(df13['vdate'])
df13['daysofweek_admit']=df13['vdate'].dt.weekday_name
df13['los_numeric']=df13['lengthofstay'].astype('int')
print(df13.dtypes)
## encode string columns
from sklearn.preprocessing import LabelEncoder
le_rcount=LabelEncoder()
df13['rcount']=le_rcount.fit_transform(df13['rcount'])
le_gender=LabelEncoder()
df13['gender']=le_gender.fit_transform(df13['gender'])


le_daysofweekadmit=LabelEncoder()
df13['daysofweek_admit']=le_daysofweekadmit.fit_transform(df13['daysofweek_admit'])
print(df13.dtypes)
import numpy as np
def turn2cat(item):
    if item>=6 :
        return '>=6_Days'
    else:
        return '<6_Days'
df13['label']=df13['los_numeric'].apply(lambda x: turn2cat(x))

# df13.head()

fid=LabelEncoder()
df13['fid']=fid.fit_transform(df13['facid'])
df13['los_numeric']=df13['los_numeric'].astype(float)

df= df13[['rcount', 'gender', 'dialysisrenalendstage', 'asthma',
       'irondef', 'pneum', 'substancedependence', 'psychologicaldisordermajor',
       'depress', 'psychother', 'fibrosisandother', 'malnutrition', 'hemo',
       'hematocrit', 'neutrophils', 'sodium', 'glucose', 'bloodureanitro',
       'creatinine', 'bmi', 'pulse', 'respiration',
       'secondarydiagnosisnonicd9',  'fid', 'daysofweek_admit','los_numeric']]
df.to_csv('preprocessed.csv')
