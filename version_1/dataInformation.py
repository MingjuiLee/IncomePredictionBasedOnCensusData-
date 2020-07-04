import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

def dataInformation(data):
    # columns label
    data.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 
    'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'wage_class']
    print(data.info())

    # make training and testing data label have same notation
    data['wage_class'] = data['wage_class'].replace({' <=50K.': ' <=50K', ' >50K.': ' >50K'})

    # Numerical data
    print("Numerical data:\n", data.describe())
    # Categorical data
    print("Categorical data:\n", data.describe(include=['O']))

    '''
    for column in data.columns:
        print(column)
        if data.dtypes[column] == np.object: # Categorical data
            print(data[column].value_counts())
        else:
            print(data[column].value_counts())
        print('\n')

    print(pd.crosstab(data['wage_class'], data['workclass']))
    print(pd.crosstab(data['wage_class'], data['education']))
    print(pd.crosstab(data['wage_class'], data['marital_status']))
    print(pd.crosstab(data['wage_class'], data['occupation']))    
    plt.hist(data['age'], bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel("age")
    plt.ylabel("Numbers")
    plt.title("age")
    '''
    #data['workclass'].value_counts().plot(kind='bar', y = 'Numbers', title = 'workclass')
    #data['occupation'].value_counts().plot(kind='bar', y = 'Numbers', title = 'occupation')
    #data['native_country'].value_counts().plot(kind='bar', y = 'Numbers', title = 'native_country')
    #plt.show()