import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def dataPreprocessing(data):
    data['occupation'] = data['occupation'].replace(' Armed-Forces', ' ?')      #Test data does not include Armed-Forces
    data['native_country'] = data['native_country'].replace(' Hungary', ' ?')   #Because Test data does not include
    data['native_country'] = data['native_country'].replace(' Scotland', ' ?')  #these two countries
    
    '''
    ## feature: age
    bin = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    data['age'] = pd.cut(data['age'], bin, labels = [1, 2, 3, 4, 5, 6, 7, 8, 9], right=False)
    data['age'] = data['age'].astype('float64')
    '''
    '''
    # Missing data: remove data points with ' ?'
    data = data.replace(' ?', np.nan) # replace ' ?' with NaN
    data = data.dropna()
    
    #data['workclass'] = data['workclass'].replace(' ?', ' Private')
    #data['occupation'] = data['occupation'].replace(' ?', ' Craft-repair')
    #data['native_country'] = data['native_country'].replace(' ?', ' United-States')
    
    ## feature: workclass
    # combine Without-pay and Never-worked into Not-work
    data['workclass'] = data['workclass'].replace(' Without-pay', ' Not-work')
    data['workclass'] = data['workclass'].replace(' Never-worked', ' Not-work')
    # combile Self-emp-inc with Self-emp-not-inc as Self-employed **need consider
    data['workclass'] = data['workclass'].replace(' Self-emp-not-inc', ' Self-employed')
    data['workclass'] = data['workclass'].replace(' Self-emp-inc', ' Self-employed')
    # combine Local-gov and State-gov, assume they have same level wage
    data['workclass'] = data['workclass'].replace(' Local-gov', ' Local_State-gov')
    data['workclass'] = data['workclass'].replace(' State-gov', ' Local_State-gov')
    
    
    ## feature: education    
    # combine preschool with 1-12th grades
    data['education'] = data['education'].replace(' 1st-4th', ' Pre-12th')
    data['education'] = data['education'].replace(' 5th-6th', ' Pre-12th')
    data['education'] = data['education'].replace(' 7th-8th', ' Pre-12th')
    data['education'] = data['education'].replace(' 9th', ' Pre-12th')
    data['education'] = data['education'].replace(' 10th', ' Pre-12th')
    data['education'] = data['education'].replace(' 11th', ' Pre-12th')
    data['education'] = data['education'].replace(' 12th', ' Pre-12th')
    data['education'] = data['education'].replace(' Preschool', ' Pre-12th')
    
    ## feature: marital_status
    # combine Married-civ-spouse and Married-AF-spouse to Married
    data['marital_status'] = data['marital_status'].replace(' Married-civ-spouse', ' Married')
    data['marital_status'] = data['marital_status'].replace(' Married-AF-spouse', ' Married')
    data['marital_status'] = data['marital_status'].replace(' Divorced', ' Not-Married')
    data['marital_status'] = data['marital_status'].replace(' Separated', ' Not-Married')
    data['marital_status'] = data['marital_status'].replace(' Married-spouse-absent', ' Not-Married')
    
    ## feature: occupation
    # MOG-A Professional And Technical Occupations, Prof_Tech
    data['occupation'] = data['occupation'].replace(' Prof-specialty', ' Prof_Tech')
    data['occupation'] = data['occupation'].replace(' Tech-support', ' Prof_Tech')  
    # MOG-B Executive, Administrative, And Managerial Occupations
    #data['occupation'] = data['occupation'].replace(' Exec-managerial', ' MOG-B')
    # MOG-C Sales Occupations
    #data['occupation'] = data['occupation'].replace(' Sales', ' MOG-C')
    # MOG-D Administrative Support Occupations, Including Clerical
    #data['occupation'] = data['occupation'].replace(' Adm-clerical', ' MOG-D')
    # MOG-E Precision Production, Craft, And Repair Occupations
    #data['occupation'] = data['occupation'].replace(' Craft-repair', ' MOG-E')
    # MOG-F Machine Operators, Assemblers, And Inspectors
    #data['occupation'] = data['occupation'].replace(' Machine-op-inspct', ' MOG-F')
    # MOG-G Transportation And Material Moving Occupations
    #data['occupation'] = data['occupation'].replace(' Transport-moving', ' MOG-G')
    # MOG-H Handlers, Equipment Cleaners, Helpers, And Laborers
    data['occupation'] = data['occupation'].replace(' Handlers-cleaners', ' Handler_Farm-fish')
    data['occupation'] = data['occupation'].replace(' Farming-fishing', ' Handler_Farm-fish')
    # Military
    #data['occupation'] = data['occupation'].replace(' Armed-Forces', ' Military') # very small numbers
    data['occupation'] = data['occupation'].replace(' Armed-Forces', ' ?')
    # Service except Private HouseholdS
    data['occupation'] = data['occupation'].replace(' Protective-serv', ' Service')
    data['occupation'] = data['occupation'].replace(' Other-service', ' Service')
    
    
    ## feature: native_country
    data['native_country'] = data['native_country'].replace(' Canada', ' High-income')
    data['native_country'] = data['native_country'].replace(' China', ' Upper-middle-income')
    data['native_country'] = data['native_country'].replace(' Columbia', ' Upper-middle-income')
    data['native_country'] = data['native_country'].replace(' Cuba', ' Upper-middle-income')
    data['native_country'] = data['native_country'].replace(' Dominican-Republic', ' Upper-middle-income')
    data['native_country'] = data['native_country'].replace(' Ecuador', ' Upper-middle-income')
    data['native_country'] = data['native_country'].replace(' El-Salvador', ' Lower-middle-income')
    data['native_country'] = data['native_country'].replace(' England', ' High-income')
    data['native_country'] = data['native_country'].replace(' France', ' High-income')
    data['native_country'] = data['native_country'].replace(' Germany', ' High-income')
    data['native_country'] = data['native_country'].replace(' Greece', ' High-income')
    data['native_country'] = data['native_country'].replace(' ColGuatemala', ' Upper-middle-income')
    data['native_country'] = data['native_country'].replace(' Haiti', ' Low-income')
    data['native_country'] = data['native_country'].replace(' Honduras', ' Lower-middle-income')
    data['native_country'] = data['native_country'].replace(' Hong', ' High-income')
    data['native_country'] = data['native_country'].replace(' Hungary', ' High-income')
    data['native_country'] = data['native_country'].replace(' India', ' Lower-middle-income')
    data['native_country'] = data['native_country'].replace(' Iran', ' Upper-middle-income')
    data['native_country'] = data['native_country'].replace(' Ireland', ' High-income')
    data['native_country'] = data['native_country'].replace(' Italy', ' High-income')
    data['native_country'] = data['native_country'].replace(' Jamaica', ' Upper-middle-income')
    data['native_country'] = data['native_country'].replace(' Japan', ' High-income')
    data['native_country'] = data['native_country'].replace(' Laos', ' Lower-middle-income')
    data['native_country'] = data['native_country'].replace(' Mexico', ' Upper-middle-income')
    data['native_country'] = data['native_country'].replace(' Nicaragua', ' Lower-middle-income')
    data['native_country'] = data['native_country'].replace(' Outlying-US(Guam-USVI-etc)', ' ?')
    data['native_country'] = data['native_country'].replace(' Peru', ' Upper-middle-income')
    data['native_country'] = data['native_country'].replace(' Philippines', ' Lower-middle-income')
    data['native_country'] = data['native_country'].replace(' Poland', ' High-income')
    data['native_country'] = data['native_country'].replace(' Portugal', ' High-income')
    data['native_country'] = data['native_country'].replace(' Puerto-Rico', ' High-income')
    data['native_country'] = data['native_country'].replace(' Scotland', ' High-income')
    data['native_country'] = data['native_country'].replace(' South', ' High-income')
    data['native_country'] = data['native_country'].replace(' Taiwan', ' High-income')
    data['native_country'] = data['native_country'].replace(' Thailand', ' Upper-middle-income')
    data['native_country'] = data['native_country'].replace(' Trinadad&Tobago', ' High-income')
    data['native_country'] = data['native_country'].replace(' United-States', ' High-income')
    data['native_country'] = data['native_country'].replace(' Vietnam', ' Lower-middle-income')
    data['native_country'] = data['native_country'].replace(' Yugoslavia', ' Upper-middle-income')
    data['native_country'] = data['native_country'].replace(' Cambodia', ' Lower-middle-income')
    '''
    # Use LabelEncoder to replace <=50K and >50K with label: 0 and 1
    label_enc = LabelEncoder()
    data['wage_class'] = label_enc.fit_transform(data['wage_class'])    # label
    label = data['wage_class'].values

    # One-Hot Coding
    std_features = data.drop(['wage_class', 'education'], axis = 1)
    #std_features = data.drop(['wage_class'], axis = 1)
    pd.set_option('display.max_columns', None)
    #categorical_cols = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
    features = pd.DataFrame(std_features)
    features = pd.get_dummies(features)
    
    return features, label