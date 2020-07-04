########################
#EE559 Final Project   #
#Student: Ming-Jui Lee #
#ID: 4673484700        #
#Data set: Adult       #
########################
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import dataInformation as dI
import dataPreprocessing as dP
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn import svm
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

## read csv file and load Training/Testing data
train_raw = pd.read_csv('adult.train_SMALLER.csv')
test_raw = pd.read_csv('adult.test_SMALLER.csv')
#test_raw = pd.read_csv('adult_test.csv')

## get basic data infomation
print("Training data basic information: ")
dI.dataInformation(train_raw)
print("Testing data basic information: ")
dI.dataInformation(test_raw)

## Data Preprocessing
(train_pro, train_label) = dP.dataPreprocessing(train_raw)
train_nb = pd.DataFrame(train_pro)
train_nb = train_nb.drop(['fnlwgt'], axis = 1)
train_features = train_pro
numerical_cols = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
scaler = MinMaxScaler()
train_features[numerical_cols] = scaler.fit_transform(train_features[numerical_cols]) # normalized
train_features = pd.DataFrame(train_features)
print("Training data after Preprocessing\n")

(test_pro, test_label) = dP.dataPreprocessing(test_raw)
test_nb = pd.DataFrame(test_pro)
test_nb = test_nb.drop(['fnlwgt'], axis = 1)
test_features = test_pro
test_features[numerical_cols] = scaler.transform(test_features[numerical_cols]) # can only us normalizing factor from training
test_features = pd.DataFrame(test_features)
print("Testing data after Preprocessing")

## Training
# Method: Perceptron (without feature selection)
perceptron_model = Perceptron(tol=1e-3, random_state=0)    # Object declaration
perceptron_model.fit(train_features, train_label)   # Training 
train_predict_per = perceptron_model.predict(train_features)
perceptron_train_acc = accuracy_score(train_label, train_predict_per) 
print("Before feature selection")
print("Perceptron Learning:")
print("Classification accuracy of classifier on the training data is {}%" .format(perceptron_train_acc * 100))
test_predict_per = perceptron_model.predict(test_features)
perceptron_test_acc = accuracy_score(test_label, test_predict_per) 
print("Classification accuracy of classifier on the testing data is {}%" .format(perceptron_test_acc * 100))

## Feature Selection
train_numerical = train_features[numerical_cols]
selector = Perceptron(tol=1e-3, random_state=0) 
selector.fit(train_numerical, train_label) 
numerical_predict = selector.predict(train_numerical)
train_numerical_acc = accuracy_score(train_label, numerical_predict) 
weight = selector.coef_
print("Training accuracy {}%" .format(train_numerical_acc))
print("Weights for ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week'] are:\n", weight)
num_label = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
weight_abs = [weight[0][0], weight[0][1], weight[0][2], weight[0][3], weight[0][4], weight[0][5]]
plt.bar(num_label, weight_abs, alpha=0.9, width = 0.35, facecolor = 'lightskyblue', edgecolor = 'white', label='weight', lw=1)
plt.legend(loc="upper left")
plt.show()
train_new = train_features.drop(['fnlwgt'], axis = 1)
test_new = test_features.drop(['fnlwgt'], axis = 1)

# Method: Perceptron (with feature selection)
perceptron_model = Perceptron(tol=1e-3, random_state=0)    # Object declaration
perceptron_model.fit(train_new, train_label)   # Training 
train_predict_per = perceptron_model.predict(train_new)
perceptron_train_acc = accuracy_score(train_label, train_predict_per) 
print("After feature selection")
print("Perceptron Learning:")
print("Classification accuracy of classifier on the training data is {}" .format(perceptron_train_acc))
test_predict_per = perceptron_model.predict(test_new)
perceptron_test_acc = accuracy_score(test_label, test_predict_per) 
print("Classification accuracy of classifier on the testing data is {}" .format(perceptron_test_acc))
perceptron_cm = confusion_matrix(test_predict_per, test_label)
print(perceptron_cm)
plt.imshow(perceptron_cm)
labels = ['negative', 'positive']
xlocations = np.array(range(len(labels)))
plt.xticks(xlocations, labels, rotation=0)
plt.yticks(xlocations, labels)
plt.title('Confusion matrix of the Perceptron classifier')
plt.xlabel('True Label')
plt.ylabel('Predict Label')
plt.colorbar()
thresh = perceptron_cm.max()
for i, j in itertools.product(range(perceptron_cm.shape[0]), range(perceptron_cm.shape[1])):
    plt.text(j, i, perceptron_cm[i, j], horizontalalignment="center", 
    color="white" if perceptron_cm[i, j] > thresh else "black")
plt.show()
print("Perceptron\n", classification_report(test_label, test_predict_per))
per_score = perceptron_model.decision_function(test_new)
per_auc = roc_auc_score(test_label, per_score)
print("Perceptron AUC: %.3f" %(per_auc))

'''
# SVM: Use cross-validation to select gamma and C
cv = StratifiedKFold(n_splits=5, shuffle=True)
C_range = np.logspace(-2, 1, 15)
gamma_range = np.logspace(-2, 1, 15)
ACC = np.zeros([15,15])
for i in range(len(gamma_range)):
    for j in range(len(C_range)):
        clf = svm.SVC(kernel='rbf', gamma=gamma_range[i], C=C_range[j])
        acc = cross_val_score(clf, train_new, train_label, cv = cv, scoring='accuracy')
        mean_acc = np.mean(acc)
        ACC[i][j] = mean_acc
        print("i j", i, j)
max_acc= np.max(ACC)
max_acc_index=np.where(ACC==np.max(ACC))
print("svm cv best acc", max_acc)
print("choosen gamma is", gamma_range[int(max_acc_index[0])])
print("choosen C is", C_range[int(max_acc_index[1])])
'''

# Model: SVM (with feature selection)
SVC_clf = svm.SVC(gamma = 0.0439397, kernel='rbf', C = 10.0)
SVC_clf.fit(train_new, train_label)
pred_train_SVC = SVC_clf.predict(train_new)
pred_test_SVC = SVC_clf.predict(test_new)
train_acc_SVC = accuracy_score(train_label, pred_train_SVC)
test_acc_SVC = accuracy_score(test_label, pred_test_SVC)
print("\nAccuracy of Support Vector Classification on training data is {}" .format(train_acc_SVC))
print("Accuracy of Support Vector Classification on testing data is {}" .format(test_acc_SVC))
svm_cm = confusion_matrix(pred_test_SVC, test_label)
print(svm_cm)
plt.imshow(svm_cm)
labels = ['negative', 'positive']
xlocations = np.array(range(len(labels)))
plt.xticks(xlocations, labels, rotation=0)
plt.yticks(xlocations, labels)
plt.title('Confusion matrix of the SVM classifier')
plt.xlabel('True Label')
plt.ylabel('Predict Label')
plt.colorbar()
thresh = svm_cm.max()
for i, j in itertools.product(range(svm_cm.shape[0]), range(svm_cm.shape[1])):
    plt.text(j, i, svm_cm[i, j], horizontalalignment="center", 
    color="white" if svm_cm[i, j] > thresh else "black")
plt.show()
print("SVM\n", classification_report(test_label, pred_test_SVC))
svm_score = SVC_clf.decision_function(test_new)
svm_auc = roc_auc_score(test_label, svm_score)
print("SVM AUC: %.3f" %(svm_auc))

'''
# KNN: use cross-validation to choose k
cv = StratifiedKFold(n_splits=5, shuffle=True)
cv_score = np.zeros(10)
for i in range(10):
    print("k", i+1)
    k_neigh = KNeighborsClassifier(n_neighbors=i+1)
    acc = cross_val_score(k_neigh, train_new, train_label, cv = cv, scoring='accuracy')
    cv_score[i] = np.mean(acc)
print("cv score", cv_score)
max_acc= np.max(cv_score)
max_acc_index=np.where(cv_score==np.max(cv_score))
k_best = int(max_acc_index[0]) + 1
print("choosen k is", k_best)
'''
# Model: KNN (with feature selection)
k_neigh = KNeighborsClassifier(n_neighbors=9)
k_neigh.fit(train_new, train_label)
k_train_pre = k_neigh.predict(train_new)
kNN_train_acc = accuracy_score(train_label, k_train_pre) 
print("\nClassifier implementing the k-nearest neighbors algorithm\n")
print("Classification accuracy of KNN classifier on the training data is {}" .format(kNN_train_acc))
k_test_pre = k_neigh.predict(test_new)
kNN_test_acc = accuracy_score(test_label, k_test_pre) 
print("Classification accuracy of KNN classifier on the testing data is {}" .format(kNN_test_acc))
kNN_cm = confusion_matrix(k_test_pre, test_label)
print(kNN_cm)
plt.imshow(kNN_cm)
labels = ['negative', 'positive']
xlocations = np.array(range(len(labels)))
plt.xticks(xlocations, labels, rotation=0)
plt.yticks(xlocations, labels)
plt.title('Confusion matrix of the k-NN classifier')
plt.xlabel('True Label')
plt.ylabel('Predict Label')
plt.colorbar()
thresh = kNN_cm.max()
for i, j in itertools.product(range(kNN_cm.shape[0]), range(kNN_cm.shape[1])):
    plt.text(j, i, kNN_cm[i, j], horizontalalignment="center", 
    color="white" if kNN_cm[i, j] > thresh else "black")
plt.show()
print("k-NN\n", classification_report(test_label, k_test_pre))
knn_score = k_neigh.predict_proba(test_new)
knn_score = knn_score[:,1]
knn_auc = roc_auc_score(test_label, knn_score)
print("k-NN AUC: %.3f" %(knn_auc))

# Naive Bayes
gnb = GaussianNB()
gnb.fit(train_nb, train_label)
train_predict_gnb = gnb.predict(train_nb)
test_predict_gnb = gnb.predict(test_nb)
gnb_train_acc = accuracy_score(train_label, train_predict_gnb)
gnb_test_acc = accuracy_score(test_label, test_predict_gnb)
print("\nApplying Naive Bayes clssifier")
print("Classification accuracy of Naive Bayes classifier on training data: {}" .format(gnb_train_acc))
print("Classification accuracy of Naive Bayes classifier on testing data: {}" .format(gnb_test_acc))
nb_cm = confusion_matrix(test_predict_gnb, test_label)
print(nb_cm)
plt.imshow(nb_cm)
labels = ['negative', 'positive']
xlocations = np.array(range(len(labels)))
plt.xticks(xlocations, labels, rotation=0)
plt.yticks(xlocations, labels)
plt.title('Confusion matrix of Naive Bayes classifier')
plt.xlabel('True Label')
plt.ylabel('Predict Label')
plt.colorbar()
thresh = nb_cm.max()
for i, j in itertools.product(range(nb_cm.shape[0]), range(nb_cm.shape[1])):
    plt.text(j, i, nb_cm[i, j], horizontalalignment="center", 
    color="white" if nb_cm[i, j] > thresh else "black")
plt.show()
print("Naive Bayes\n", classification_report(test_label, test_predict_gnb))
nb_score = gnb.predict_proba(test_new)
nb_score = nb_score[:,1]
nb_auc = roc_auc_score(test_label, nb_score)
print("Naive Bayes AUC: %.3f" %(nb_auc))







