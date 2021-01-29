# adaboost_vs_svm
Understanding the similarity and difference between Adaboost M1 and Linear Support Vector Machine (SVM) classifiers


* [linear_svm.py](/code/linear_svm.py): Linear SVM was implemented using **sklearn** library and used to classify given datasets [dataset1](/data/dataset1.csv) and [dataset2](/data/dataset2.csv). Decision boundaries were plotted with different values of "c" to better understand underlying mechanism in linear svms.


* [adaboost.py](/code/adaboost.py): Adaboost-M1 classifier based on linear SVM was implemented by myself without any library to classify data belong to two classes [class A](/data/classA.csv) and [class B](/data/classB.csv). 10-times-10-fold cross validation was used to show that mean and variance of accuracy improved comparing to only linear svm. 




