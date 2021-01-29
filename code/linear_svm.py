# This script contains code for Linear SVMs
# Author: Sami Alperen Akgun
# Email: sami.alperen.akgun@gmail.com

import os
import numpy as np
from matplotlib import pyplot as plt 
from sklearn import svm #svm library from sklearn Python module

def read_data(data_path, filename,feature_number):
    """
        This function reads the data from data_path/filename
        WARNING: This function assumes that features of data
        is separated by commas in the file
        Input: data_path --> The full directory path of data
        filename --> name of the file (With extension)
        feature_number --> Total feature number in the data
        Output: X --> numpy array that contains feature values
        size(X) --> sample size x feature number
        Y --> numpy array that contains labels
        size(Y) --> sample size x 1
    """

    with open(data_path + "/" + filename, 'r', encoding='utf-8-sig') as f: 
        X = np.genfromtxt(f, delimiter=',')[:,0:feature_number]


    # Last column of datafile contains output labels
    Y = np.genfromtxt(data_path + "/" + filename,delimiter=",")[:,feature_number]
    Y = Y.reshape(X.shape[0])

    return X,Y


def plot_svc_decision_function(model, ax=None, plot_support=True):
    """ Obtained from https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html"""
    """ It is obtained in order to plot margins """
    """ Otherwise, one can see my own version below to plot decision boundaries without margins!"""
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim() 
    ylim = ax.get_ylim() 
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    #ax.set_xlim(xlim)
    #ax.set_ylim(ylim)

def main():
    """
        This is the main function of this script
    """

    ##### 1-) Load data
    # If you run the code from pattern_recognition_assignment3 path, uncomment below
    data_dir = os.getcwd() + '/data' 
    # If you run the code from code directory, uncomment below
    #data_path = os.getcwd() +  ".." / "data"/
    X_dset1, Y_dset1 = read_data(data_dir,"dataset1.csv",2)
    X_dset2, Y_dset2 = read_data(data_dir,"dataset2.csv",2)
    
    ##### 1-) Visualize the dataset on a figure
    class0_X_dset1 = X_dset1[Y_dset1==0 ,:]
    class1_X_dset1 = X_dset1[Y_dset1==1 ,:]
    class0_X_dset2 = X_dset2[Y_dset2==0 ,:]
    class1_X_dset2 = X_dset2[Y_dset2==1 ,:]
 
    
    fig0 = plt.figure()
    plt.plot(class0_X_dset1[:,0],class0_X_dset1[:,1],'ro',label="Class 0")
    plt.plot(class1_X_dset1[:,0],class1_X_dset1[:,1],'gx',label="Class 1")
    plt.title("Dataset1 Dataset Visualization")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()

    fig0_2 = plt.figure()
    plt.plot(class0_X_dset2[:,0],class0_X_dset2[:,1],'ro',label="Class 0")
    plt.plot(class1_X_dset2[:,0],class1_X_dset2[:,1],'gx',label="Class 1")
    plt.title("Dataset2 Dataset Visualization")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()

    
    ##### 2-) Train a linear SVM on the datase
    # Try with different c values
    c_values = [0.001,0.01,0.1,1]

    ### c = 0.001 case
    clf_dset1_p001 = svm.SVC(kernel='linear', C = c_values[0])
    clf_dset1_p001.fit(X_dset1,Y_dset1)
    
    weights = clf_dset1_p001.coef_[0]
    m = -weights[0] / weights[1]
    x_axis = np.linspace(0,5,100)
    y_axis = m * x_axis - clf_dset1_p001.intercept_[0] / weights[1]

    fig1 = plt.figure()
    #plt.plot(x_axis, y_axis)
    plt.plot(class0_X_dset1[:,0],class0_X_dset1[:,1],'ro',label="Class 0")
    plt.plot(class1_X_dset1[:,0],class1_X_dset1[:,1],'gx',label="Class 1")
    plot_svc_decision_function(clf_dset1_p001)
    plt.title("Linear SVM with c=%.3f for Dataset1" %c_values[0])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()


    clf_dset2_p001 = svm.SVC(kernel='linear', C = c_values[0])
    clf_dset2_p001.fit(X_dset2,Y_dset2)

    weights_2 = clf_dset2_p001.coef_[0]
    m_2 = -weights_2[0] / weights_2[1]
    x_axis_2 = np.linspace(-0.6,0.3,100)
    y_axis_2 = m_2 * x_axis_2 - clf_dset2_p001.intercept_[0] / weights_2[1]

    fig1_2 = plt.figure()
    #plt.plot(x_axis_2, y_axis_2)
    plt.plot(class0_X_dset2[:,0],class0_X_dset2[:,1],'ro',label="Class 0")
    plt.plot(class1_X_dset2[:,0],class1_X_dset2[:,1],'gx',label="Class 1")
    plot_svc_decision_function(clf_dset2_p001)
    plt.title("Linear SVM with c=%.3f for Dataset2" %c_values[0])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    

    ### c = 0.01 case
    clf_dset1_p01 = svm.SVC(kernel='linear', C = c_values[1])
    clf_dset1_p01.fit(X_dset1,Y_dset1)

    weights2 = clf_dset1_p01.coef_[0]
    m2 = -weights2[0] / weights2[1]
    x_axis2 = np.linspace(0,5,100)
    y_axis2 = m2 * x_axis2 - clf_dset1_p01.intercept_[0] / weights2[1]

    fig2 = plt.figure()
    plt.plot(x_axis2, y_axis2)
    plt.plot(class0_X_dset1[:,0],class0_X_dset1[:,1],'ro',label="Class 0")
    plt.plot(class1_X_dset1[:,0],class1_X_dset1[:,1],'gx',label="Class 1")
    plot_svc_decision_function(clf_dset1_p01)
    plt.title("Linear SVM with c=%.3f for Dataset1" %c_values[1])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()


    clf_dset2_p01 = svm.SVC(kernel='linear', C = c_values[1])
    clf_dset2_p01.fit(X_dset2,Y_dset2)

    weights2_2 = clf_dset2_p01.coef_[0]
    m2_2 = -weights2_2[0] / weights2_2[1]
    x_axis2_2 = np.linspace(-0.6,0.3,100)
    y_axis2_2 = m2_2 * x_axis2_2 - clf_dset2_p01.intercept_[0] / weights2_2[1]

    fig2_2 = plt.figure()
    #plt.plot(x_axis2_2, y_axis2_2)
    plt.plot(class0_X_dset2[:,0],class0_X_dset2[:,1],'ro',label="Class 0")
    plt.plot(class1_X_dset2[:,0],class1_X_dset2[:,1],'gx',label="Class 1")
    plot_svc_decision_function(clf_dset2_p01)
    plt.title("Linear SVM with c=%.3f for Dataset2" %c_values[1])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()

    ### c = 0.1 case
    clf_dset1_p1 = svm.SVC(kernel='linear', C = c_values[2])
    clf_dset1_p1.fit(X_dset1,Y_dset1)

    weights3 = clf_dset1_p1.coef_[0]
    m3 = -weights3[0] / weights3[1]
    x_axis3 = np.linspace(0,5,100)
    y_axis3 = m3 * x_axis3 - clf_dset1_p1.intercept_[0] / weights3[1]

    fig3 = plt.figure()
    #plt.plot(x_axis3, y_axis3)
    plt.plot(class0_X_dset1[:,0],class0_X_dset1[:,1],'ro',label="Class 0")
    plt.plot(class1_X_dset1[:,0],class1_X_dset1[:,1],'gx',label="Class 1")
    plot_svc_decision_function(clf_dset1_p1)
    plt.title("Linear SVM with c=%.3f for Dataset1" %c_values[2])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()


    clf_dset2_p1 = svm.SVC(kernel='linear', C = c_values[2])
    clf_dset2_p1.fit(X_dset2,Y_dset2)

    weights3_2 = clf_dset2_p1.coef_[0]
    m3_2 = -weights3_2[0] / weights3_2[1]
    x_axis3_2 = np.linspace(-0.6,0.3,100)
    y_axis3_2 = m3_2 * x_axis3_2 - clf_dset2_p1.intercept_[0] / weights3_2[1]

    fig3_2 = plt.figure()
    #plt.plot(x_axis3_2, y_axis3_2)
    plt.plot(class0_X_dset2[:,0],class0_X_dset2[:,1],'ro',label="Class 0")
    plt.plot(class1_X_dset2[:,0],class1_X_dset2[:,1],'gx',label="Class 1")
    plot_svc_decision_function(clf_dset2_p1)
    plt.title("Linear SVM with c=%.3f for Dataset2" %c_values[2])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()

    ### c = 1 case
    clf_dset1_1 = svm.SVC(kernel='linear', C = c_values[3])
    clf_dset1_1.fit(X_dset1,Y_dset1)

    weights4 = clf_dset1_1.coef_[0]
    m4 = -weights4[0] / weights4[1]
    x_axis4 = np.linspace(0,5,100)
    y_axis4 = m4 * x_axis4 - clf_dset1_1.intercept_[0] / weights4[1]

    fig4 = plt.figure()
    #plt.plot(x_axis4, y_axis4)
    plt.plot(class0_X_dset1[:,0],class0_X_dset1[:,1],'ro',label="Class 0")
    plt.plot(class1_X_dset1[:,0],class1_X_dset1[:,1],'gx',label="Class 1")
    plot_svc_decision_function(clf_dset1_1)
    plt.title("Linear SVM with c=%.3f for Dataset1" %c_values[3])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()


    clf_dset2_1 = svm.SVC(kernel='linear', C = c_values[3])
    clf_dset2_1.fit(X_dset2,Y_dset2)

    weights4_2 = clf_dset2_1.coef_[0]
    m4_2 = -weights4_2[0] / weights4_2[1]
    x_axis4_2 = np.linspace(-0.6,0.3,100)
    y_axis4_2 = m4_2 * x_axis4_2 - clf_dset2_1.intercept_[0] / weights4_2[1]

    fig4_2 = plt.figure()
    #plt.plot(x_axis4_2, y_axis4_2)
    plt.plot(class0_X_dset2[:,0],class0_X_dset2[:,1],'ro',label="Class 0")
    plt.plot(class1_X_dset2[:,0],class1_X_dset2[:,1],'gx',label="Class 1")
    plot_svc_decision_function(clf_dset2_1)
    plt.title("Linear SVM with c=%.3f for Dataset2" %c_values[3])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()











    plt.show(block=False) # show all the figures at once
    plt.waitforbuttonpress(1)
    input("Please press any key to close all figures.")
    plt.close("all")

 


if __name__ == "__main__": main()
