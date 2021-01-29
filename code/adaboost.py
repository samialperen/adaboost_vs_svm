# This script contains code for Adaboost M1 Classfier
# Author: Sami Alperen Akgun
# Email: sami.alperen.akgun@gmail.com

import os
import numpy as np
from matplotlib import pyplot as plt 
from sklearn import svm #svm library from sklearn Python module
# To divide the data into training and test set, just for visualization part
# Otherwise, see ten_times_10_fold_cv function (my_implementation)
from sklearn.model_selection import train_test_split 


def read_data(data_path, filename1,filename2,feature_number):
    """
        This function reads the data from data_path/filename
        WARNING: This function assumes that features of data
        is separated by commas in the file
        Input: data_path --> The full directory path of data
        filename --> name of the file (With extension)
        feature_number --> Total feature number in the data
        filename1 --> File that contains data related to class A
        filename2 --> File that contains data related to class B
        Output: X --> numpy array that contains feature values
        sample size = len(filename1) + len(filename2)
        size(X) --> sample size x feature number
        Y --> numpy array that contains labels
        size(Y) --> (sample size,)
    """

    # Features of Class A
    with open(data_path + "/" + filename1, 'r', encoding='utf-8-sig') as f: 
        X_A = np.genfromtxt(f, delimiter=',')[:,0:feature_number]
    # Features of Class B
    with open(data_path + "/" + filename2, 'r', encoding='utf-8-sig') as f: 
        X_B = np.genfromtxt(f, delimiter=',')[:,0:feature_number]
    
    # Combine features of Class A and Class B to a single X feature array
    X = np.concatenate((X_A,X_B),axis=0)

    # Assume Class A label is -1 and Class B label is 1
    classA_sample_size = X_A.shape[0]
    classB_sample_size = X_B.shape[0]
    # m = total sample size  
    m = classA_sample_size + classB_sample_size
    Y = np.ones(m) * -1
    Y[classA_sample_size:] = 1  
    

    return X,Y

    ## Shuffle the data 
    #x1_size, x2_size = X.shape
    #y1_size, y2_size = Y.shape
#
    #combined_data = np.concatenate((X,Y),axis=1)
    #np.random.shuffle(combined_data) #this function shuffles
#
    #X_shuffled = combined_data[:,0:x2_size]
    #Y_shuffled = combined_data[:,x2_size:]
    #return X_shuffled, Y_shuffled

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

def ten_fold_cv(X_data,Y_data,c_value):
    """
        This function applies 10 fold Cross validation on the given input data
        Inputs: X_data --> input data contains features and their values
                Y_data --> input data contains output labels
                c_value --> Penalization factor for linear SVM classifier      
        Output: true_error --> scalar
    """

    # Shuffle data at the beginning 
    x1_size, x2_size = X_data.shape

    Y_data = Y_data.reshape(Y_data.shape[0],1) 
    
    combined_data = np.concatenate((X_data,Y_data),axis=1)
    np.random.shuffle(combined_data) #this function shuffles

    X_shuffled = combined_data[:,0:x2_size]
    Y_shuffled = combined_data[:,x2_size:]
    Y_shuffled = Y_shuffled.reshape(Y_shuffled.shape[0])
    
    # Divide data into 10 parts
    m = x1_size #total sample size
    remainder = 10 - m%10 #last piece will have this much less elements
    regular_length = (m+remainder) // 10
    # For example, when m=958 --> remainder is 2
    # Piece1 length --> 96 , Piece2 length --> 96 ... Piece 10 length --> 96-2=94
    
    best_accuracy = 0 
    accuracies = [] # this list contains accuracy for each validation set

    ##### First 9 pieces --> last piece might have different length than regular_length
    for i in range(9):
        # initial and end point index
        init = i*regular_length
        end = (i+1) * regular_length
        
        # selected test set
        test_X = X_shuffled[init:end,:] #contains features related to groundtruth
        test_Y = Y_shuffled[init:end] #contains true labels (groundtruth)
        
        # selected training set --> rest of data after selecting test set
        x_part1 = X_shuffled[0:init,:]
        x_part2 = X_shuffled[end:,:]
        y_part1 = Y_shuffled[0:init]
        y_part2 = Y_shuffled[end:]
        training_X = np.concatenate((x_part1,x_part2))
        training_Y = np.concatenate((y_part1,y_part2))

        # train the tree using training set
        clf = svm.SVC(kernel='linear', C = c_value)
        clf.fit(training_X,training_Y)
        
       
        predicted_labels = np.empty((test_Y.shape),dtype=object)
        predicted_labels = clf.predict(test_X)
        
        # This variable holds the total number of correct predictions
        correct_prediction_number = len(test_Y[test_Y == predicted_labels])
        current_accuracy = correct_prediction_number/regular_length
     
        accuracies.append(current_accuracy) #hold all accuracies in a list

        if current_accuracy >= best_accuracy:
            best_test_X = test_X
            best_test_Y = test_Y
            best_accuracy = current_accuracy

    ##### Last piece --> length could be less than regular_length
    # selected test set
    test_X_last_piece = X_shuffled[end:,:] #contains features related to groundtruth
    test_Y_last_piece = Y_shuffled[end:] #contains true labels (groundtruth)
    
    # selected training set --> rest of data after selecting test set
    training_X_last_piece = X_shuffled[0:end,:]
    training_Y_last_piece = Y_shuffled[0:end]
    
    # train the last tree using last training set piece
    clf_last_piece = svm.SVC(kernel='linear', C = c_value)
    clf_last_piece.fit(training_X_last_piece,training_Y_last_piece)
        
    last_predicted_labels = np.empty((test_Y_last_piece.shape),dtype=object)
    last_predicted_labels = clf_last_piece.predict(test_X_last_piece)

    last_length = test_X_last_piece.shape[0]

    last_prediction_number = len(test_Y_last_piece[test_Y_last_piece == last_predicted_labels])
    last_accuracy = last_prediction_number/last_length
    
    accuracies.append(last_accuracy)

    if last_accuracy >= best_accuracy:
        best_accuracy = last_accuracy
        best_test_X = test_X_last_piece
        best_test_Y = test_Y_last_piece    

    return accuracies, best_accuracy, best_test_X, best_test_Y

def adaboost_ten_fold_cv(X_data,Y_data,c_value):
    """
        This function applies 10 fold Cross validation on the given input data
        using adaboost_M1 approach and linear SVM classifier as a weak classifier
        Inputs: X_data --> input data contains features and their values
                Y_data --> input data contains output labels
                c_value --> Penalization factor for linear SVM classifier      
        Output: accuracies --> list contains 10 accuracy value for each fold
                best_accuracy --> best accuracy among accuracies
    """

    # Shuffle data at the beginning 
    x1_size, x2_size = X_data.shape

    Y_data = Y_data.reshape(Y_data.shape[0],1) 
    
    combined_data = np.concatenate((X_data,Y_data),axis=1)
    np.random.shuffle(combined_data) #this function shuffles

    X_shuffled = combined_data[:,0:x2_size]
    Y_shuffled = combined_data[:,x2_size:]
    Y_shuffled = Y_shuffled.reshape(Y_shuffled.shape[0])
    
    # Divide data into 10 parts
    m = x1_size #total sample size
    remainder = 10 - m%10 #last piece will have this much less elements
    regular_length = (m+remainder) // 10
    # For example, when m=958 --> remainder is 2
    # Piece1 length --> 96 , Piece2 length --> 96 ... Piece 10 length --> 96-2=94
    
    best_accuracy = 0 
    accuracies = [] # this list contains accuracy for each validation set

    ##### First 9 pieces --> last piece might have different length than regular_length
    for i in range(9):
        # initial and end point index
        init = i*regular_length
        end = (i+1) * regular_length
        
        # selected test set
        test_X = X_shuffled[init:end,:] #contains features related to groundtruth
        test_Y = Y_shuffled[init:end] #contains true labels (groundtruth)
        
        # selected training set --> rest of data after selecting test set
        x_part1 = X_shuffled[0:init,:]
        x_part2 = X_shuffled[end:,:]
        y_part1 = Y_shuffled[0:init]
        y_part2 = Y_shuffled[end:]
        training_X = np.concatenate((x_part1,x_part2))
        training_Y = np.concatenate((y_part1,y_part2))

        # Train using adaboost 
        h_list, beta_list, weight_list = adaboost_M1(training_X,training_Y,c_value,50)
        # Predict using adaboost model        
        
        predicted_labels = np.empty((test_Y.shape),dtype=object)
        predicted_labels = adaboost_M1_predict(test_X,h_list,beta_list)
        
        # This variable holds the total number of correct predictions
        correct_prediction_number = len(test_Y[test_Y == predicted_labels])
        current_accuracy = correct_prediction_number/regular_length
     
        accuracies.append(current_accuracy) #hold all accuracies in a list

        if current_accuracy >= best_accuracy:
            best_test_X = test_X
            best_test_Y = test_Y
            best_accuracy = current_accuracy

    ##### Last piece --> length could be less than regular_length
    # selected test set
    test_X_last_piece = X_shuffled[end:,:] #contains features related to groundtruth
    test_Y_last_piece = Y_shuffled[end:] #contains true labels (groundtruth)
    
    # selected training set --> rest of data after selecting test set
    training_X_last_piece = X_shuffled[0:end,:]
    training_Y_last_piece = Y_shuffled[0:end]
    
    # train the last one using last training set piece
    h_list_last, beta_list_last, weight_list_last = adaboost_M1(training_X_last_piece,training_Y_last_piece,c_value,50)
     
    last_predicted_labels = np.empty((test_Y_last_piece.shape),dtype=object)
    last_predicted_labels = adaboost_M1_predict(test_X_last_piece,h_list_last,beta_list_last)

    last_length = test_X_last_piece.shape[0]

    last_prediction_number = len(test_Y_last_piece[test_Y_last_piece == last_predicted_labels])
    last_accuracy = last_prediction_number/last_length
    
    accuracies.append(last_accuracy)

    if last_accuracy >= best_accuracy:
        best_accuracy = last_accuracy
        best_test_X = test_X_last_piece
        best_test_Y = test_Y_last_piece    

    return accuracies, best_accuracy, best_test_X, best_test_Y

def ten_times_10_fold(X_in,Y_in,c_value,Use_Adaboost=False):
    """
        This function calls 10-fold cross validation function 10 times
        and returns best accuracy, best training set and mean and variance
        of calculated all accuracies , i.e. 100 accuracy value       
    """
    
    ten_time_accurs_mean = []
    ten_time_accurs_var = []
    ten_time_best_accr = 0

    for i in range(10):
        if Use_Adaboost == False:
            one_time_accurs,one_time_best_accr,one_time_best_X,one_time_best_Y=( 
                     ten_fold_cv(X_in,Y_in,c_value))
        elif Use_Adaboost == True:
            one_time_accurs,one_time_best_accr,one_time_best_X,one_time_best_Y=( 
                     adaboost_ten_fold_cv(X_in,Y_in,c_value))
        ten_time_accurs_mean.append(np.mean(one_time_accurs))
        ten_time_accurs_var.append(np.var(one_time_accurs))
        if one_time_best_accr >= ten_time_best_accr:
            ten_time_best_test_X = one_time_best_X
            ten_time_best_test_Y = one_time_best_Y
            ten_time_best_accr = one_time_best_accr

    print("Mean of the accuracy: ", np.mean(ten_time_accurs_mean))
    print("Var of the accuracy: ", np.var(ten_time_accurs_var))
    print("Best accuracy: ", ten_time_best_accr)

    return ten_time_best_accr, ten_time_best_test_X, ten_time_best_test_Y

def h_error(Y_prediction,Y_truth,D):
    """
        This function calculates hypothesis error for trained model h
        Inputs: Y_prediction --> Array contains predicted output labels 
                Y_truth --> Ground truth labels
                h --> Trained hypothesis, i.e. model
                D --> Weights --> Shape (m,) where m=#Samples
        Output: epsilon --> hypothesis error, scalar
    """
    # Select the weights that correspond to the incorrect classification cases
    D_wrong = D[Y_truth != Y_prediction]
    # Sum wrong weights to obtain hypothesis error epsilon
    epsilon = np.sum(D_wrong)
    return epsilon

def update_weights(Y_prediction,Y_truth,beta,current_D):
    """
        This function updates weights for adaboost function
        using current weights, current_D, and beta values.
        Inputs: Y_prediction --> Array contains predicted output labels
                Y_truth --> Array contains ground truth labels
                beta --> scalar, current beta value
                current_D --> Array contains current weights, shape = (m,)
                where m is total number of samples
        Output: D_next --> weights for the next iteration of adaboots, shape (m,)
    """
    # Select only correct predictions and scale them with beta value
    D = current_D
    D[Y_truth == Y_prediction] *= beta
    D_next = D/sum(D) #normalize
    #print("always should be equal to one: ", sum(D_next/sum(D_next)))
    return D_next



def adaboost_M1(X_data,Y_data,c_value,T):
    """
        This function applies 10 fold Cross validation on the given input data
        Inputs: X_data --> input data contains features and their values
                Y_data --> input data contains output labels
                c_value --> Penalization factor for linear SVM classifier
                T --> Max number of weak learners, i.e. max number of iterations
                for adaboost mainloop      
        Output: h --> fitted linear SVM model
    """

    m = X_data.shape[0] # Total number of samples
    n = X_data.shape[1] # Total number of features
    h_list = [] # The list to hold all hyptohesis, i.e. learned models
    D = [] # List to hold all weights for each hypothesis
    beta_list = [] # List to hold all beta values
    # Initialize weight list with 1/m for all samples
    D.append( np.ones(m) * (1/m) )# --> shape = (m,)

    t = 0
    while t<T:
        # Pick 100 examples as a training set out of m data with probability
        # distribution D[t]
       
        training_indices = np.random.choice(D[t].shape[0],100,replace=False,p=D[t])
        training_X = X_data[training_indices,:]
        training_Y = Y_data[training_indices]
        h = svm.SVC(kernel='linear', C = c_value) # current hypothesis 
        h.fit(training_X,training_Y) # train using training set
        Y_predicted = h.predict(X_data) # predict using all samples
      
        epsilon = h_error(Y_predicted,Y_data,D[t])
        
        # if error is more than 50%, start the loop again
        # This means we are discarding the classifier corresponding error
        # rate more than 50% and we are trying to resample another training
        # set --> Therefore, in the end we will have 50 trained classifiers
        if epsilon >= 0.5:
            # Not update t, so that we can iterate with the same t value
            continue

        beta = epsilon / (1-epsilon)
        D.append( update_weights(Y_predicted,Y_data,beta,D[t]) ) #--> D[t+1]
        h_list.append(h)
        beta_list.append(beta)
        t+=1 # While loop!

    return h_list, beta_list, D


def adaboost_M1_predict(X,h_list,beta_list):
    """
        This function creates a prediction for given test
        set X using h_list that contains trained classifiers using
        adaboost_M1 approach
        Inputs: X--> Test set, shape = mxn
                h_list --> Contains all trained hypotheses, shape = (T,)
                beta_list --> Contains beta values for corresponding h's 
                              shape = (T,)
        Output: Y_prediction = Array of size (m,) that contains predicted labels
    """
    total = [] 
    for i in range(len(h_list)):
        total.append( h_list[i].predict(X) * np.log(1/beta_list[i]) )

    # Class A label -1 and Class B label +1 , so sign function will work here
    # np.sign --> returns +1 for positive and returns -1 for negative input
    Y_prediction = np.sign(sum(total)) # --> shape (m,)
    return Y_prediction

def plot_adaboost(h_list, beta_list, X, Y ):
    """
        This function plots the decision boundary for the 
        adaboost-M1 classifier
    """
    x1_mesh, x2_mesh = np.meshgrid(np.arange(150, 451), np.arange(0, 351))
    x1_temp = x1_mesh.ravel()
    x1 = x1_temp.reshape(x1_temp.shape[0],1)
    x2_temp = x2_mesh.ravel()
    x2 = x2_temp.reshape(x2_temp.shape[0],1)
    X = np.concatenate((x1,x2),axis=1)
    X_mesh = np.concatenate((x1_mesh,x2_mesh))
    level = adaboost_M1_predict(X, h_list, beta_list)            
    Level = np.asarray(level).reshape(x1_mesh.shape)
    plt.contour(x1_mesh,x2_mesh,Level,2,colors="blue")
    
    

def main():
    """
        This is the main function of this script
    """

    ##### 1-) Load data
    # If you run the code from pattern_recognition_assignment3 path, uncomment below
    data_dir = os.getcwd() + '/data' 
    # If you run the code from code directory, uncomment below
    #data_path = os.getcwd() +  ".." / "data"/
    X_dset, Y_dset = read_data(data_dir,"classA.csv","classB.csv",2)
    
    ###### 1-) Visualize the dataset on a figure
    classA_X_dset = X_dset[Y_dset==-1 ,:]
    classB_X_dset = X_dset[Y_dset==1 ,:]
    
    fig0 = plt.figure()
    plt.plot(classA_X_dset[:,0],classA_X_dset[:,1],'ro',label="Class A")
    plt.plot(classB_X_dset[:,0],classB_X_dset[:,1],'gx',label="Class B")
    plt.title("Dataset Visualization for Adaboost M1 Classifier")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()

    

    
    ###### 2-) Train a linear SVM on the datase
    # Try with different c values
    c_values = [0.1,1,10,100]

    ### c = 0.1 case
    clf_dset_p1 = svm.SVC(kernel='linear', C = c_values[0])
    clf_dset_p1.fit(X_dset,Y_dset)
    
    weights = clf_dset_p1.coef_[0]
    m = -weights[0] / weights[1]
    x_axis = np.linspace(150,450,1000)
    y_axis = m * x_axis - clf_dset_p1.intercept_[0] / weights[1]

    fig1 = plt.figure()
    #plt.plot(x_axis, y_axis)
    plt.plot(classA_X_dset[:,0],classA_X_dset[:,1],'ro',label="Class A")
    plt.plot(classB_X_dset[:,0],classB_X_dset[:,1],'gx',label="Class B")
    plot_svc_decision_function(clf_dset_p1)
    plt.title("Linear SVM with c=%.3f for Adaboost M1 Classifier" %c_values[0])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()

    ### c = 1 case
    clf_dset_1 = svm.SVC(kernel='linear', C = c_values[1])
    clf_dset_1.fit(X_dset,Y_dset)
    
    weights2 = clf_dset_1.coef_[0]
    m2 = -weights2[0] / weights2[1]
    x_axis2 = np.linspace(150,450,1000)
    y_axis2 = m2 * x_axis2 - clf_dset_1.intercept_[0] / weights2[1]

    fig2 = plt.figure()
    #plt.plot(x_axis2, y_axis2)
    plt.plot(classA_X_dset[:,0],classA_X_dset[:,1],'ro',label="Class A")
    plt.plot(classB_X_dset[:,0],classB_X_dset[:,1],'gx',label="Class B")
    plot_svc_decision_function(clf_dset_1)
    plt.title("Linear SVM with c=%.3f for Adaboost M1 Classifier" %c_values[1])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()

    ### c = 10 case
    clf_dset_10 = svm.SVC(kernel='linear', C = c_values[2])
    clf_dset_10.fit(X_dset,Y_dset)
    
    weights3 = clf_dset_10.coef_[0]
    m3 = -weights3[0] / weights3[1]
    x_axis3 = np.linspace(150,450,1000)
    y_axis3 = m3 * x_axis3 - clf_dset_10.intercept_[0] / weights3[1]

    fig3 = plt.figure()
    #plt.plot(x_axis3, y_axis3)
    plt.plot(classA_X_dset[:,0],classA_X_dset[:,1],'ro',label="Class A")
    plt.plot(classB_X_dset[:,0],classB_X_dset[:,1],'gx',label="Class B")
    plot_svc_decision_function(clf_dset_10)
    plt.title("Linear SVM with c=%.3f for Adaboost M1 Classifier" %c_values[2])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()


    ### c = 100 case
    clf_dset_100 = svm.SVC(kernel='linear', C = c_values[3])
    clf_dset_100.fit(X_dset,Y_dset)
    
    weights4 = clf_dset_100.coef_[0]
    m4 = -weights4[0] / weights4[1]
    x_axis4 = np.linspace(150,450,1000)
    y_axis4 = m4 * x_axis4 - clf_dset_100.intercept_[0] / weights4[1]

    fig4 = plt.figure()
    #plt.plot(x_axis4, y_axis4)
    plt.plot(classA_X_dset[:,0],classA_X_dset[:,1],'ro',label="Class A")
    plt.plot(classB_X_dset[:,0],classB_X_dset[:,1],'gx',label="Class B")
    plot_svc_decision_function(clf_dset_100)
    plt.title("Linear SVM with c=%.3f for Adaboost M1 Classifier" %c_values[3])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()


    c_values = [0.1,1,10,100]
    for i in range(len(c_values)):
        print("C value: ", c_values[i])
        best_accr, best_test_X, best_test_Y = ten_times_10_fold(X_dset,Y_dset,c_values[i])
    
    
    ### Part 4 --> Best c value is 1
    best_accr, best_test_X, best_test_Y = ten_times_10_fold(X_dset,Y_dset,1,True)
    
    
    ### Part 5
    # 10% Test Set, 90% Training Set
    X_train, X_test, Y_train, Y_test= train_test_split(X_dset, Y_dset, test_size= 0.1)
    list_h, list_beta, list_D = adaboost_M1(X_train,Y_train,1,50)
    
    fig5 = plt.figure()
    plot_adaboost(list_h,list_beta,X_dset,Y_dset)
    plt.plot(classA_X_dset[:,0],classA_X_dset[:,1],'ro',label="Class A")
    plt.plot(classB_X_dset[:,0],classB_X_dset[:,1],'gx',label="Class B")
    plt.title("Adaboost Decision Boundary")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    
  
    
    

 





    plt.show(block=False) # show all the figures at once
    plt.waitforbuttonpress(1)
    input("Please press any key to close all figures.")
    plt.close("all")

 


if __name__ == "__main__": main()
