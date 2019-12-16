import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
import math
import pandas as pd
import random
import sklearn.semi_supervised as lp
import copy


def diffusion(x,y,gamma=0.2):
    """
    THE DIFFUSION KERNEL FUNCTION CORRESPONDING TO REFERENCE (Kondor & Lafferty, 2002)
    Implemented as given in the spectrum transform of the laplacian
    Here we build the KERNEL matrix, which is used in both SSL algorithms.

    ### Later implemented in it's pure form for purposes of analysis

    :param x:array of shape (n_samples_X, n_features)
    :param y:array of shape (n_samples_Y, n_features)

    :param gamma: Although gamma is an adjustable hyperparameter, Here we use it as the square of the standard deviation of the data, according to the above cited paper.
    :return: KERNEL MATRIX
    """
    d = cdist(x, y)
    #d=np.square(d)
    gamma = np.std(d)
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            d[i][j]=math.exp(-(gamma**2)*d[i][j]/2)

    return d/math.sqrt(2*math.pi*(gamma**2))

def reg_laplacian(x,y,gamma=0.3):
    """
    Regular Laplacian Kernel of the lowest order. (1)
    :param x:
    :param y:
    :param gamma:
    :return:
    """

    d = cdist(x, y,'minkowski',p=1.)
    std = np.std(d)
    K = np.exp(-d /std)
    return K

def higher_order_laplacian(x,y,gamma=0.3):
    """
    Higher orders of the laplacian kernel, now set to 3
    :param x:
    :param y:
    :param gamma:
    :return:
    """
    d = cdist(x, y,'minkowski',p=3.)
    std = np.std(d)
    K = np.exp(-d /(std**3)*3)
    return K

def inverse_cosine(x,y,gamma=0.2):

    """
    INVERSE COSINE KERNEL FUNCTION.

    :param x:
    :param y:
    :param gamma:
    :return:
    """

    d=cdist(x,y,'cosine')

    #for i in range(d.shape[0]):
    #    for j in range(d.shape[1]):
    #        d[i][j]=math.cos((d[i][j]*math.pi)/4)
    std = np.std(d)
    K = np.exp(-1 / 2 * np.square(d / std))
    return K



def gaussian_kernel(x,y,gamma=0.2):
    """
    Standard gaussian kernel function.
    Show in (Kondor, Lafferty) to be equal to Dirac spike funtion.

    """
    if y is None:
        d = squareform(pdist(x))
    else:
        d = cdist(x, y)
    std = np.std(d)
    K = np.exp(-1 / 2 * np.square(d / std))
    return K


def label_propagation(y_labeled, X_labeled, X_unlabeled, X_train, kernel, mu=1., verbose=True):
    """
    Label propagation algorithm on the data
    :param y_labeled: targets for labeled points
    :param X_labeled:  data for labeled points
    :param X_unlabeled: data for unlabeled points
    :param X_train: data for where to evaluate the label
    :param kernel: kernel function
    :param mu: hyperparameter for the label prop algo
    :param verbose: do you want to print stuff?
    :return:
    """
    n_unlabeled = X_unlabeled.shape[0]
    n_labeled = X_labeled.shape[0]
    n_train = X_train.shape[0]

    # concatenate all data points
    X_eval = np.concatenate((X_labeled, X_unlabeled, X_train), 0)
    #Easier to run the Kernel on one large concatenated matrix, mensioned in paper
    W = kernel(X_eval, None)
    D = np.sum(W, 0)

    eps = 1E-9  # arbitrary small number
    # Matrix of ones for n_labeled and zeroes for unlabeled and the ones we need to predict plus the hyperparameters
    A = np.diag(np.concatenate((np.ones(n_labeled), np.zeros(n_unlabeled + n_train))) + mu * D + mu * eps)
    y_hat_0 = np.concatenate((y_labeled, np.zeros((n_unlabeled + n_train))))
    y_hat = copy.copy(y_hat_0)

    for iter in range(100):
        y_hat_old = y_hat
        y_hat = np.linalg.solve(A, mu * np.dot(W, y_hat) + y_hat_0)

        if np.linalg.norm(y_hat - y_hat_old) < 0.01:
            if verbose:
                print(np.linalg.norm(y_hat - y_hat_old),"This is norm",'\n')
            break
    else:
        if verbose:
            print("Run it again, make it verbose, either the values are not converging or it needs more iterations")

    return y_hat[-n_train:]

def jaccard_kernel(x,y,gamma=0.2):
    """
    Jaccard Kernel
    :param x:
    :param y:
    :param gamma:
    :return:
    """
    d=cdist(x,y,'jaccard')
    std = np.std(d)
    K = np.exp(-1 / 2 * np.square(d / std))
    #K = np.exp(-d / std)
    return K
def hamming_kernel(x,y,gamma=0.2):
    """
    Hamming Kernel
    :param x:
    :param y:
    :param gamma:
    :return:
    """
    d=cdist(x,y,'hamming')
    std = np.std(d)
    K = np.exp(-1 / 2 * np.square(d / std))

    return K

def correlation_kernel(x,y,gamma=0.2):
    """
    Correlation Kernel
    :param x:
    :param y:
    :param gamma:
    :return:
    """

    d=cdist(x,y,'correlation')

    std = np.std(d)
    K = np.exp(-1 / 2 * np.square(d / std))
    return K
def main(path="../isolet",kernel=reg_laplacian):
    feature_name_file = path + '.names'
    infile = path + '.data'
    print('loading the dataset')
    print('we are using %s' % str(path))
    with open(feature_name_file, 'r+') as f:
    	lines = f.readlines()
    raw = pd.read_csv(infile,sep=',')
    print(raw.shape)

    SPLIT=int((raw.shape[0]-1)*(4/10))

    Y = raw.iloc[1:SPLIT,:-1].values
    Y_labels=raw.iloc[1:SPLIT,-1].values

## Y IS FOR TESTING

    X_labels= raw.iloc[SPLIT:,-1].values
    X=raw.iloc[SPLIT:,:-1].values
    X_labels_saved=copy.copy(X_labels)

    for p in range (1):
        X_labels=copy.copy(X_labels_saved)
        for i in range(int((X.shape[0]-SPLIT)/5)):
            j=random.randint(SPLIT,X.shape[0]-1)
            X_labels[j]=-1

    #print(X_labels.shape,"Labels")
    #print(X.shape,"X")
    #print(Y.shape,"Y")
    #print(laplacian(X,Y))

        k=lp.LabelPropagation(kernel=kernel,gamma=1, n_neighbors=7, max_iter=1000, tol=0.01, n_jobs=-1)
        k.fit(X,X_labels)
        print(k.score(Y, Y_labels))
    #plt.plot([print(k.score(Y,Y_labels)),((p*100)/X.shape[0])])

if __name__=="__main__":
    main()