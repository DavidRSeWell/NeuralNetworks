import numpy as np

def make_ortho_compare_matrix(X ,Y):
    '''
        make a row v row ortho comparison
        of X on Y
    '''

    assert X.shape == Y.T.shape

    ortho_X_Y = np.zeros((X.shape[0],Y.shape[1]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            dot_prod = np.dot(X[i] ,Y.T[j])
            ortho_X_Y[i][j] = dot_prod

    return ortho_X_Y


if __name__ == '__main__':


    run_test_compare_ortho = 1
    if run_test_compare_ortho:

        X = np.random.random((5,14))
        Y = np.random.random((14,5))

        make_ortho_compare_matrix(X,Y)


