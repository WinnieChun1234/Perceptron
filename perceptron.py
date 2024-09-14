################################################################################
# COMP3317 Computer Vision
# Assignment 5 - Perceptron learning
################################################################################
import sys, argparse
import numpy as np
import random

################################################################################
#  train the perceptron using the training data
################################################################################
def train(train_data, max_itr = 100) :
    # input:
    #    train_data - a n x 5 numpy ndarray (dtype = np.float64) holding the
    #                 training data (n being the number of samples), with the
    #                 first 4 columns being the attributes and the last column
    #                 being the labels of the training data
    #    max_itr    - the max no. of iterations to prevent infinite loop in case
    #                 training data are not linearly separable
    # output:
    #    params - a 5-D numpy ndarray (dtype = np.float64) holding the learned
    #             parameters of the perceptron

    # TODO : initialize parameters of the perceptron to small random values
    params = np.random.normal(size=(5,)) # params = w = b
    weight = np.random.standard_normal(size=(5,))


    # TODO : extract input values from train_data and augment the inputs with x0 = 1
    x = train_data
    x = np.insert(x, 0, 1, axis=1)
    x = np.delete(x, 5, axis=1)


    # TODO : extract target outputs from train_data
    t = train_data[:,4]


    # TODO: repeat until converged or max no. of iterations has been reached
    count = 0
    while count < max_itr :
        # TODO: make the prediction
        y = np.matmul(x, params)
        for i in range(y.shape[0]):
            if y[i] < 0:
                y[i]  = 0
            else:
                y[i]  = 1
        err = np.subtract(y, t)
        weight = np.subtract(params, np.matmul(x.T, err))
        if np.array_equal(weight, params) == True:
            break
        else:
            params = weight
        count += 1

    return params

################################################################################
#  predict the outputs for the testing data
################################################################################
def predict(test_data, params) :
    # input:
    #    test_data - a n x 4 numpy ndarray (dtype = np.float64) holding values
    #                of the attributes for the testing data (n being number of
    #                samples)
    # output:
    #    out - a n-D numpy ndarray (dtype = np.unit8) holding the predicted
    #          outputs of the testing data (n being number of samples), with 0
    #          for Iris setosa and 1 for Iris versicolor

    # TODO : predict the outputs for the testing data
    params = np.delete(params,0,0)
    y = np.matmul(test_data, params.T)
    for i in range(y.shape[0]):
        if y[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
    out = y

    return out

################################################################################
#  save testing results to a file
################################################################################
def save_predictions(outputfile, test_data, out) :
    # input:
    #    outputfile - path of the output file
    #    test_data  - a n x 4 numpy ndarray (dtype = np.float64) holding the
    #                 values of the attributes for the testing data (n being
    #                 number of samples)
    #    out - a n-D numpy ndarray (dtype = np.uint8) holding the predictions
    #          for testing data

    try :
        file = open(outputfile, 'w')
        file.write('{} samples\n'.format(len(test_data)))
        file.write('sepal length (cm), sepal width (cm), petal length (cm), petal width (cm), species (0 for setosa, 1 for versicolor)\n')
        for i in range(len(test_data)) :
            file.write('{:.1f}  {:.1f}  {:.1f}  {:.1f}  {:.0f}\n'.format(test_data[i,0], test_data[i,1], test_data[i,2], test_data[i,3], out[i]))
    except :
        print('Error occurs in writting output to \'{}\'.'.format(outputfile))
        sys.exit(1)

################################################################################
#  load data from a file
################################################################################
def load_data(inputfile) :
    # input:
    #    inputfile - path of the file containing the training / testing data
    # return:
    #    data - a n x 5 numpy ndarray (dtype = np.float64) holding values of the
    #           attributes and labels for the training data (n being the number
    #           of samples); or a n x 4 numpy ndarray holding values of the
    #           attributes for the testing data

    try :
        file = open(inputfile, 'r')
        lines = file.readlines()
        data = np.array([np.fromstring(line.strip(), dtype = np.float64, sep=' ') for line in lines[2:]])
        file.close()
        return data
    except :
        print('Error occurs in loading data from \'{}\'.'.format(inputfile))
        sys.exit(1)

################################################################################
#  main
################################################################################
def main() :
    parser = argparse.ArgumentParser(description = 'COMP3317 Assignment 5')
    parser.add_argument('-i', '--train', type = str, default = 'train.data',
                        help = 'filename of training data')
    parser.add_argument('-t', '--test', type = str, default = 'test.data',
                        help = 'filename of testing data')
    parser.add_argument('-o', '--output', type = str,
                        help = 'filename for outputting predictions')
    args = parser.parse_args()

    print('-------------------------------------------')
    print('COMP3317 Assignment 5 - Perceptron learning')
    print('training data : {}'.format(args.train))
    print('testing data : {}'.format(args.test))
    print('output file : {}'.format(args.output))
    print('-------------------------------------------')

    # load the training data
    train_data = load_data(args.train)
    print('{} training data loaded from \'{}\'...'.format(len(train_data), args.train))

    # train the perceptron with the training data
    print('perform perceptron learning...')
    params = train(train_data)

    # load the testing data
    test_data = load_data(args.test)
    print('{} testing data loaded from \'{}\'...'.format(len(test_data), args.test))

    # use the learned parameters to predict outputs for the testing data
    print('predicting the outputs for the testing data...')
    out = predict(test_data, params)
    print(out)

    # save the predictions to a file
    if args.output :
        save_predictions(args.output, test_data, out)
        print('predictions saved to \'{}\'...'.format(args.output))

if __name__ == '__main__':
    main()
