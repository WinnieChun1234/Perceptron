# Perceptron

This is part of the fifth assignmnet of my computer vision course in the university. The assignment comes with partially completed Python programs and the tasks are to train a perceptron for classifying Iris setosa and Iris versicolor (i.e., a binary classification problem), and to make predictions based on attributes of the flowers.


### To train the perceptron and make predictions on new image, I have done the following features:

1. train() - training a perceptron for classifying Iris setosa and Iris versicolor (i.e., a binary classification problem);
    1. Initialize the parameters of the perceptron to small random numbers.
       Two 5-D array using np.random.normal() and np.random.standard_normal() respectively, namely params and weight
    2. Update the parameters of the perceptron based on the perceptron learning algorithm as described in the lecture notes.
          1.   extract input values from train_data and augment the inputs with x0 = 1
                1. assign train_data as variable x
                2. insert 0 into the first column of x
                3. delete the last column of x since it is the labels
          2.  extract target outputs from train_data
                assign the last column of train_data as t, where t represents the vector of labels/truth
          3.  epeat until converged or max no. of iterations has been reached
                1. set up counter
                2. set up while-loop with maximum iterations as max_itr
                3. compute the z value as the output vector y = x * params
                4. by the binary function, return the entry of y as 0 if the entry of y is negative, otherwise 1
                5. compute the error term err = y - t
                6. update the weight = params - x.T * err, where params is the old-weight
                7. check if the new weight (weight) is the same as the old weight ( params).
                      if yes, then break the loop
                      if no, assign the new weight(weight) to the old weight (params)
                8. increase the count by 1
          iv.  return the params
2. predict() - for making predictions based on attributes of the flowers.
    Predict class labels based on the learned parameters of the perceptron.
      1.  delete the first entry of params
      2.  compute the output vector y = test_data * params.T
      3.  set up for-loop from 0 to n (y is a n-D array)
      4.  check the entry of y. By the binary function, return the entry of y as 0 if the entry of y is negative, otherwise 1
      5.  return out as the vector y  
