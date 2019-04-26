<b> Problem statement </b>:

Implement a generative classification model for this data assuming Gaussian class-conditional distributions of
the positive and negative class examples.

<b>Description of dataset</b>:

You are provided a dataset in the file binclass.txt. In this file, the first two numbers on each line
denote the two features and the third number is the binary label.

<b>Implementation</b>:

Use MLE estimates for the unknown parameters. Your implementation need not be specific to two-dimensional inputs and
it should be almost equally easy to implement it such that it works for any number of features,

On a two-dimensional plane, plot the examples from both the classes (use red color for positives and blue color
for negatives) and the learned decision boundary for this model. Note that we are not providing any separate
test data. Your task is only to learn the decision boundary using the provided training data and visualize it.

Finally, try out a Support Vector Machine (SVM) classifier (with linear kernel) on this data and show the learn
decision boundary. For this part, you do not need to implement SVM. There are many nice implementations
of SVM available, such as the one in scikit-learn and the very popular libSVM toolkit. Assume the “C”
hyperparameter of SVM in these implementations to be 1.

Repeat the same experiments but now using a different dataset binclassv2.txt.
Looking at the results of both the parts, which of the two models (generative classification with Gaussian classconditional
and SVM) do you think seems to work better for each of these datasets, and in general?
