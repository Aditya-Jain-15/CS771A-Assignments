<b>Problem Statement</b>:

Your task is to implement and test prototype based classification using the provided
dataset. 

<b>Description of dataset</b>:

In this dataset, each input represents an image of an animal and output is the class (what this animal is). The dataset
has a total of 50 classes and the number of features for each input is 4096. However, we are going to give a small twist to the basic prototype based classification problem. The training set provided to you has only examples from 40 of the classes. We will refer to these 40 classes as “seen classes” (have training data for these classes) and the remaining 10 classes as “unseen classes” (do not have training data for these classes). The test inputs will be only from these 10 unseen clases.

<b>Implementation</b>:

Recall that prototype based classification requires computing the mean of each class. While computing the
means of the 40 seen classes is easy (since we have training data from these classes), what we actually need is
the mean of the remaining 10 classes (since these are our test classes). How do we get these means?

Well, we clearly need some additional information about the classes in order to solve this problem (without
that there is no hope of solving this problem). To this end, you are provided an 85-dimensional class attribute
vector for each class (both seen as well as unseen classes). Each class attribute vector contains
information about that class and consists of 85 binary-valued attributes representing the class (e.g., whether this
animal has stripes).
Now consider two ways how these class attribute vectors can be used to obtain the means of unseen classes:

 Method 1: Model the mean of each unseen class as a convex combination of the means of the 40 seen classes.

 Method 2: Train a linear model that can predict the mean of any class using its class attribute vector. We can train this linear model using our training data and then apply it to predict for each unseen class using its class attribute vector. Note that this can be posed as a multi-output regression problem.
