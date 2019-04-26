<b>Problem Statement 1 </b>:

For this problem, your task is to implement a simplified version of the Probabilistic
PCA (PPCA) model and play with it on a dataset of face images. 

<b> Description of dataset </b>:

The provided dataset contains a total of 165
face images of 15 individuals. Each image is 64x64 grayscale.

<b> Implementation </b>:

The first simplication will be that we will assume sigma^2 = 0 and the second simplification will be that we will
use ALT-OPT instead of EM. The overall algorithm will basically be a simpler version of the algorithm given
on slide 10 of lecture 18. Since you are using ALT-OPT instead of EM, you don’t even have to compute the
posterior or the expectation of the latent variables. It is really that simple! :-) You will effectively be doing PCA
but using ALT-OPT instead of eigendecomposition! You will run your ALT-OPT algorithm for a fixed number of iteration (100 iterations), though it might converge to a reasonable solution in fewer iterations.

Run your implementation with K = 10; 20; 30; 40; 50; 100 and using all the data (165 images). After the model
has been learned, for each value of K, reconstruct any 5 of the images (of your choice!) and visually inspect the
reconstructed images, comparing them with their original versions. The goal is to see how (or whether)
increasing K improves the reconstruction (or not). What do you observe visually as K increases? Does the
reconstruction get better or worse? Briefly explain the reason for what you observe.

For each K, also show 10 of the basis images that you think
look sort of interesting (for K = 10, it will mean showing all the columns of W; for K = 20; 30; 40; 50; 100,
you will need to select a subset of 10 columns fromWand show those).

<b>Problem Statement 2 </b>: 

Your goal is to run K-means clustering on two-dimensional embedding of this data. To get the two-dimensional
embeddings, you will use two approaches: PCA and tSNE. You can use any implementation of PCA and
tSNE, e.g., from sklearn if you are using Python.

<b> Description of dataset </b>:

You are provided a subset of the MNIST data consisting of 10,000 images of digits
0-9. Each image is of size 28 X 28 (784 dimensional). The dataset also contains the digit labels.

<b> Implementation </b>:

Run K-means with K = 10 for both cases (PCA based 2D embeddings and tSNE based 2D embeddings), using
10 different initializations and show the clustering results in form of the plots of the obtained clusterings with
each cluster shown in a different color. Visually, which of the two 2D embedding methods (PCA/tSNE) seems
to work better for the clustering task?
