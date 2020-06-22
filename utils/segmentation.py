# File with the function ML to use cluster methos for segmentation
from scipy import ndimage
import  numpy as np
from sklearn import cluster
def denoise_img(img):
    return ndimage.median_filter(img, 3)

def denoise_batch(batch_img):
    """:param batch_img an array of dim 3 (nb_elem,m,m)
    :returns the batch with median filter applied to all tiles"""
    assert len(batch_img.shape)==3, "Input batch should be  3 dimensions "
    output_batch=np.ones(batch_img.shape)
    for i in range(batch_img.shape[0]):
        output_batch[i,:,:]=denoise_img(batch_img[i,:,:])
    return output_batch


def km_clust(array, n_clusters,rd_state=2):
    # Create a line array, the lazy way
    X = array.reshape((-1, 1))
    # Define the k-means clustering problem
    k_m = cluster.KMeans(n_clusters=n_clusters, n_init=4,random_state=rd_state)
    # Solve the k-means clustering problem
    k_m.fit(X)
    # Get the coordinates of the clusters centres as a 1D array
    values = k_m.cluster_centers_.squeeze()
    # Get the label of each point
    labels = k_m.labels_
    return k_m




