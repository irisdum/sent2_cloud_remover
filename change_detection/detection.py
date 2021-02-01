import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
import warnings
import cv2
import pickle
from utils.image_find_tbx import create_safe_directory


def find_vector_set(diff_image: np.array, n_channel=1, kernel_dim=5):
    """

    Args:
        kernel_dim: the input array will be cut into blocks of size (n_channel,kernel_dim,kernel_dim)
        diff_image:  3D images. If initally 2D images add a dimension to axis 0 in order to have for instance
        (1,n,n)
        n_channel: the nber of Channel. Set to 1 if 2D images

    Returns:

    """

    i = 0
    j = 0
    assert diff_image.shape[0] < diff_image.shape[1], "Should be channel first image (n_c,n,n) not {}".format(
        diff_image.shape)
    new_size = (diff_image.shape[1], diff_image.shape[2])
    vector_set = np.zeros((int(new_size[0] * new_size[1] / kernel_dim ** 2), (kernel_dim ** 2) * n_channel))
    if new_size[0] % kernel_dim != 0:
        warnings.warn("Warning, the input image dimension {} is not divided by the kernel size {}\n"
                      " pixels might be not taken into account".format(new_size, kernel_dim))

    while j < new_size[0]:
        k = 0
        while k < new_size[1]:
            block = diff_image[:, j:j + kernel_dim, k:k + kernel_dim]
            feature = block.ravel()
            vector_set[i, :] = feature
            i = i + 1
            k = k + kernel_dim
        j = j + kernel_dim

    mean_vec = np.mean(vector_set, axis=0)
    print("vec_set",vector_set.shape)
    print("mean set",np.mean(vector_set, axis=1).shape)
    vector_set = vector_set - mean_vec
    return vector_set, mean_vec


def find_FVS(EVS, diff_image, mean_vec, kernel_dim, padding="symmetric"):
    """

    Args:
        EVS: Eigen Vector space
        diff_image:
        mean_vec: mean vector
        kernel_dim:
        padding:

    Returns:

    """
    if kernel_dim % 2 == 0:
        i_before = int(kernel_dim / 2)
        i_after = int(i_before)
    else:
        i_before = int(kernel_dim // 2)
        i_after = int(i_before + 1)
    input_dim = diff_image.shape
    #print(((0, 0), (i_before, i_after - 1), (i_before, i_after - 1)), diff_image.shape)
    diff_image = np.pad(diff_image,
                        ((0, 0), (i_before, i_after - 1), (i_before, i_after - 1)), padding)

    #print("Before padding shape {} then {}".format(input_dim, diff_image.shape))
    feature_vector_set = []
    i = i_before
    #print(diff_image.shape[1] - i_after)
    #print(diff_image.shape[0] - i_after)
    count = 0
    while i < diff_image.shape[1] - i_after + 1:
        j = i_before
        while j < diff_image.shape[2] - i_after + 1:
            # print(i - i_before,i + i_after, j - i_before,j + i_after)
            block = diff_image[:, i - i_before:i + i_after, j - i_before:j + i_after]
            # print(block.shape)
            feature = block.flatten()
            # print(feature.shape)
            feature_vector_set.append(feature)
            j = j + 1
            count += 1
        i = i + 1
    print("before multiply shape feature vector {}".format(np.array(feature_vector_set).shape))
    #print(EVS.shape)
    FVS = np.dot(np.array(feature_vector_set)-mean_vec, EVS) # Feature_vector_space (Npixels,h*h*nchannel) EVS dim (h*h*nchannel,S number PCA components)
    #FVS dim will be (Npixel, S)
    #FVS = FVS - mean_vec
    print("\nfeature vector space size", FVS.shape)
    assert FVS.shape[0] == input_dim[1] * input_dim[2], "Dimension problem FVS shape is {} and should be {}".format(
        FVS.shape[0], input_dim[1] * input_dim[2])
    return FVS


def clustering(FVS, components, input_shape, kmeans=None):
    """

    Args:
        kmeans:
        FVS: Feature vector space
        components: nber of cluster K in Kmeans
        input_shape: (N*N)

    Returns:

    """
    if kmeans is None:
        kmeans = KMeans(components, verbose=0)
        kmeans.fit(FVS)
    output = kmeans.predict(FVS)
    count = Counter(output)
    least_index = min(count, key=count.get)
    change_map = np.reshape(output,input_shape)
    return least_index, change_map,kmeans


def map_detection(image1, image2, kernel_dim=4, n_components="full", k=2, padding="symmetric"):
    """

    Args:
        image1: a numpy array of dimension (n_channel, N,N)
        image2:  a numpy array of dimension (n_channel, N,N)
        kernel_dim: size of the non-overlapping block, h in the article
        n_components: number of components for PCA, S in the article
        k: nber of cluster in the K means
        padding: type of padding applied with tf.pad

    Returns:

    """
    assert image1.shape[1] % kernel_dim == 0, "To not loose any information please use a kernel_dim {} that divides the" \
                                              " image dimension {}".format(kernel_dim, image1.shape[1])

    diff_image = abs(image1 - image2)
    input_dim = (diff_image.shape[1], diff_image.shape[2])
    vector_set, mean_vec = find_vector_set(diff_image, n_channel=diff_image.shape[0], kernel_dim=kernel_dim)
    # RUN ACP
    pca = PCA(n_components=n_components)
    pca.fit(vector_set)
    print(pca.components_.shape)
    EVS = np.transpose(pca.components_) #shape will be (features,n_components)
    #TODO try if works better with transpose
    print("compo", EVS.shape)
    print("The amount of variance explained by the componants of ACP", pca.explained_variance_)
    print("We are using a symetric padding to add the missing dimension ")

    FVS = find_FVS(EVS, diff_image, mean_vec, kernel_dim, padding)
    least_index, change_map,kmeans = clustering(FVS, k, input_dim)
    change_map[change_map == least_index] = 255
    change_map[change_map != 255] = 0

    change_map = change_map.astype(np.uint8)
    kernel = np.asarray(((0, 0, 1, 0, 0),
                         (0, 1, 1, 1, 0),
                         (1, 1, 1, 1, 1),
                         (0, 1, 1, 1, 0),
                         (0, 0, 1, 0, 0)), dtype=np.uint8)
    cleanChangeMap = cv2.erode(change_map, kernel)

    return diff_image, change_map, cleanChangeMap


def ACP_on_batch(batch1, batch2, kernel_dim=4, n_components=3, k=2, padding="symmetric",band=None, path_save=None,load_dir=None):
    """
    NOT FINISEHD DO NOT LOOK !
    Args:
        load_dir:
        path_save:
        band: should be a list, by default 0 [0]
        batch1:
        batch2:
        kernel_dim:
        n_components:
        k:
        padding:
        save:

    Returns:

    """
    if band is None:
        band=[0]
    list_vector_set = []
    print("Batch size {}".format(batch1.shape[0]))
    if load_dir is  None:
        for i in range(batch1.shape[0]):  # TODO parallelize to improve the computation time if too slow
            image1 = batch1[i, :, :, band]
            image2 = batch2[i, :, :, band]

            diff_image = abs(image1 - image2)
            # print(diff_image.shape)
            input_dim = (diff_image.shape[1], diff_image.shape[2])
            vector_set, mean_vec = find_vector_set(diff_image, n_channel=diff_image.shape[0], kernel_dim=kernel_dim)
            list_vector_set += [vector_set]

        batch_vect_set = np.array(list_vector_set)
        print("batch vect set {}".format(batch_vect_set.shape))
        vector_set_dim = (list_vector_set[0].shape[0], list_vector_set[0].shape[1])
        print(vector_set_dim)
        batch_vect_set = batch_vect_set.reshape(
            (len(list_vector_set) * vector_set_dim[0], vector_set_dim[1]))  # a way to concatenate all the data
        print("after reshape {}".format(batch_vect_set.shape))

     # batch_vect_set has the features built for the whole input batches
        pca = PCA(n_components=n_components)
        pca.fit(batch_vect_set)
        EVS = np.transpose(pca.components_)

        batch_mean_vect = np.mean(batch_vect_set, axis=0)
        kmeans=None
    else:
        EVS=np.load(load_dir+"EVS")
        batch_mean_vect=np.load(load_dir+"mean_vect")
        kmeans=pickle.load(open("{}kmeans.pkl".format(path_save), "rb"))
    list_FVS = []

    for i in range(batch1.shape[0]):
        image1 = batch1[i, :, :, band]
        image2 = batch2[i, :, :, band]
        diff_image = abs(image1 - image2)
        # print("FVS diff im {}".format(diff_image.shape))
        FVS = find_FVS(EVS, diff_image, batch_mean_vect, kernel_dim, padding)
        list_FVS += [FVS]
    batch_FVS = np.array(list_FVS)
    dim = list_FVS[0].shape
    batch_FVS = batch_FVS.reshape(len(list_FVS) * dim[0], dim[1])
    print("FVS shape{}".format(batch_FVS.shape))
    shape_change_map = (batch1.shape[0], batch1.shape[1], batch1.shape[2])
    least_index, change_map,kmeans = clustering(batch_FVS, k,
                                                shape_change_map,kmeans=kmeans)
    change_map[change_map == least_index] = 255
    change_map[change_map != 255] = 0
    change_map.reshape(shape_change_map)
    kernel = np.asarray(((0, 0, 1, 0, 0),
                         (0, 1, 1, 1, 0),
                         (1, 1, 1, 1, 1),
                         (0, 1, 1, 1, 0),
                         (0, 0, 1, 0, 0)), dtype=np.uint8)
    batch_clean_change_map = np.zeros(change_map.shape)
    for i in range(change_map.shape[0]):
        batch_clean_change_map[i, :, :] = cv2.erode(change_map[i, :, :].astype(np.uint8), kernel)

    if path_save is not None: #TO SAVE THE MODEL
        assert path_save[-1]=="/".format("Wrong path_save name should be a directory path {}".format(path_save))
        create_safe_directory(path_save)
        np.save(path_save+"EVS",EVS)
        np.save(path_save+"mean_vect",batch_mean_vect)
        pickle.dump(kmeans, open("{}kmeans.pkl".format(path_save), "wb"))

    return None, change_map, batch_clean_change_map