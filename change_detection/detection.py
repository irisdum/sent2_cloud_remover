import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
import warnings
import cv2


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
    vector_set = vector_set - mean_vec
    return vector_set, mean_vec


def find_FVS(EVS, diff_image, mean_vec, kernel_dim, padding="symmetric"):
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
    #print("count", count)
    #print(len(feature_vector_set))
    #print(np.array(feature_vector_set).shape)
    #print(EVS.shape)
    FVS = np.dot(feature_vector_set, EVS)
    FVS = FVS - mean_vec
    print("\nfeature vector space size", FVS.shape)
    assert FVS.shape[0] == input_dim[1] * input_dim[2], "Dimension problem FVS shape is {} and should be {}".format(
        FVS.shape[0], input_dim[1] * input_dim[2])
    return FVS


def clustering(FVS, components, input_shape):
    kmeans = KMeans(components, verbose=0)
    kmeans.fit(FVS)
    output = kmeans.predict(FVS)
    count = Counter(output)
    least_index = min(count, key=count.get)
    change_map = np.reshape(output, (input_shape[0], input_shape[1]))
    return least_index, change_map


def map_detection(image1, image2, kernel_dim=4, n_components="full", k=2, padding="symmetric"):
    assert image1.shape[1] % kernel_dim == 0, "To not loose any information please use a kernel_dim {} that divides the" \
                                              " image dimension {}".format(kernel_dim, image1.shape[1])

    diff_image = abs(image1 - image2)
    input_dim = (diff_image.shape[1], diff_image.shape[2])
    vector_set, mean_vec = find_vector_set(diff_image, n_channel=diff_image.shape[0], kernel_dim=kernel_dim)
    # RUN ACP
    pca = PCA(n_components=n_components)
    pca.fit(vector_set)
    EVS = pca.components_
    #print("compo", EVS.shape)
    print("The amount of variance explained by the componants of ACP", pca.explained_variance_)
    print("We are using a symetric padding to add the missing dimension ")

    FVS = find_FVS(EVS, diff_image, mean_vec, kernel_dim, padding)
    least_index, change_map = clustering(FVS, k, input_dim)
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
