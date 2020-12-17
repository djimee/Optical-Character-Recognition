"""Dummy classification system.

Skeleton code for a assignment solution.

To make a working solution you will need to rewrite parts
of the code below. In particular, the functions
reduce_dimensions and classify_page currently have
dummy implementations that do not do anything useful.

version: v1.0
"""
import numpy as np
import utils.utils as utils
import scipy.linalg

def divergence(class1, class2):
    """Compute a vector of 1-D divergences - taken from lab 6/7
    Params:
    class1 - data matrix for class 1, each row is a sample
    class2 - data matrix for class 2
    
    Returns: 
    d12 - a vector of 1-D divergence scores
    """

    # Compute the mean and variance of each feature vector element
    m1 = np.mean(class1, axis=0)
    m2 = np.mean(class2, axis=0)
    v1 = np.var(class1, axis=0)
    v2 = np.var(class2, axis=0)

    # Plug mean and variances into the formula for 1-D divergence.
    # (Note that / and * are being used to compute multiple 1-D
    #  divergences without the need for a loop)
    
    d12 = 0.5 * (v1 / v2 + v2 / v1 - 2) + 0.5 * ( m1 - m2 ) * (m1 - m2) * (1.0 / v1 + 1.0 / v2)

    return d12

def load_wordlist(wordlist_filename):
    """Import wordlist text file - used word list: http://www.mieliestronk.com/corncob_lowercase.txt
    """
    # load the wordlist from the file
    print ("Loading word list from file...")
    wordlist = list()
    with open(wordlist_filename) as f:
        for line in f:
            wordlist.append(line.rstrip('\n'))
    print (" ", len(wordlist), "words loaded.")

    return wordlist

def calculate_principal_components(train_data, numPC):
    """Method to calculate the the principal components - taken from lab 6/7

    Params:
    train_data - training data feature vectors stored as 
        rows in a matrix
    countPC - number of principal components to compute 

    Returns:
    v - principal componeents stored in a vector
    """
    
    # computing the principal components of the first numPC components
    covx = np.cov(train_data, rowvar=0)
    N = covx.shape[0]
    w, v = scipy.linalg.eigh(covx, eigvals=(N - numPC, N - 1))
    v = np.fliplr(v)

    return v

def reduce_dimensions(feature_vectors_full, model):
    """Reduces the dimensions by finding the 10 best features

    Params:
    feature_vectors_full - feature vectors stored as rows
       in a matrix
    model - a dictionary storing the outputs of the model
       training stage

    Returns:
    features - 10 best features obtained in dimensionality reduction
    """
    # extract training labels from the model
    train_labels = np.array(model["labels_train"])

    # feature selection algorithm
    divergences = []
    sorted_indexes = []

    # do nested for loop to get divergences between all pairs, append results to array
    for i in range (1,27):
        for j in range (1,27):
            first = feature_vectors_full[train_labels[:] == i, :]
            second = feature_vectors_full[train_labels[:] == j, :]
            if first.shape[0] <= 1 or second.shape[0] <= 1:
                continue 
            d12 = divergence(first, second)
            divergences.append(d12)

    # create vector of 10 zeros to store sorted indexes 
    sorted_indexes = np.zeros(10)

    # sum values for divergence into sorted divergences vector
    for index in divergences:
        sorted_indexes += index

    # sort the divergences and print the top 10 features
    sorted_indexes = np.argsort(-sorted_indexes)
    features = sorted_indexes[0:10]
    
    return features

def get_bounding_box_size(images):
    """Compute bounding box size given list of images."""
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    return height, width

def images_to_feature_vectors(images, bbox_size=None):
    """Reformat characters into feature vectors.

    Takes a list of images stored as 2D-arrays and returns
    a matrix in which each row is a fixed length feature vector
    corresponding to the image.abs

    Params:
    images - a list of images stored as arrays
    bbox_size - an optional fixed bounding box size for each image
    """

    # If no bounding box size is supplied then compute a suitable
    # bounding box by examining sizes of the supplied images.
    if bbox_size is None:
        bbox_size = get_bounding_box_size(images)

    bbox_h, bbox_w = bbox_size
    nfeatures = bbox_h * bbox_w
    fvectors = np.empty((len(images), nfeatures))
    for i, image in enumerate(images):
        padded_image = np.ones(bbox_size) * 255
        h, w = image.shape
        h = min(h, bbox_h)
        w = min(w, bbox_w)
        padded_image[0:h, 0:w] = image[0:h, 0:w]
        fvectors[i, :] = padded_image.reshape(1, nfeatures)
    return fvectors

def process_training_data(train_page_names):
    """Perform the training stage and return results in a dictionary.

    Params:
    train_page_names - list of training page names
    """
    print("Reading data")
    images_train = []
    labels_train = []
    for page_name in train_page_names:
        images_train = utils.load_char_images(page_name, images_train)
        labels_train = utils.load_labels(page_name, labels_train)
    labels_train = np.array(labels_train)

    print("Extracting features from training data")
    bbox_size = get_bounding_box_size(images_train)
    fvectors_train_full = images_to_feature_vectors(images_train, bbox_size)

    model_data = dict()
    model_data["labels_train"] = labels_train.tolist()
    model_data["bbox_size"] = bbox_size

    # calculate the first 50 principal components and the mean 
    # of the training data and input values into the model
    print("Inputting principal components and mean into model")
    principal_components = calculate_principal_components(fvectors_train_full, 50)
    model_data["principal_components"] = principal_components.tolist()
    mean = np.mean(fvectors_train_full)
    model_data["mean"] = mean.tolist()

    print("Reducing to 10 dimensions")
    # calculate best features from projected training feature vectors (pcatrain_data) and input into model
    pcatrain_data = np.dot((fvectors_train_full - np.mean(fvectors_train_full)), principal_components)
    features = reduce_dimensions(pcatrain_data, model_data)
    model_data["features"] = features.tolist()

    # find feature vectors corresponding to the best features in pcatrain_data and input into model
    pcatrain_data_features = pcatrain_data[:, features]
    model_data["fvectors_train"] = pcatrain_data_features.tolist()

    # load and input the wordlist into the model
    wordlist = load_wordlist("wordlist.txt")
    model_data["wordlist"] = wordlist

    return model_data

def load_test_page(page_name, model):
    """Load test data page.

    This function must return each character as a 10-d feature
    vector with the vectors stored as rows of a matrix.

    Params:
    page_name - name of page file
    model - dictionary storing data passed from training stage
    """
    bbox_size = model["bbox_size"]
    images_test = utils.load_char_images(page_name)
    fvectors_test = images_to_feature_vectors(images_test, bbox_size)

    # Perform the reconstruction
    features = np.array(model["features"])
    principal_components = np.array(model["principal_components"])
    mean = np.array(model["mean"])

    # Get the feature vectors corresponding to the best features found from doing feature selection
    pcatest_data = np.dot((fvectors_test - mean), principal_components)
    fvectors_test_reduced = pcatest_data[:, features]

    return fvectors_test_reduced

def classify(train, train_labels, test, features=None):
    """Perform nearest neighbour classification - taken from lab 6/7
    Params:
    train - training data feature vector stores in rows as a matrix
    train_labels - training labels for the training data
    test - test data feature vector stores in rows as a matrix
    features - features, if none provided, all of them are used
    """

    # Use all feature is no feature parameter has been supplied
    if features is None:
        features=np.arange(0, train.shape[1])

    # Select the desired features from the training and test data
    train = train[:, features]
    test = test[:, features]
    
    # Super compact implementation of nearest neighbour 
    x=np.dot(test, train.transpose())
    modtest=np.sqrt(np.sum(test * test, axis=1))
    modtrain=np.sqrt(np.sum(train * train, axis=1))
    dist=x / np.outer(modtest, modtrain.transpose()) # cosine distance
    nearest=np.argmax(dist, axis=1)

    # Label returned using the nearest neighbour classification
    label = train_labels[0, nearest]

    return label

def classify_page(page, model):
    """Classify the page using the classify method

    Params:
    page - matrix, each row is a feature vector to be classified
    model - dictionary, stores the output of the training stage
    """
    fvectors_train = np.array(model["fvectors_train"])
    labels_train = (np.array(model["labels_train"]))
    
    return classify(fvectors_train, labels_train[np.newaxis], page)

def correct_errors(page, labels, bboxes, model):
    """Dummy error correction. Returns labels unchanged.

    Params:
    page - 2d array, each row is a feature vector to be classified
    labels - the output classification label for each feature vector
    bboxes - 2d array, each row gives the 4 bounding box coords of the character
    model - dictionary, stores the output of the training stage
    """


    return labels