import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth
from fcmeans import FCM
from sklearn.mixture import GaussianMixture as GMM
from imutils import paths
import matplotlib as mpl
import scipy.ndimage as ndi 
from sklearn.preprocessing import StandardScaler

SUPPORTED_SPACE_COLOR = ["RGB", "L*A*B*", "HSV","GRAY"]

def fuzzy_cmeans(image, number_of_cluster, fcm_centers=None):
    """
    """
    print("[INFO] ... \n Starting fuzzy C Means with image shape : {} .".format(image.shape))
    original_shape = image.shape
    img = image.reshape(image.shape[0]*image.shape[1], image.shape[2])
    fcm = FCM(n_clusters=number_of_cluster)
    if str(type(fcm_centers)) == "<class 'NoneType'>" :
        fcm.fit(img)
        print("looking for centers")
    else :
        fcm.fit(img)
        fcm.centers = fcm_centers
    fcm.predict(img)
    fcm_centers = fcm.centers
    #print(fcm_centers)
    fcm_labels  = fcm.u.argmax(axis=1)
    return np.reshape(fcm_labels, original_shape[0:2]), fcm_centers, list(np.unique(fcm_labels, return_counts=True))

def dbscan(image):
    """
    """
    original_shape = image.shape
    print("[INFO] ... \n Starting DbScan with image shape : {} .".format(image.shape))
    feature_image=np.float32(np.reshape(image, [-1, original_shape[-1]]))
    db = DBSCAN(eps=5, min_samples=100, metric = 'euclidean',algorithm ='auto')
    db.fit_predict(feature_image)
    labels = db.labels_
    labels+=1
    print(np.unique(labels))
    number_of_cluster = len(np.unique(labels))
    return np.reshape(labels, original_shape[:2]), number_of_cluster, list(np.unique(labels, return_counts=True))


def gaussian(image, number_of_cluster):
    print("[INFO] ... \n Starting gaussian with image shape : {} .".format(image.shape))
    original_shape = image.shape
    img = image.reshape(image.shape[0]*image.shape[1], image.shape[2])
    gmm = GMM(n_components=number_of_cluster).fit(img)
    labels = gmm.predict(img)
    return np.reshape(labels, original_shape[:2]), list(np.unique(labels, return_counts=True))

def preprocess_input():
    pass

def read_image(path, space_color="RGB", chanels_indexs=None, scale_percent=100):
    assert space_color.upper() in SUPPORTED_SPACE_COLOR , "Wrong space colors ..."
    image = cv2.imread(path)
    while image.shape[0] * image.shape[1] > 40000 :
        scale_percent-=1
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        image = cv2.resize(image, dim)
    if space_color.upper() == "HSV" :
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if chanels_indexs != None :
            image_to_return = image_hsv[:,:,chanels_indexs]
        else :
            image_to_return = image_hsv
    elif space_color.upper() == "RGB" :
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if chanels_indexs != None :
            image_to_return = image_rgb[:,:,chanels_indexs]
        else :
            image_to_return = image_rgb
    elif space_color.upper() == "GRAY" :
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_to_return = np.expand_dims(image_gray,axis=2)
    return image_to_return, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)