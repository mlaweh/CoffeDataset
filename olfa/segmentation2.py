import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth
import skfuzzy as fuzz
from sklearn.mixture import BayesianGaussianMixture as BGM
from fcmeans import FCM
from pyclustering.cluster.fcm import fcm
from sklearn.mixture import GaussianMixture as GMM
from imutils import paths
import matplotlib as mpl
import scipy.ndimage as ndi 
from pyclustering.utils import read_image, draw_image_mask_segments
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import math

SUPPORTED_SPACE_COLOR = ["RGB", "L*A*B*", "HSV","GRAY", "HSI"]


def RGB_TO_HSI(img):

    with np.errstate(divide='ignore', invalid='ignore'):

        #Load image with 32 bit floats as variable type
        bgr = np.float32(img)/255

        #Separate color channels
        blue = bgr[:,:,0]
        green = bgr[:,:,1]
        red = bgr[:,:,2]

        #Calculate Intensity
        def calc_intensity(red, blue, green):
            return np.divide(blue + green + red, 3)

        #Calculate Saturation
        def calc_saturation(red, blue, green):
            minimum = np.minimum(np.minimum(red, green), blue)
            saturation = 1 - (3 / (red + green + blue + 0.001) * minimum)

            return saturation

        #Calculate Hue
        def calc_hue(red, blue, green):
            hue = np.copy(red)

            for i in range(0, blue.shape[0]):
                for j in range(0, blue.shape[1]):
                    hue[i][j] = 0.5 * ((red[i][j] - green[i][j]) + (red[i][j] - blue[i][j])) / \
                                math.sqrt((red[i][j] - green[i][j])**2 +
                                        ((red[i][j] - blue[i][j]) * (green[i][j] - blue[i][j])))
                    hue[i][j] = math.acos(hue[i][j])

                    if blue[i][j] <= green[i][j]:
                        hue[i][j] = hue[i][j]
                    else:
                        hue[i][j] = ((360 * math.pi) / 180.0) - hue[i][j]

            return hue

        #Merge channels into picture and return image
        hsi = cv2.merge((calc_hue(red, blue, green), calc_saturation(red, blue, green), calc_intensity(red, blue, green)))
    hsi[:,:, 0]=hsi[:,:, 0] * 255
    hsi[:,:, 1]=hsi[:,:, 1] * 255
    hsi[:,:, 2]=hsi[:,:, 2] * 255
    return hsi.astype('uint8')


def fuzzy_cmeans(image, number_of_cluster, fcm_centers=None):
    """
    """
    print("[INFO] ... \n Starting fuzzy C Means with image shape : {} .".format(image.shape))
    original_shape = image.shape
    img = image.reshape(image.shape[0]*image.shape[1], image.shape[2])
    fcm = FCM(n_clusters=number_of_cluster, centers=fcm_centers)
    fcm.fit(img)
    fcm.predict(img)
    #print(fcm_centers)
    fcm_labels  = fcm.u.argmax(axis=1)
    return np.reshape(fcm_labels, original_shape[0:2]), fcm_centers, list(np.unique(fcm_labels, return_counts=True))
def fuzzy_cmeans_normalized(image, number_of_cluster, fcm_centers=None):
    """
    """
    print("[INFO] ... \n Starting fuzzy C Means with image shape : {} .".format(image.shape))
    original_shape = image.shape
    img = image.reshape(image.shape[0]*image.shape[1], image.shape[2])
    scaler = StandardScaler()
    img = scaler.fit_transform(img)
    fcm = FCM(n_clusters=number_of_cluster, centers=fcm_centers)
    fcm.fit(img)
    fcm.predict(img)
    prob = fcm.u
    pd.DataFrame(prob[:].round(3)).to_csv("fcm_prob_results.csv")
    #print(fcm_centers)
    fcm_labels  = fcm.u.argmax(axis=1)
    return np.reshape(fcm_labels, original_shape[0:2]), fcm_centers, list(np.unique(fcm_labels, return_counts=True))
def dbscan(image):
    """
    """
    original_shape = image.shape
    print("[INFO] ... \n Starting DbScan with image shape : {} .".format(image.shape))    
    feature_image=np.float32(np.reshape(image, [-1, original_shape[-1]]))
    eps=1
    while True :
        db = DBSCAN(eps=eps, min_samples=100, metric = 'euclidean',algorithm ='auto')
        db.fit_predict(feature_image)
        if len(np.unique(db.labels_)) == 3 :
            break
        eps+=1
    db = DBSCAN(eps=eps, min_samples=100, metric = 'euclidean',algorithm ='auto')
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
    prob = gmm.predict_proba(img)
    pd.DataFrame(prob[:].round(3)).to_csv("gmm_prob_results.csv")
    return np.reshape(labels, original_shape[:2]), list(np.unique(labels, return_counts=True))

def bayesian_gaussian(image, number_of_cluster):
    print("[INFO] ... \n Starting gaussian with image shape : {} .".format(image.shape))
    original_shape = image.shape
    img = image.reshape(image.shape[0]*image.shape[1], image.shape[2])
    gmm = BGM(n_components=number_of_cluster).fit(img)
    labels = gmm.predict(img)
    prob = gmm.predict_proba(img)
    pd.DataFrame(prob[:].round(3)).to_csv("gmm_prob_results.csv")
    return np.reshape(labels, original_shape[:2]), list(np.unique(labels, return_counts=True))


def preprocess_input():
    pass

def read_image(path, space_color="RGB", chanels_indexs=None, scale_percent=100):
    assert space_color.upper() in SUPPORTED_SPACE_COLOR , "Wrong space colors ..."
    image = cv2.imread(path)
    """
    while image.shape[0] * image.shape[1] > 45000 :
        scale_percent-=1
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        image = cv2.resize(image, dim)
    """
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
    elif space_color.upper() == "HSI" :
        image_to_return = RGB_TO_HSI(image)
        if chanels_indexs != None :
            image_to_return =image_to_return[:,:,chanels_indexs]
    return image_to_return, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)