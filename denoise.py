import numpy as np
from scipy import ndimage
from skimage import filters, restoration


nb_classes = 10 

def blur(X, type="median", filter_size=3, sigma=1.0):

    X = np.clip(X, -1, 1)
    
    if type == "median":
        blured_X = np.array(list(map(lambda x: ndimage.median_filter(x, filter_size), 
                                     X)))
    elif type == "gaussian":
        blured_X = np.array(list(map(lambda x: ndimage.gaussian_filter(x, filter_size),
                                     X)))
    elif type == "f_gaussian":
        blured_X = np.array(list(map(lambda x: filters.gaussian_filter(x.reshape((28, 28)), sigma=sigma).reshape(784),
                                     X))) 
    elif type == "tv_chambolle":
        blured_X = np.array(list(map(lambda x: restoration.denoise_tv_chambolle(x.reshape((28, 28)), weight=0.2).reshape(784),
                                     X)))
    elif type == "tv_bregman":
        blured_X = np.array(list(map(lambda x: restoration.denoise_tv_bregman(x.reshape((28, 28)), weight=5.0).reshape(784),
                                     X)))
    elif type == "bilateral":
        blured_X = np.array(list(map(lambda x: restoration.denoise_bilateral(np.abs(x).reshape((28, 28))).reshape(784),
                                     X)))
    elif type == "nl_means":
        blured_X = np.array(list(map(lambda x: restoration.nl_means_denoising(x.reshape((28, 28))).reshape(784),
                                     X)))
        
    elif type == "none":
        blured_X = X 

    else:
        raise ValueError("unsupported filter type", type)

    return blured_X 

