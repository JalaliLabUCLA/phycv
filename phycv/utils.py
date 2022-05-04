import numpy as np
import torch
import torch.fft
import mahotas as mh
import kornia

'''
in utils.py, we define common operations in both PST and PAGE

cart2pol: transform cartesian coordinates to polar coordinates
denoise: apply a low pass filter to the original image
morph: morphological operation from analog features to digital features

'''

def cart2pol(x, y):
    '''
    transform cartesian coordinates to polar coordinates
    Arguments: cartesian coordinates: x, y
    Return: polar coordinates: rho, theta

    '''
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return (theta, rho)

def cart2pol_torch(x,y):
    '''
    cart2pol GPU version

    '''
    theta = torch.atan2(y, x)
    rho = torch.hypot(x, y)
    return (theta, rho)


def denoise(img, rho, sigma_LPF):
    '''
    apply a low pass filter to the original image
    Arguments: original image, polar coordinates rho, sigma of the low-pass-filter
    Return: filtered image

    '''
    img_orig_f = np.fft.fft2(img)
    expo = np.fft.fftshift(np.exp(-0.5*np.power((np.divide(rho, np.sqrt((sigma_LPF**2)/np.log(2)))),2)))
    img_filtered = np.real(np.fft.ifft2((np.multiply(img_orig_f,expo))))

    return img_filtered


def denoise_torch(img, rho, sigma_LPF):
    '''
    denoise GPU version

    '''
    img_orig_f = torch.fft.fft2(img)
    expo = torch.fft.fftshift(torch.exp(-0.5*torch.pow((torch.divide(rho, np.sqrt((sigma_LPF ** 2) / np.log(2)))), 2)))
    img_filtered = torch.real(torch.fft.ifft2((torch.mul(img_orig_f, expo))))

    return img_filtered


def morph(img, feature, thresh_min, thresh_max):
    '''
    apply morphological operation to transform analog features to digial features
    Argument: original image, analog features output by PST/PAGE, threshhold min and threshold max
    Return: digital feature after morphological operations

    '''
    digital_feature = np.zeros(feature.shape)
    
    digital_feature[feature>thresh_max] = 1 # Bi-threshold decision
    digital_feature[feature<thresh_min] = 1 # as the output phase has both positive and negative values
    digital_feature[img<(np.amax(img)/20)]=0 # Removing edges in the very dark areas of the image (noise)

    digital_feature = mh.thin(digital_feature, 1)
    digital_feature = mh.bwperim(digital_feature, 4)
    digital_feature = mh.thin(digital_feature, 1)
    digital_feature = mh.erode(digital_feature, np.ones((1, 1)))

    return digital_feature.astype(np.float32)


def morph_torch(img, feature, thresh_min, thresh_max, kernel, device):
    '''
    morph GPU version (torch and kornia)

    '''
    digital_feature = torch.zeros(feature.shape).to(device)
    digital_feature[feature>thresh_max] = 1 # Bi-threshold decision
    digital_feature[feature<thresh_min] = 1 # as the output phase has both positive and negative values
    digital_feature[img<(torch.max(img)/20)]=0 # Removing edges in the very dark areas of the image (noise)

    # for PST
    if len(feature.shape)==2:
        digital_feature = kornia.utils.image._to_bchw(digital_feature)
        digital_feature = kornia.morphology.closing(digital_feature, kernel)

    # for PAGE
    elif len(feature.shape)==3:
        digital_feature = torch.permute(digital_feature,(2,0,1))
        digital_feature = kornia.utils.image._to_bchw(torch.unsqueeze(digital_feature, 1))         
        digital_feature = torch.squeeze(kornia.morphology.closing(digital_feature, kernel))
        digital_feature = torch.permute(digital_feature, (1,2,0))

    return torch.squeeze(digital_feature)
