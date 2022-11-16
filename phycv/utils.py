import cv2
import kornia
import mahotas as mh
import numpy as np
import torch
import torch.fft


def cart2pol(x, y):
    """convert cartesian coordiates to polar coordinates

    Args:
        x (np.ndarray): cartesian coordinates in x direction
        y (np.ndarray): cartesian coordinates in y direction

    Returns:
        tuple: polar coordinates theta and rho
    """
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return (theta, rho)


def cart2pol_torch(x, y):
    """convert cartesian coordiates to polar coordinates with PyTorch

    Args:
        x (torch.Tensor): cartesian coordinates in x direction
        y (torch.Tensor): cartesian coordinates in x direction

    Returns:
        tuple: polar coordinates theta and rho
    """
    theta = torch.atan2(y, x)
    rho = torch.hypot(x, y)
    return (theta, rho)


def denoise(img, rho, sigma_LPF):
    """apply a low pass filter to denoise the image

    Args:
        img (np.ndarray): original image
        rho (np.ndarray): polar coordinates
        sigma_LPF (float): variance of the low pass filter

    Returns:
        np.ndarray: denoised image
    """
    img_orig_f = np.fft.fft2(img)
    expo = np.fft.fftshift(
        np.exp(
            -0.5 * np.power((np.divide(rho, np.sqrt((sigma_LPF**2) / np.log(2)))), 2)
        )
    )
    img_filtered = np.real(np.fft.ifft2((np.multiply(img_orig_f, expo))))

    return img_filtered


def denoise_torch(img, rho, sigma_LPF):
    """apply a low pass filter to denoise the image with PyTorch

    Args:
        img (torch.Tensor): original image
        rho (torch.Tensor): polar coordinates
        sigma_LPF (float): std of the low pass filter

    Returns:
        torch.Tensor: denoised image
    """
    img_orig_f = torch.fft.fft2(img)
    expo = torch.fft.fftshift(
        torch.exp(
            -0.5
            * torch.pow((torch.divide(rho, np.sqrt((sigma_LPF**2) / np.log(2)))), 2)
        )
    )
    img_filtered = torch.real(torch.fft.ifft2((torch.mul(img_orig_f, expo))))

    return img_filtered


def morph(img, feature, thresh_min, thresh_max):
    """apply morphological operation to transform analog features to digial features

    Args:
        img (np.ndarray): original image
        feature (np.ndarray): analog feature
        thresh_min (float): minimum thershold, we keep features < thresh_min
        thresh_max (float): maximum thershold, we keep features > thresh_max

    Returns:
        np.ndarray: digital features (binary edge)
    """

    digital_feature = np.zeros(feature.shape)

    digital_feature[feature > thresh_max] = 1
    digital_feature[feature < thresh_min] = 1
    digital_feature[img < (np.amax(img) / 20)] = 0

    digital_feature = mh.thin(digital_feature, 1)
    digital_feature = mh.bwperim(digital_feature, 4)
    digital_feature = mh.thin(digital_feature, 1)
    digital_feature = mh.erode(digital_feature, np.ones((1, 1)))

    return digital_feature.astype(np.float32)


def morph_torch(img, feature, thresh_min, thresh_max, kernel, device):
    """apply morphological operation to transform analog features to digial features in PyTorch

    Args:
        img (torch.Tensor): original image
        feature (torch.Tensor): analog feature
        thresh_min (torch.Tensor): minimum thershold, we keep features < thresh_min
        thresh_max (torch.Tensor): maximum thershold, we keep features > thresh_max
        device (torch.device)

    Returns:
        torch.Tensor: digital features (binary edge)
    """
    digital_feature = torch.zeros(feature.shape).to(device)
    digital_feature[feature > thresh_max] = 1
    digital_feature[feature < thresh_min] = 1
    digital_feature[img < (torch.max(img) / 20)] = 0

    # for PST
    if len(feature.shape) == 2:
        digital_feature = kornia.utils.image._to_bchw(digital_feature)
        digital_feature = kornia.morphology.closing(digital_feature, kernel)

    # for PAGE
    elif len(feature.shape) == 3:
        digital_feature = torch.permute(digital_feature, (2, 0, 1))
        digital_feature = kornia.utils.image._to_bchw(
            torch.unsqueeze(digital_feature, 1)
        )
        digital_feature = torch.squeeze(
            kornia.morphology.closing(digital_feature, kernel)
        )
        digital_feature = torch.permute(digital_feature, (1, 2, 0))

    return torch.squeeze(digital_feature)
