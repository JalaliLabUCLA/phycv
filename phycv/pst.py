import cv2
import numpy as np
from numpy.fft import fft2, fftshift, ifft2

from .utils import normalize, cart2pol, denoise, morph


class PST:
    def __init__(self, h=None, w=None):
        """initialize the PST CPU version class

        Args:
            h (int, optional): height of the image to be processed. Defaults to None.
            w (int, optional): width of the image to be processed. Defaults to None.
        """
        self.h = h
        self.w = w

    def load_img(self, img_file=None, img_array=None):
        """load the image from an ndarray or from an image file

        Args:
            img_file (str, optional): path to the image. Defaults to None.
            img_array (np.ndarray, optional): image in the form of np.ndarray. Defaults to None.
        """
        if img_array is not None:
            self.img = img_array
        else:
            self.img = cv2.imread(img_file)
            if not self.h and not self.w:
                self.h = self.img.shape[0]
                self.w = self.img.shape[1]
            else:
                self.img = cv2.imresize(self.img, [self.h, self.w])
        # convert to grayscale if it is RGB
        if self.img.ndim == 3:
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def init_kernel(self, S, W):
        """initialize the phase kernel of PST

        Args:
            S (float): phase strength of PST
            W (float): warp strength of PST
        """
        # set the frequency grid
        u = np.linspace(-0.5, 0.5, self.h)
        v = np.linspace(-0.5, 0.5, self.w)
        [U, V] = np.meshgrid(u, v, indexing="ij")
        [self.THETA, self.RHO] = cart2pol(U, V)
        # construct the PST Kernel
        self.pst_kernel = W * self.RHO * np.arctan(W * self.RHO) - 0.5 * np.log(
            1 + (W * self.RHO) ** 2
        )
        self.pst_kernel = S * self.pst_kernel / np.max(self.pst_kernel)

    def apply_kernel(self, sigma_LPF, thresh_min, thresh_max, morph_flag):
        """apply the phase kernel onto the image

        Args:
            sigma_LPF (float): std of the low pass filter
            thresh_min (float): minimum thershold, we keep features < thresh_min
            thresh_max (float): maximum thershold, we keep features > thresh_max
            morph_flag (boolean): whether apply morphological operation
        """
        self.img_denoised = denoise(img=self.img, rho=self.RHO, sigma_LPF=sigma_LPF)
        self.img_pst = ifft2(
            fft2(self.img_denoised) * fftshift(np.exp(-1j * self.pst_kernel))
        )
        self.pst_feature = normalize(np.angle(self.img_pst))
        if morph_flag == 0:
            self.pst_output = self.pst_feature
        else:
            self.pst_output = morph(
                img=self.img,
                feature=self.pst_feature,
                thresh_max=thresh_max,
                thresh_min=thresh_min,
            )

    def run(
        self,
        img_file,
        S,
        W,
        sigma_LPF,
        thresh_min,
        thresh_max,
        morph_flag,
    ):
        """wrap all steps of PST into a single run method

        Args:
            img_file (str): path to the image.
            S (float): phase strength of PST
            W (float): warp strength of PST
            sigma_LPF (float): std of the low pass filter
            thresh_min (float): minimum thershold, we keep features < thresh_min
            thresh_max (float): maximum thershold, we keep features > thresh_max
            morph_flag (boolean): whether apply morphological operation

        Returns:
            np.ndarray: PST output
        """
        self.load_img(img_file=img_file)
        self.init_kernel(S, W)
        self.apply_kernel(sigma_LPF, thresh_min, thresh_max, morph_flag)

        return self.pst_output
