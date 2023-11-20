import cv2
import numpy as np
from numpy.fft import fft2, fftshift, ifft2

from .utils import cart2pol, normalize


class VEVID:
    def __init__(self, h=None, w=None):
        """initialize the VEVID CPU version class

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
            # directly load the image from numpy array
            self.img_bgr = img_array
            self.h = img_array.shape[0]
            self.w = img_array.shape[1]
        else:
            # load the image from the image file
            self.img_bgr = cv2.imread(img_file)
            if not self.h and not self.w:
                self.h = self.img_bgr.shape[0]
                self.w = self.img_bgr.shape[1]
            else:
                self.img_bgr = cv2.resize(self.img_bgr, [self.w, self.h])

        self.img_hsv = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2HSV) / 255.0

    def init_kernel(self, S, T):
        """initialize the phase kernel of VEViD

        Args:
            S (float): phase strength
            T (float): variance of the spectral phase function
        """
        # create the frequency grid
        u = np.linspace(-0.5, 0.5, self.h)
        v = np.linspace(-0.5, 0.5, self.w)
        [U, V] = np.meshgrid(u, v, indexing="ij")
        # construct the kernel
        [self.THETA, self.RHO] = cart2pol(U, V)
        self.vevid_kernel = np.exp(-self.RHO**2 / T)
        self.vevid_kernel = (self.vevid_kernel / np.max(abs(self.vevid_kernel))) * S

    def apply_kernel(self, b, G, color=False, lite=False):
        """apply the phase kernel onto the image

        Args:
            b (float): regularization term
            G (float): phase activation gain
            color (bool, optional): whether to run color enhancement. Defaults to False.
            lite (bool, optional): whether to run VEViD lite. Defaults to False.
        """
        if color:
            channel_idx = 1
        else:
            channel_idx = 2
        vevid_input = self.img_hsv[:, :, channel_idx]
        if lite:
            vevid_phase = np.arctan2(-G * (vevid_input + b), vevid_input)
        else:
            vevid_input_f = fft2(vevid_input + b)
            img_vevid = ifft2(vevid_input_f * fftshift(np.exp(-1j * self.vevid_kernel)))
            vevid_phase = np.arctan2(G * np.imag(img_vevid), vevid_input)
        vevid_phase_norm = normalize(vevid_phase)
        self.img_hsv[:, :, channel_idx] = vevid_phase_norm
        self.img_hsv = (self.img_hsv * 255).astype(np.uint8)
        self.vevid_output = cv2.cvtColor(self.img_hsv, cv2.COLOR_HSV2RGB)

    def run(self, img_file, S, T, b, G, color=False):
        """run the full VEViD algorithm

        Args:
            img_file (str): path to the image
            S (float): phase strength
            T (float): variance of the spectral phase function
            b (float): regularization term
            G (float): phase activation gain
            color (bool, optional): whether to run color enhancement. Defaults to False.

        Returns:
            np.ndarray: enhanced image
        """
        self.load_img(img_file=img_file)
        self.init_kernel(S, T)
        self.apply_kernel(b, G, color, lite=False)

        return self.vevid_output

    def run_lite(self, img_file, b, G, color=False):
        """run the VEViD lite algorithm

        Args:
            img_file (str): path to the image
            b (float): regularization term
            G (float): phase activation gain
            color (bool, optional): whether to run color enhancement. Defaults to False.

        Returns:
            np.ndarray: enhanced image
        """
        self.load_img(img_file=img_file)
        self.apply_kernel(b, G, color, lite=True)

        return self.vevid_output
