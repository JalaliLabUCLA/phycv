import cv2
import numpy as np
from numpy.fft import fft2, fftshift, ifft2

from .utils import cart2pol, denoise, morph, normalize


class PAGE:
    def __init__(self, direction_bins, h=None, w=None):
        """initialize the PAGE CPU version class

        Args:
            direction_bins (int): number of different diretions of edge to be extracted
            h (int, optional): height of the image to be processed. Defaults to None.
            w (int, optional): width of the image to be processed. Defaults to None.
        """
        self.h = h
        self.w = w
        self.direction_bins = direction_bins

    def load_img(self, img_file=None, img_array=None):
        """load the image from an ndarray or from an image file

        Args:
            img_file (str, optional): path to the image. Defaults to None.
            img_array (np.ndarray, optional): image in the form of np.ndarray. Defaults to None.
        """
        if img_array is not None:
            # directly load the image from numpy array
            self.img = img_array
            self.h = img_array.shape[0]
            self.w = img_array.shape[1]
        else:
            # load the image from the image file
            self.img = cv2.imread(img_file)
            # read the image size or resize to the indicated size (height x width)
            if not self.h and not self.w:
                self.h = self.img.shape[0]
                self.w = self.img.shape[1]
            else:
                self.img = cv2.imresize(self.img, [self.h, self.w])
            # convert to grayscale if it is RGB
        if self.img.ndim == 3:
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def init_kernel(self, mu_1, mu_2, sigma_1, sigma_2, S1, S2):
        """initialize the phase kernel of PST

        Args:
            mu_1 (float): Center frequency of a normal distributed passband filter ϕ1
            mu_2 (float):  Center frequency of log-normal  distributed passband filter ϕ2
            sigma_1 (float): Standard deviation of normal distributed passband filter ϕ1
            sigma_2 (float): Standard deviation of log-normal distributed passband filter ϕ2
            S1 (float): Phase strength of ϕ1
            S2 (float): Phase strength of ϕ2
        """
        # set the frequency grid
        u = np.linspace(-0.5, 0.5, self.h)
        v = np.linspace(-0.5, 0.5, self.w)
        [U, V] = np.meshgrid(u, v, indexing="ij")
        [self.THETA, self.RHO] = cart2pol(U, V)

        min_direction = np.pi / 180
        direction_span = np.pi / self.direction_bins
        directions = np.arange(min_direction, np.pi, direction_span)

        # create PAGE kernels channel by channel
        self.page_kernel = np.zeros([self.h, self.w, self.direction_bins])
        for i in range(self.direction_bins):
            tetav = directions[i]
            # Project onto new directionality basis for PAGE filter creation
            Uprime = U * np.cos(tetav) + V * np.sin(tetav)
            Vprime = -U * np.sin(tetav) + V * np.cos(tetav)

            # Create Normal component of PAGE filter
            Phi_1 = np.exp(-0.5 * ((abs(Uprime) - mu_1) / sigma_1) ** 2) / (
                1 * np.sqrt(2 * np.pi) * sigma_1
            )
            Phi_1 = (Phi_1 / np.max(Phi_1[:])) * S1

            # Create Log-Normal component of PAGE filter
            Phi_2 = np.exp(-0.5 * ((np.log(abs(Vprime)) - mu_2) / sigma_2) ** 2) / (
                abs(Vprime) * np.sqrt(2 * np.pi) * sigma_2
            )
            Phi_2 = (Phi_2 / np.max(Phi_2[:])) * S2

            # Add overall directional filter to PAGE filter array
            self.page_kernel[:, :, i] = Phi_1 * Phi_2

    def apply_kernel(self, sigma_LPF, thresh_min, thresh_max, morph_flag):
        """apply the phase kernel onto the image
        Args:
            sigma_LPF (float): std of the low pass filter
            thresh_min (float): minimum thershold, we keep features < thresh_min
            thresh_max (float): maximum thershold, we keep features > thresh_max
            morph_flag (boolean): whether apply morphological operation
        """

        # denoise on the loaded image
        self.img_denoised = denoise(img=self.img, rho=self.RHO, sigma_LPF=sigma_LPF)
        self.page_output = np.zeros([self.h, self.w, self.direction_bins])
        # apply the kernel channel by channel
        for i in range(self.direction_bins):
            self.img_page = ifft2(
                fft2(self.img_denoised)
                * fftshift(np.exp(-1j * self.page_kernel[:, :, i]))
            )
            self.page_feature = normalize(np.angle(self.img_page))
            # apply morphological operation if applicable
            if morph_flag == 0:
                self.page_output[:, :, i] = self.page_feature
            else:
                self.page_output[:, :, i] = morph(
                    img=self.img,
                    feature=self.page_feature,
                    thresh_max=thresh_max,
                    thresh_min=thresh_min,
                )

    def create_page_edge(self):
        """create results which color-coded directional edges"""
        # Create a weighted color image of PAGE output to visualize directionality of edges
        weight_step = 255 * 3 / self.direction_bins
        color_weight = np.arange(0, 255, weight_step)
        self.page_edge = np.zeros([self.h, self.w, 3])
        # step_edge = int(round(self.direction_bins/3))
        step_edge = self.direction_bins // 3
        for i in range(step_edge):
            self.page_edge[:, :, 0] = (
                color_weight[i] * self.page_output[:, :, i] + self.page_edge[:, :, 0]
            )
            self.page_edge[:, :, 1] = (
                color_weight[i] * self.page_output[:, :, i + step_edge]
                + self.page_edge[:, :, 1]
            )
            self.page_edge[:, :, 2] = (
                color_weight[i] * self.page_output[:, :, i + (2 * step_edge)]
                + self.page_edge[:, :, 2]
            )

        self.page_edge = (self.page_edge - np.min(self.page_edge)) / (
            np.max(self.page_edge) - np.min(self.page_edge)
        )

    def run(
        self,
        img_file,
        mu_1,
        mu_2,
        sigma_1,
        sigma_2,
        S1,
        S2,
        sigma_LPF,
        thresh_min,
        thresh_max,
        morph_flag,
    ):
        """wrap all steps of PAGE into a single run method

        Args:
            img_file (str): path to the image.
            mu_1 (float): Center frequency of a normal distributed passband filter ϕ1
            mu_2 (float):  Center frequency of log-normal  distributed passband filter ϕ2
            sigma_1 (float): Standard deviation of normal distributed passband filter ϕ1
            sigma_2 (float): Standard deviation of log-normal distributed passband filter ϕ2
            S1 (float): Phase strength of ϕ1
            S2 (float): Phase strength of ϕ2
            sigma_LPF (float): std of the low pass filter
            thresh_min (float): minimum thershold, we keep features < thresh_min
            thresh_max (float): maximum thershold, we keep features > thresh_max
            morph_flag (boolean): whether apply morphological operation

        Returns:
            np.ndarray: color-coded directional edge
        """
        # wrap load_img, init_kernel, apply_kernel, create_page_edge in one run
        self.load_img(img_file=img_file)
        self.init_kernel(mu_1, mu_2, sigma_1, sigma_2, S1, S2)
        self.apply_kernel(sigma_LPF, thresh_min, thresh_max, morph_flag)
        self.create_page_edge()

        return self.page_edge
