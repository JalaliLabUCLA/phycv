import numpy as np
import torch
import torchvision
from torch.fft import fft2, fftshift, ifft2
from torchvision.io import read_image
from torchvision.transforms.functional import resize, rgb_to_grayscale

from .utils import cart2pol_torch, denoise_torch, morph_torch, normalize


class PAGE_GPU:
    def __init__(self, direction_bins, device, h=None, w=None):

        """initialize the PAGE GPU version class

        Args:
            direction_bins (int): number of different diretions of edge to be extracted
            device(torch.device)
            h (int, optional): height of the image to be processed. Defaults to None.
            w (int, optional): width of the image to be processed. Defaults to None.
        """
        self.h = h
        self.w = w
        self.direction_bins = direction_bins
        self.device = device

    def load_img(self, img_file=None, img_array=None):
        """load the image from an ndarray or from an image file

        Args:
            img_file (str, optional): path to the image. Defaults to None.
            img_array (torch.Tensor, optional): image in the form of torch.Tensor. Defaults to None.
        """
        if img_array is not None:
            # directly load the image from the array instead of the file
            if img_array.get_device() == self.device:
                self.img = img_array
            else:
                self.img = img_array.to(self.device)
            # convert to grayscale if it is RGB
            if self.img.dim() == 3 and self.img.shape[0] != 1:
                self.img = rgb_to_grayscale(self.img)
            # read the image size or resize to the indicated size (height x width)
            if not self.h and not self.w:
                self.img = torch.squeeze(self.img)
                self.h = self.img.shape[0]
                self.w = self.img.shape[1]
            else:
                self.img = torch.squeeze(resize(self.img, [self.h, self.w]))

        else:
            # load the image from the image file
            # torchvision read_image only supports 'jpg' and 'png'
            if img_file.split(".")[-1] in ["jpg", "png", "jpeg"]:
                self.img = torchvision.io.read_image(img_file).to(self.device)
                # convert to grayscale if it is RGB
                if self.img.dim() == 3 and self.img.shape[0] != 1:
                    self.img = rgb_to_grayscale(self.img)
                # read the image size or resize to the indicated size (height x width)
                if not self.h and not self.w:
                    self.img = torch.squeeze(self.img)
                    self.h = self.img.shape[0]
                    self.w = self.img.shape[1]
                else:
                    self.img = torch.squeeze(resize(self.img, [self.h, self.w]))

            else:
                # use opencv to load other format of image
                self.img = cv2.imread(img_file)
                if self.img.ndim == 3:
                    self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
                if not self.h and not self.w:
                    self.h = self.img.shape[0]
                    self.w = self.img.shape[1]
                else:
                    self.img = cv2.imresize(self.img, [self.h, self.w])

                self.img = torch.from_numpy(self.img).to(self.device)

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
        u = torch.linspace(-0.5, 0.5, self.h, device=self.device).float()
        v = torch.linspace(-0.5, 0.5, self.w, device=self.device).float()
        [U, V] = torch.meshgrid(u, v, indexing="ij")
        [self.THETA, self.RHO] = cart2pol_torch(U, V)

        min_direction = np.pi / 180
        direction_span = np.pi / self.direction_bins
        directions = torch.arange(min_direction, np.pi, direction_span)

        # create PAGE kernels in parallel by broadcasting
        tetavs = torch.unsqueeze(directions, dim=0).to(self.device)
        Us = torch.unsqueeze(U, dim=-1)
        Vs = torch.unsqueeze(V, dim=-1)
        Uprimes = Us * torch.cos(tetavs) + Vs * torch.sin(tetavs)
        Vprimes = -Us * torch.sin(tetavs) + Vs * torch.cos(tetavs)

        Phi_1s = torch.exp(-0.5 * ((torch.abs(Uprimes) - mu_1) / sigma_1) ** 2) / (
            1 * np.sqrt(2 * np.pi) * sigma_1
        )
        Phi_1s = (
            Phi_1s / torch.max(Phi_1s.view(-1, self.direction_bins), dim=0)[0]
        ) * S1

        Phi_2s = torch.exp(
            -0.5 * ((torch.log(torch.abs(Vprimes)) - mu_2) / sigma_2) ** 2
        ) / (abs(Vprimes) * np.sqrt(2 * np.pi) * sigma_2)
        Phi_2s = (
            Phi_2s / torch.max(Phi_2s.view(-1, self.direction_bins), dim=0)[0]
        ) * S2
        self.page_kernel = Phi_1s * Phi_2s

    def apply_kernel(self, sigma_LPF, thresh_min, thresh_max, morph_flag):

        """apply the phase kernel onto the image
        Args:
            sigma_LPF (float): std of the low pass filter
            thresh_min (float): minimum thershold, we keep features < thresh_min
            thresh_max (float): maximum thershold, we keep features > thresh_max
            morph_flag (boolean): whether apply morphological operation
        """
        # denoise on the loaded image
        self.img_denoised = denoise_torch(
            img=self.img, rho=self.RHO, sigma_LPF=sigma_LPF
        )
        # apply the page kernel
        self.img_page = ifft2(
            fft2(self.img_denoised).unsqueeze(-1)
            * fftshift(torch.exp(-1j * self.page_kernel), dim=(0, 1)),
            dim=(0, 1),
        )
        self.page_feature = normalize(torch.angle(self.img_page))
        # apply morphological operation if applicable
        if morph_flag == 0:
            self.page_output = self.page_feature
        else:
            self.page_output = morph_torch(
                img=self.img,
                feature=self.page_feature,
                thresh_min=thresh_min,
                thresh_max=thresh_max,
                device=self.device,
            )

    def create_page_edge(self):
        """create results which color-coded directional edges"""
        # Create a weighted color image of PAGE output to visualize directionality of edges
        weight_step = 255 * 3 / self.direction_bins
        color_weight = torch.arange(0, 255, weight_step).to(self.device)
        self.page_edge = torch.zeros([self.h, self.w, 3]).to(self.device)
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

        self.page_edge = (self.page_edge - torch.min(self.page_edge)) / (
            torch.max(self.page_edge) - torch.min(self.page_edge)
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
            torch.Tensor: color-coded directional edge
        """
        self.load_img(img_file=img_file)
        self.init_kernel(mu_1, mu_2, sigma_1, sigma_2, S1, S2)
        self.apply_kernel(sigma_LPF, thresh_min, thresh_max, morph_flag)
        self.create_page_edge()

        return self.page_edge
