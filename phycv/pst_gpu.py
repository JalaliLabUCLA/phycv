import numpy as np
import torch
import torchvision
from torch.fft import fft2, fftshift, ifft2
from torchvision.io import read_image
from torchvision.transforms.functional import resize, rgb_to_grayscale

from .utils import cart2pol_torch, denoise_torch, morph_torch, normalize


class PST_GPU:
    def __init__(self, device, h=None, w=None):
        """initialize the PST GPU version class

        Args:
            device (torch.device)
            h (int, optional): height of the image to be processed. Defaults to None.
            w (int, optional): width of the image to be processed. Defaults to None.
        """
        self.h = h
        self.w = w
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

    def init_kernel(self, S, W):
        """initialize the phase kernel of PST

        Args:
            S (float): phase strength of PST
            W (float): warp of PST
        """
        # set the frequency grid
        u = torch.linspace(-0.5, 0.5, self.h, device=self.device).float()
        v = torch.linspace(-0.5, 0.5, self.w, device=self.device).float()
        [U, V] = torch.meshgrid(u, v, indexing="ij")
        [self.THETA, self.RHO] = cart2pol_torch(U, V)
        # construct the PST Kernel
        self.pst_kernel = W * self.RHO * torch.arctan(W * self.RHO) - 0.5 * torch.log(
            1 + (W * self.RHO) ** 2
        )
        self.pst_kernel = S * self.pst_kernel / torch.max(self.pst_kernel)

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
        # apply the pst kernel
        self.img_pst = ifft2(
            fft2(self.img_denoised) * fftshift(torch.exp(-1j * self.pst_kernel))
        )
        self.pst_feature = normalize(torch.angle(self.img_pst))
        # apply morphological operation if applicable
        if morph_flag == 0:
            self.pst_output = self.pst_feature
        else:
            self.pst_output = morph_torch(
                img=self.img,
                feature=self.pst_feature,
                thresh_max=thresh_max,
                thresh_min=thresh_min,
                device=self.device,
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
            img_file (str): _description_
            S (float): _description_
            W (float): _description_
            sigma_LPF (float): _description_
            thresh_min (float): _description_
            thresh_max (float): _description_
            morph_flag (boolean): _description_

        Returns:
            torch.Tensor: PST output
        """
        # wrap load_img, init_kernel, apply_kernel in one run
        self.load_img(img_file=img_file)
        self.init_kernel(S, W)
        self.apply_kernel(sigma_LPF, thresh_min, thresh_max, morph_flag)

        return self.pst_output
