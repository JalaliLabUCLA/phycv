import cv2
import numpy as np
import torch
from kornia.color import hsv_to_rgb, rgb_to_hsv
from torch.fft import fft2, fftshift, ifft2
from torchvision.io import read_image
from torchvision.transforms.functional import resize, rgb_to_grayscale

from .utils import cart2pol_torch, normalize


class VEVID_GPU:
    def __init__(self, device, h=None, w=None):
        """initialize the VEVID GPU version class

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
                self.img_rgb = img_array
            else:
                self.img_rgb = img_array.to(self.device)
            if not self.h and not self.w:
                self.h = self.img_rgb.shape[-2]
                self.w = self.img_rgb.shape[-1]
            # convert from RGB to HSV
            self.img_hsv = rgb_to_hsv(self.img_rgb)

        else:
            # load the image from the image file
            # torchvision read_image currently only supports 'jpg' and 'png'
            # use opencv to read other image formats
            if img_file.split(".")[-1] in ["jpg", "png", "jpeg"]:
                self.img_rgb = read_image(img_file).to(self.device)
            else:
                self.img_bgr = cv2.imread(img_file)
                self.img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
                self.img_rgb = torch.from_numpy(
                    np.transpose(self.img_rgb, (2, 0, 1))
                ).to(self.device)
            if not self.h and not self.w:
                self.h = self.img_rgb.shape[-2]
                self.w = self.img_rgb.shape[-1]
            else:
                self.img_rgb = resize(self.img_rgb, [self.h, self.w])
            # convert from RGB to HSV
            # rgb_to_hsv in kornia requires the input RGB image to be in the range of 0-1
            self.img_hsv = rgb_to_hsv((self.img_rgb.float()) / 255.0)

    def init_kernel(self, S, T):
        """initialize the phase kernel of VEViD

        Args:
            S (float): phase strength
            T (float): variance of the spectral phase function
        """
        # create the frequency grid
        u = torch.linspace(-0.5, 0.5, self.h, device=self.device).float()
        v = torch.linspace(-0.5, 0.5, self.w, device=self.device).float()
        [U, V] = torch.meshgrid(u, v, indexing="ij")
        # construct the kernel
        [self.THETA, self.RHO] = cart2pol_torch(U, V)
        self.vevid_kernel = torch.exp(-self.RHO**2 / T)
        self.vevid_kernel = (self.vevid_kernel / torch.max(abs(self.vevid_kernel))) * S

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
        vevid_input = self.img_hsv[channel_idx, :, :]
        if lite:
            vevid_phase = torch.atan2(-G * (vevid_input + b), vevid_input)
        else:
            vevid_input_f = fft2(vevid_input + b)
            img_vevid = ifft2(
                vevid_input_f * fftshift(torch.exp(-1j * self.vevid_kernel))
            )
            vevid_phase = torch.atan2(G * torch.imag(img_vevid), vevid_input)
        
        vevid_phase_norm = normalize(vevid_phase)
        self.img_hsv[channel_idx, :, :] = vevid_phase_norm
        self.vevid_output = hsv_to_rgb(self.img_hsv)

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
            torch.Tensor: enhanced image
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
            torch.Tensor: enhanced image
        """
        self.load_img(img_file=img_file)
        self.apply_kernel(b, G, color, lite=True)

        return self.vevid_output
