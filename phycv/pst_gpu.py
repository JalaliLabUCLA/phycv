from .utils import cart2pol_torch, denoise_torch, morph_torch
import torch
import torchvision
from torchvision.io import read_image
from torchvision.transforms.functional import rgb_to_grayscale, resize
import numpy as np


class PST_GPU:
    def __init__(self, device, h=None, w=None):
        self.h = h
        self.w = w
        self.device = device
    

    def load_img(self, img_file):
        # load the image from the image file
        # torchvision read_image only supports 'jpg' and 'png'
        if img_file.split('.')[-1] in ['jpg', 'png', 'jpeg']:
            self.img = torchvision.io.read_image(img_file).to(self.device)
            # convert to grayscale if it is RGB
            if self.img.dim() == 3 and self.img.shape[0]!=1:
                self.img = rgb_to_grayscale(self.img)
            # read the image size or resize to the indicated size (height x width)
            if not self.h and not self.w:
                self.img = torch.squeeze(self.img)
                self.h = self.img.shape[0]
                self.w = self.img.shape[1]
            else:
                self.img = torch.squeeze(resize(self.img, [self.h, self.w]))
        else:
            # use mahotas to load other format of image
            self.img = mh.imread(img_file)
            if self.img.ndim == 3:
                self.img = mh.colors.rgb2grey(self.img)
            if not self.h and not self.w:
                self.h = self.img.shape[0]
                self.w = self.img.shape[1]
            else:
                self.img = mh.imresize(self.img, [self.h, self.w])
            
            self.img = torch.from_numpy(self.img).to(device)


    def init_kernel(self, phase_strength, warp_strength):
        # set the frequency grid
        u = torch.linspace(-0.5, 0.5, self.h, device=self.device).float()
        v = torch.linspace(-0.5, 0.5, self.w, device=self.device).float()
        [U, V] = (torch.meshgrid(u, v))
        [self.THETA, self.RHO] = cart2pol_torch(U, V)
        # construct the PST Kernel
        self.pst_kernel=torch.multiply(warp_strength*self.RHO, torch.arctan(warp_strength*self.RHO))-0.5*torch.log(1+torch.pow(warp_strength*self.RHO,2))
        self.pst_kernel=self.pst_kernel/torch.max(self.pst_kernel)*phase_strength


    def apply_kernel(self, sigma_LPF, thresh_min, thresh_max, morph_flag):
        # denoise on the loaded image
        self.img_denoised = denoise_torch(img=self.img, rho=self.RHO, sigma_LPF=sigma_LPF)
        # apply the pst kernel
        self.img_pst = torch.fft.ifft2((torch.fft.fft2(self.img_denoised)) * torch.fft.fftshift(torch.exp(-1j * self.pst_kernel)))
        self.pst_feature = torch.angle(self.img_pst)
        # apply morphological operation if applicable
        if morph_flag == 0:
            self.pst_output = self.pst_feature
        else:
            kernel = torch.tensor([[0.0, 1.0, 0.0],
                               [1.0, 1.0, 1.0],
                               [0.0, 1.0, 0.0]]).to(self.device)
            self.pst_output = morph_torch(img=self.img, feature=self.pst_feature, thresh_max=thresh_max, thresh_min=thresh_min, kernel=kernel, device=self.device)
    

    def run(self, img_file, phase_strength, warp_strength, sigma_LPF, thresh_min, thresh_max, morph_flag):
        # wrap load_img, init_kernel, apply_kernel in one run
        self.load_img(img_file)
        self.init_kernel(phase_strength, warp_strength)
        self.apply_kernel(sigma_LPF, thresh_min, thresh_max, morph_flag)

        return self.pst_output