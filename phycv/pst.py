from .utils import cart2pol, denoise, morph
import mahotas as mh
import numpy as np

class PST:
    def __init__(self, h=None, w=None):
        self.h = h
        self.w = w
    

    def load_img(self, img_file):
        # load the image from the image file
        self.img = mh.imread(img_file)
        # convert to grayscale if it is RGB
        if self.img.ndim == 3:
            self.img = mh.colors.rgb2grey(self.img)
        # read the image size or resize to the indicated size (height x width)
        if not self.h and not self.w:
            self.h = self.img.shape[0]
            self.w = self.img.shape[1]
        else:
            self.img = mh.imresize(self.img, [self.h, self.w])
        

    def init_kernel(self, phase_strength, warp_strength):
        # set the frequency grid
        u = np.linspace(-0.5, 0.5, self.h)
        v = np.linspace(-0.5, 0.5, self.w)
        [U1, V1] = (np.meshgrid(u, v))
        U = U1.T
        V = V1.T
        [self.THETA, self.RHO] = cart2pol(U, V)
        # construct the PST Kernel
        self.pst_kernel = np.multiply(warp_strength*self.RHO, np.arctan(warp_strength*self.RHO))-0.5*np.log(1+np.power(warp_strength*self.RHO,2))
        self.pst_kernel = self.pst_kernel/np.max(self.pst_kernel)*phase_strength


    def apply_kernel(self, sigma_LPF, thresh_min, thresh_max, morph_flag):
        # denoise on the loaded image
        self.img_denoised = denoise(img=self.img, rho=self.RHO, sigma_LPF=sigma_LPF)
        # apply the pst kernel
        self.img_pst = np.fft.ifft2((np.fft.fft2(self.img_denoised)) * np.fft.fftshift(np.exp(-1j * self.pst_kernel)))
        self.pst_feature = np.angle(self.img_pst)
        # apply morphological operation if applicable
        if morph_flag == 0:
            self.pst_output = self.pst_feature
        else:
            self.pst_output = morph(img=self.img, feature=self.pst_feature, thresh_max=thresh_max, thresh_min=thresh_min)


    def run(self, img_file, phase_strength, warp_strength, sigma_LPF, thresh_min, thresh_max, morph_flag):
        # wrap load_img, init_kernel, apply_kernel in one run
        self.load_img(img_file)
        self.init_kernel(phase_strength, warp_strength)
        self.apply_kernel(sigma_LPF, thresh_min, thresh_max, morph_flag)

        return self.pst_output
