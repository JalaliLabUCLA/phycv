from .utils import cart2pol, denoise, morph
import mahotas as mh
import numpy as np

class PAGE:
    def __init__(self,direction_bins, h=None, w=None):
        self.h = h
        self.w = w
        self.direction_bins = direction_bins
    
    
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
        
    
    def init_kernel(self, mu_1, mu_2, sigma_1, sigma_2, S1, S2):
        # initialize the filter arrays of page
        # mu_1            Center frequency of a normal/Gaussian distributed passband filter ϕ1
        # mu_2            Center frequency of log-normal distributed distributed passband filter ϕ2
        # sigma_1         Standard deviation sigma of normal/Gaussian distributed passband filter ϕ1
        # sigma_2         Standard deviation sigma of log-normal distributed passband filter ϕ2
        # S1              Strength (Amplitude) of ϕ1 filter
        # S2              Strength (Amplitude) of ϕ2 filter
        
        # set the frequency grid
        u = np.linspace(-0.5, 0.5, self.h)
        v = np.linspace(-0.5, 0.5, self.w)
        [U1, V1] = (np.meshgrid(u, v))
        U = U1.T
        V = V1.T
        [self.THETA, self.RHO] = cart2pol(U, V)

        min_direction = np.pi/180
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
            Phi_1 = np.exp(-0.5 * ((abs(Uprime) - mu_1) / sigma_1)** 2) / (1 * np.sqrt(2 * np.pi) * sigma_1)
            Phi_1 = (Phi_1 / np.max(Phi_1[:])) * S1

            # Create Log-Normal component of PAGE filter
            Phi_2 = np.exp(-0.5 * ((np.log(abs(Vprime)) - mu_2) / sigma_2) ** 2) / (abs(Vprime) * np.sqrt(2 * np.pi) * sigma_2)
            Phi_2 = (Phi_2 / np.max(Phi_2[:])) * S2

            # Add overall directional filter to PAGE filter array
            self.page_kernel[:,:,i]= Phi_1 * Phi_2
    

    def apply_kernel(self, sigma_LPF, thresh_min, thresh_max, morph_flag):
        # denoise on the loaded image
        self.img_denoised = denoise(img=self.img, rho=self.RHO, sigma_LPF=sigma_LPF)
        self.page_output = np.zeros([self.h, self.w, self.direction_bins])
        # apply the kernel channel by channel
        for i in range(self.direction_bins):
            self.img_page = np.fft.ifft2((np.fft.fft2(self.img_denoised)) * np.fft.fftshift(np.exp(-1j * self.page_kernel[:,:,i])))
            self.page_feature = np.angle(self.img_page)
            # apply morphological operation if applicable
            if morph_flag == 0:
                self.page_output[:,:,i] = self.page_feature
            else:
                self.page_output[:,:,i] = morph(img=self.img, feature=self.page_feature, thresh_max=thresh_max, thresh_min=thresh_min)


    def create_page_edge(self):
        # Create a weighted color image of PAGE output to visualize directionality of edges
        weight_step = 255*3/self.direction_bins
        color_weight = np.arange(0, 255, weight_step)
        self.page_edge = np.zeros([self.h, self.w, 3])
        step_edge = int(round(self.direction_bins/3))
        for i in range(step_edge):
            self.page_edge[:,:,0] = color_weight[i] * self.page_output[:,:,i] + self.page_edge[:,:,0]
            self.page_edge[:,:,1] = color_weight[i] * self.page_output[:,:,i+step_edge] + self.page_edge[:,:,1]
            self.page_edge[:,:,2] = color_weight[i] * self.page_output[:,:,i+(2*step_edge)] + self.page_edge[:,:,2]

        self.page_edge = (self.page_edge - np.min(self.page_edge)) / (np.max(self.page_edge) - np.min(self.page_edge))


    def run(self, img_file, mu_1, mu_2, sigma_1, sigma_2, S1, S2, sigma_LPF, thresh_min, thresh_max, morph_flag):
        # wrap load_img, init_kernel, apply_kernel, create_page_edge in one run
        self.load_img(img_file)
        self.init_kernel(mu_1, mu_2, sigma_1, sigma_2, S1, S2)
        self.apply_kernel(sigma_LPF, thresh_min, thresh_max, morph_flag)
        self.create_page_edge()
        
        return self.page_edge