import numpy as np
import cv2

from .utils import normalize

class VLight:
    def __init__(self, row=None, col=None):
        """Initialize the VLight class

        Args:
            row (int, optional): height of the image to be processed. Defaults to None.
            col (int, optional): width of the image to be processed. Defaults to None.
        """
        self.row = row
        self.col = col
        self.vlight_lut = None
        self.v = None

    def generate_vlight_lut(self, v):
        """
        Generates and updates the lookup table for VLight based on the parameter v.
        The LUT is kept in floating-point precision for accurate calculations.
        
        Args:
        - v (float): VLight parameter.
        """
        if self.v == v and self.vlight_lut is not None:
            return
        
        b = 1 / (5 * v + 0.05)
        G = 1 - v**2
        
        # Generate LUT using vectorized operations for efficiency
        pixel_values = np.linspace(0, 1, 256, dtype=np.float32)
        lut_values = np.arctan2(-G * (pixel_values + b), pixel_values)
        lut_values = np.interp(lut_values, (lut_values.min(), lut_values.max()), (0, 255)).astype(np.uint8)
        
        self.vlight_lut = lut_values
        self.v = v
    
    def load_img(self, img_file=None, img_array=None):
        """load the image from an ndarray or from an image file

        Args:
            img_file (str, optional): path to the image. Defaults to None.
            img_array (np.ndarray, optional): image in the form of np.ndarray. Defaults to None.
        """
        if img_array is not None:
            # directly load the image from numpy array
            self.img_bgr = img_array
            self.row = img_array.shape[0]
            self.col = img_array.shape[1]
        else:
            # load the image from the image file
            self.img_bgr = cv2.imread(img_file)
            if not self.row and not self.col:
                self.row = self.img_bgr.shape[0]
                self.col = self.img_bgr.shape[1]
            else:
                self.img_bgr = cv2.resize(self.img_bgr, [self.col, self.row])

        # Check and convert float images to 8-bit if in range [0, 1]
        if self.img_bgr.dtype == np.float32 or self.img_bgr.dtype == np.float64:
            self.img_bgr = (self.img_bgr * 255).astype(np.uint8)

    def apply_kernel(self, v, color=False, lut=True):
        """
        Override the apply_kernel method to use the LUT-based approach for faster processing.
        
        Args:
        - img_array (np.ndarray): Input image array in BGR format.
        - v (float): VLight parameter for LUT generation.
        - color (bool, optional): Whether to apply the transformation to the S (color) channel instead of V. Defaults to False.
        - lut (bool, optional): Whether to apply the lut acceleration or not, defaults to True
        Returns:
        - np.ndarray: Enhanced image in BGR format.
        """
        if color:
            channel = 1
        else:
            channel = 2
        
        if lut is True:
            # Convert the image from BGR to HSV
            self.img_hsv = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2HSV)
            
            # Generate the LUT based on the vlight parameter
            self.generate_vlight_lut(v)
            
            # Apply LUT to the selected channel (V or S)
            self.img_hsv[:, :, channel] = cv2.LUT(self.img_hsv[:, :, channel], self.vlight_lut)
            
            # Convert the image back to BGR format
            self.vlight_output = cv2.cvtColor(self.img_hsv, cv2.COLOR_HSV2BGR)
        else:
            self.img_hsv = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2HSV) / 255.0
            vlight_input = self.img_hsv[:, :, channel]
            b = 1 / (5 * v + 0.05)
            G = 1 - v**2
            vlight_phase = np.arctan2(-G * (vlight_input + b), vlight_input)
            vlight_phase_norm = normalize(vlight_phase)
            self.img_hsv[:, :, channel] = vlight_phase_norm
            self.img_hsv = (self.img_hsv * 255).astype(np.uint8)
            self.vlight_output = cv2.cvtColor(self.img_hsv, cv2.COLOR_HSV2BGR)

    def run(self, img_file, v, color=False, lut=True):
        """run the VLight algorithm

        Args:
            - img_file (str): path to the image
            - v (float): VLight Parameteter
            - color (bool, optional): whether to run color enhancement. Defaults to False.
            - lut (bool, optional): Whether to apply the lut acceleration or not, defaults to True
        Returns:
            np.ndarray: enhanced image
        """
        self.load_img(img_file=img_file)
        self.apply_kernel(v=v, color=color, lut=lut)

        return self.vlight_output

    def run_img_array(self, img_array, v, color=False, lut=True):
        """run the VLight LUT accelerated algorithm

        Args:
            - img_array (np.ndarray): Input image array in BGR format.
            - v (float): VLight parameter.
            - color (bool, optional): whether to run color enhancement. Defaults to False.
            - lut (bool, optional): Whether to apply the lut acceleration or not, defaults to True
        Returns:
            np.ndarray: enhanced image
        """
        self.load_img(img_array=img_array)
        self.apply_kernel(v=v, color=color, lut=lut)
        return self.vlight_output