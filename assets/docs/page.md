## Phase-Stretch Adaptive Gradient-field Extractor (PAGE)

### **Code Architect**

```python
class PAGE:
    def __init__(self, direction_bins, h=None, w=None):

    def load_img(self, img_file=None, img_array=None):

    def init_kernel(self, mu_1, mu_2, sigma_1, sigma_2, S1, S2):
    
    def apply_kernel(self, sigma_LPF, thresh_min, thresh_max, morph_flag):    

    def create_page_edge(self):

    def run(self, img_file, mu_1, mu_2, sigma_1, sigma_2, S1, S2,\
            sigma_LPF, thresh_min, thresh_max, morph_flag):
```

* To instantiate a `PAGE` class, three parameters `h` (height),  `w` (width), and `direction_bins` are needed. `h` and `w` are set to `None` by default. If you leave them as `None` by default, their value will be determined when calling the following `load_img` method. 

* The `load_img` method can load the image from either an image file indicated by `img_file` or an image array indicated by `img_array`. Then it converts the image to greyscale if it is in RGB format, if `h` and `w` are not indicated in the `__init__` method, they will be determined by the shape of the image. Otherwise, the image will be reshaped to the indicated size.

* The `init_kernel` method initializes the PAGE kernel according to the parameters `mu_1`,  `mu_2`,  `sigma_1`,  `sigma_2`,  `S1`,  `S2`. In the CPU version, kernels for different frequency bins are initialized in serial. As for the derivation of the kernel and the physical meaning of parameters, please refer to references [9], [10].

* The `apply_kernel` method first denoises the loaded image with a low-pass filter characterized by `sigma_LPF`, then it applies the initialized kernel to the denoised image, finally it applies morphological operation if `morph_flag==1`. The thresholds in the morphological operation are indicated by `thresh_min` and `thresh_max`.

* The `create_page_edge` method creates a weighted color image of PAGE output to visualize directionality of edges.

* The `run` method wraps `load_img`,  `init_kernel`,  `apply_kernel`,  `create_page_edge` together.

### **GPU Acceleration**

The GPU version of PST significantly accelerates the PST algorithm by using GPU(s). As defined in `phycv/page_gpu.py` , the architect of the `PAGE_GPU` class is similar to the original `PAGE` class with same attributes and methods. The main differences are:

1. You have to indicate the `device` (in a PyTorch fashion) when instantiating the class.
2. Image IO is done by `torchvision` instead of `opencv`, matrix operation is done by `torch` instead of `numpy`, morphological operation is done by `kornia` instead of `mahotas`
3. The `init_kernel` method initializes kernels for different direction bins in parallel by using broadcasting. 
4. The `apply_kernel` method applies kernels for different direction bins to the image in parallel by using broadcasting. 
5. The returned result locates on GPU and is in the form of `torch.Tensor`.

### **Examples**

*Example 1 - Single Image* ( `scripts/run_page.py` )

In this example, we run PAGE on a single image and all steps are performed in a single `run` method.

*Example 2 - Video* ( `scripts/run_page_video.py` )

In this example, we run PAGE on a video, when using fixed parameters for all frames, the `init_kernel` can be called only once to save computation time. Note that running this script takes some time because saving processed results and putting them together to generate a video is time-consuming.