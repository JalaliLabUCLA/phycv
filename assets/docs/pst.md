## Phase-Stretch Transform (PST)

### **Code Architect**

```python
class PST:
    def __init__(self, h=None, w=None):

    def load_img(self, img_file=None, img_array=None):

    def init_kernel(self, S, W):
    
    def apply_kernel(self, sigma_LPF, thresh_min, thresh_max, morph_flag):
    
    def run(self, img_file, phase_strength, warp_strength, \
            sigma_LPF, thresh_min, thresh_max, morph_flag):

```

* The `__init__` method has two parameters `h` (height) and `w` (width), which indicates the size of image the algorithm will operate on. They are set to `None` by default. If you leave them as `None` by default, their value will be determined when calling the following `load_img` method. 

* The `load_img` method can load the image from either an image file indicated by `img_file` or an image array indicated by `img_array`. Then it converts the image to greyscale if it is in RGB format, if `h` and `w` are not indicated in the `__init__` method, they will be determined by the shape of the image. Otherwise, the image will be reshaped to the indicated size.

* The `init_kernel` method initializes the PST kernel according to phase strength `S` and warp strength `W`. Please refer to [3], [4] for the explanation of these parameters.

* The `apply_kernel` method first denoises the loaded image with a low-pass filter characterized by `sigma_LPF`, then it applies the initialized kernel to the denoised image, finally it applies morphological operation if `morph_flag==1`. The thresholds in the morphological operation are indicated by `thresh_min` and `thresh_max`.

* The `run` method wraps `load_img`,  `init_kernel`,  `apply_kernel` together.

### **GPU Acceleration**

The GPU version of PST significantly accelerates the PST algorithm by using GPU(s). As defined in `phycv/pst_gpu.py` , the architect of the `PST_GPU` class is similar to the original `PST` class with same attributes and methods. The main differences are:

1. You have to indicate the `device` (in a PyTorch fashion) when instantiating the class.
2. Image IO is done by `torchvision` instead of `opencv`, matrix operation is done by `torch` instead of `numpy`.
3. The returned result locates on GPU and is in the form of `torch.Tensor`.

### **Examples**

*Example 1 - Single Image* ( `scripts/run_pst.py` )

In this example, we run PST on a single image and all steps are performed in a single `run` method.

*Example 2 - Video* ( `scripts/run_pst_video.py` )

In this example, we run PST on a video, when using fixed parameters for all frames, the `init_kernel` can be called only once to save computation time. Note that running this script takes some time because saving processed results and putting them together to generate a video is time-consuming.