## Vision Enhancement via Virtual diffraction and coherent Detection (VEViD)

### **Code Architect**

```python
class VEVID:
    def __init__(self, h=None, w=None):

    def load_img(self, img_file=None, img_array=None):

    def init_kernel(self, S, T):
    
    def apply_kernel(self, b, G, color=False, lite=False):    

    def run(self, img_file, S, T, b, G, color=False): 
    
    def run_lite(self, img_file, b, G, color=False):
```

* To instantiate a `VEVID` class, three parameters `h` (height),  `w` (width). `h` and `w` are set to `None` by default. If you leave them as `None` by default, their value will be determined when calling the following `load_img` method. 

* The `load_img` method can load the image from either an image file indicated by `img_file` or an image array indicated by `img_array`. Then it converts the image to from RGB to HSV. If `h` and `w` are not indicated in the `__init__` method, they will be determined by the shape of the image. Otherwise, the image will be reshaped to the indicated size.

* The `init_kernel` method initializes the VEViD kernel according to the parameters phase scale `S` and phase variance `T`. The explanation of the parameters can be found in [11].

* The `apply_kernel` method applies the initialized kernel to V channel of the HSV image with a regularization constant `b` and a phase activation gain `G`. `color` is a boolean flag to indicate whether we run VEViD for low-light enhancement or color enhancement. `lite` is a boolean flag to indicate whether the approximated version of VEViD is used. The explanation of the parameter, low-light enhancement vs. color enhancement and the approximation can be found in [11].

* The `run` method runs the full VEViD algorithm which wraps `load_img`,  `init_kernel`,  `apply_kernel` together.

* The `run_lite` method runs the accelerated VEViD algorithm called VEViD-lite which wraps `load_img` and  `apply_kernel` together.


### **GPU Accelerationn**

The GPU version of VEViD significantly accelerates the PST algorithm by using GPU(s). As defined in `phycv/vevid_gpu.py` , the architect of the `VEVID_GPU` class is similar to the original `VEVID` class with same attributes and methods. The main differences are:

1. You have to indicate the `device` to operate on when instantiating the class.
2. The image is loaded as `torch.Tensor` instead of `numpy.ndarray`.
3. The returned result also locates on GPU and is in the form of `torch.Tensor`.

### **Examples**

*Example 1 - Single Image* ( `scripts/run_vevid.py` )

In this example, we run VEViD on a single image and all steps are performed in a single `run` method.

*Example 2 - Video* ( `scripts/run_vevid_video.py` and `scripts/run_vevid_lite_video.py` )

In these examples, we run VEViD and VEViD-lite on a video.