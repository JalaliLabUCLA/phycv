"""
VEViD test script 1: 
All steps (load_img -> init_kernel -> apply_kernel) are performed in a single run method. 
This is for users who want to get the result on a single image with indicated VEViD parameters. 

"""
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from phycv import VEVID, VEVID_GPU


def main():
    # indicate image file, height and width of the image, and GPU device (if applicable)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_file = "./assets/input_images/street_scene.png"
    original_image = mpimg.imread(img_file)

    output_path = "./output/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # VEViD parameters
    S = 0.2
    T = 0.01
    b = 0.16
    G = 1.4

    # run VEVID CPU version
    vevid_cpu = VEVID()
    vevid_output_cpu = vevid_cpu.run(img_file=img_file, S=S, T=T, b=b, G=G)

    # run VEVID GPU version
    vevid_gpu = VEVID_GPU(device=device)
    vevid_output_gpu = vevid_gpu.run(img_file=img_file, S=S, T=T, b=b, G=G)
    vevid_output_gpu = (
        (np.transpose(vevid_output_gpu.cpu().numpy(), (1, 2, 0))) * 255
    ).astype(np.uint8)

    # visualize the results
    f, axes = plt.subplots(1, 2, figsize=(12, 8))
    axes[0].imshow(original_image)
    axes[0].axis("off")
    axes[0].set_title("original image")
    axes[1].imshow(vevid_output_cpu)
    axes[1].axis("off")
    axes[1].set_title("PhyCV Low-Light Enhancement (CPU version)")
    plt.savefig(os.path.join(output_path, "VEViD_CPU_compare.jpg"), bbox_inches="tight")
    vevid_cpu_result = Image.fromarray(vevid_output_cpu)
    vevid_cpu_result.save(os.path.join(output_path, "VEViD_CPU_output.jpg"))

    f, axes = plt.subplots(1, 2, figsize=(12, 8))
    axes[0].imshow(original_image)
    axes[0].axis("off")
    axes[0].set_title("original image")
    axes[1].imshow(vevid_output_gpu)
    axes[1].axis("off")
    axes[1].set_title("PhyCV Low-Light Enhancement (GPU version)")
    plt.savefig(os.path.join(output_path, "VEViD_GPU_compare.jpg"), bbox_inches="tight")

    vevid_gpu_result = Image.fromarray(vevid_output_gpu)
    vevid_gpu_result.save(os.path.join(output_path, "VEViD_GPU_output.jpg"))


if __name__ == "__main__":
    main()
