"""
run PAGE on a single image: 
All steps (load_img -> init_kernel -> apply_kernel -> create_edge) are performed in a single run method. 

"""
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from phycv import PAGE, PAGE_GPU


def main():
    # indicate image file, height and width of the image, and GPU device (if applicable)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_file = "./assets/input_images/wind_rose.png"
    original_image = mpimg.imread(img_file)

    output_path = "./output/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # PAGE parameters
    mu_1 = 0
    mu_2 = 0.35
    sigma_1 = 0.05
    sigma_2 = 0.7
    S1 = 0.8
    S2 = 0.8
    sigma_LPF = 0.1
    thresh_min = 0.0
    thresh_max = 0.9
    morph_flag = 1

    # run PAGE CPU version
    page_cpu = PAGE(direction_bins=10)
    page_edge_cpu = page_cpu.run(
        img_file,
        mu_1,
        mu_2,
        sigma_1,
        sigma_2,
        S1,
        S2,
        sigma_LPF,
        thresh_min,
        thresh_max,
        morph_flag,
    )

    # run PAGE GPU version
    page_gpu = PAGE_GPU(direction_bins=10, device=device)
    page_edge_gpu = page_gpu.run(
        img_file,
        mu_1,
        mu_2,
        sigma_1,
        sigma_2,
        S1,
        S2,
        sigma_LPF,
        thresh_min,
        thresh_max,
        morph_flag,
    )
    page_edge_gpu = page_edge_gpu.cpu().numpy()

    # visualize the results
    f, axes = plt.subplots(1, 2, figsize=(12, 8))
    axes[0].imshow(original_image)
    axes[0].axis("off")
    axes[0].set_title("Original Image")
    axes[1].imshow(page_edge_cpu)
    axes[1].axis("off")
    axes[1].set_title("PhyCV Directional Edge Detection")
    plt.savefig(os.path.join(output_path, "PAGE_CPU_compare.jpg"), bbox_inches="tight")
    page_cpu_result = Image.fromarray((page_edge_cpu * 255).astype(np.uint8))
    page_cpu_result.save(os.path.join(output_path, "PAGE_CPU_output.jpg"))

    f, axes = plt.subplots(1, 2, figsize=(12, 8))
    axes[0].imshow(original_image)
    axes[0].axis("off")
    axes[0].set_title("Original Image")
    axes[1].imshow(page_edge_gpu)
    axes[1].axis("off")
    axes[1].set_title("PhyCV Directional Edge Detection")
    plt.savefig(os.path.join(output_path, "PAGE_GPU_compare.jpg"), bbox_inches="tight")
    page_gpu_result = Image.fromarray((page_edge_gpu * 255).astype(np.uint8))
    page_gpu_result.save(os.path.join(output_path, "PAGE_GPU_output.jpg"))


if __name__ == "__main__":
    main()
