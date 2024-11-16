import os
import time
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from phycv import VLight

def main():
    # indicate image file, height and width of the image
    img_file = "./assets/input_images/street_scene.png"
    original_image = cv2.imread(img_file)  # Load the image in BGR format (OpenCV default)

    output_path = "./output/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # VLight parameters
    v = 0.90

    # Run VLight CPU version (expects and outputs BGR)
    vlight_cpu = VLight()

    start_time = time.time()
    vlight_output_cpu = vlight_cpu.run(img_file=img_file, v=v, color=False, lut=True)
    print(f"Time elapsed: {(time.time() - start_time) * 1000}")

    # Visualize the results
    f, axes = plt.subplots(1, 2, figsize=(12, 8))

    # Display original BGR image (converted to RGB for visualization)
    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].axis("off")
    axes[0].set_title("Original Image")

    # Display VLight output (also converted to RGB for visualization)
    axes[1].imshow(cv2.cvtColor(vlight_output_cpu, cv2.COLOR_BGR2RGB))
    axes[1].axis("off")
    axes[1].set_title("VLight Low-Light Enhancement")

    # Save the figure with both images displayed side by side
    plt.savefig(os.path.join(output_path, "VLight_CPU_compare.jpg"), bbox_inches="tight")

    # Save the VLight output in BGR format as a separate image
    vlight_cpu_result = Image.fromarray(cv2.cvtColor(vlight_output_cpu, cv2.COLOR_BGR2RGB))
    vlight_cpu_result.save(os.path.join(output_path, "VLight_CPU_output.jpg"))

if __name__ == "__main__":
    main()