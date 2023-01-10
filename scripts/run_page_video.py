"""
Run PAGE on a video. 
All steps (load_img -> init_kernel -> apply_kernel -> create_edge) are performed seperately.
For fixed kernel, we only need to initialize the kernel once for the first frame 

"""

import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from phycv import PAGE_GPU


def main():
    # indicate the video to be processed
    vid = torchvision.io.read_video("./assets/input_videos/video_nature.mp4")
    print("video loaded!")

    # get how many frames are in the video
    # create a empty array to store the PAGE output
    vid_frames = vid[0]
    length = vid_frames.shape[0]
    frame_h = vid_frames[0].shape[0]
    frame_w = vid_frames[0].shape[1]
    page_out_vid = np.zeros((length, frame_h, frame_w, 3))

    # indicate PAGE parameters
    mu_1 = 0
    mu_2 = 0.35
    sigma_1 = 0.05
    sigma_2 = 0.8
    S1 = 0.8
    S2 = 0.8
    sigma_LPF = 0.1
    thresh_min = -1
    thresh_max = 0.002
    morph_flag = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # run PAGE for each frame
    for i in range(length):
        frame = torch.permute(vid_frames[i], (2, 0, 1))
        frame = frame.to(device)
        # if it is the first frame, we have to initialized the kernel
        if i == 0:
            page = PAGE_GPU(direction_bins=10, device=device)
            page.load_img(img_array=frame)
            page.init_kernel(mu_1, mu_2, sigma_1, sigma_2, S1, S2)
            page.apply_kernel(sigma_LPF, thresh_min, thresh_max, morph_flag)
            page.create_page_edge()
        # for rest of the frames, directly use the initialized kernel
        else:
            page.load_img(img_array=frame)
            page.apply_kernel(sigma_LPF, thresh_min, thresh_max, morph_flag)
            page.create_page_edge()

        page_out_vid[i] = page.page_edge.cpu().numpy()

    print("create video...")
    output_path = "./output/PAGE/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # save the results for each frame
    for i in range(length):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 16))
        ax1.imshow(vid_frames[i])
        ax1.axis("off")
        ax1.set_title("Original Video Frame", fontsize=16)
        ax2.imshow(page_out_vid[i])
        ax2.axis("off")
        ax2.set_title("Real-time Directional Edge Detection by PhyCV", fontsize=18)
        idx = (4 - len(str(i))) * "0" + str(i)
        plt.savefig(os.path.join(output_path, f"{idx}.jpg"), bbox_inches="tight")
        plt.close()

    # create video from the processed frames
    with imageio.get_writer("./output/PAGE_demo.mp4", fps=25) as writer:
        for filename in sorted(os.listdir(output_path)):
            image = imageio.imread(output_path + filename)
            writer.append_data(image)


if __name__ == "__main__":
    main()
