"""
Run PST on a video. 
All steps (load_img -> init_kernel -> apply_kernel) are performed seperately.
For fixed kernel, we only need to initialize the kernel once for the first frame 

"""

import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from phycv import PST_GPU


def main():
    # indicate the video to be processed
    vid = torchvision.io.read_video("./assets/input_videos/video_nature.mp4")
    print("video loaded!")

    # get how many frames are in the video
    # create a empty array to store the PST output
    vid_frames = vid[0]
    length = vid_frames.shape[0]
    frame_h = vid_frames[0].shape[0]
    frame_w = vid_frames[0].shape[1]
    pst_out_vid = np.zeros((length, frame_h, frame_w))

    # indicate PST parameters
    S = 0.5
    W = 15
    sigma_LPF = 0.15
    thresh_min = -1
    thresh_max = 0.003
    morph_flag = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # run PST for each frame
    for i in range(length):
        frame = torch.permute(vid_frames[i], (2, 0, 1))
        # if it is the first frame, we have to initialized the kernel
        if i == 0:
            pst = PST_GPU(device=device)
            pst.load_img(img_array=frame)
            pst.init_kernel(S, W)
            pst.apply_kernel(sigma_LPF, thresh_min, thresh_max, morph_flag)
        # for rest of the frames, directly use the initialized kernel
        else:
            pst.load_img(img_array=frame)
            pst.apply_kernel(sigma_LPF, thresh_min, thresh_max, morph_flag)
        # save output into the array
        pst_out_vid[i] = pst.pst_output.cpu().numpy()

    print("create video...")
    output_path = "./output/PST/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # save the results for each frame
    for i in range(length):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 16))
        ax1.imshow(vid_frames[i])
        ax1.axis("off")
        ax1.set_title("Original Video Frame", fontsize=16)
        ax2.imshow(pst_out_vid[i], cmap="gray")
        ax2.axis("off")
        ax2.set_title("Real-time Edge Detection by PhyCV", fontsize=16)
        idx = (4 - len(str(i))) * "0" + str(i)
        plt.savefig(os.path.join(output_path, f"{idx}.jpg"), bbox_inches="tight")
        plt.close()

    # create video from the processed frames
    with imageio.get_writer("./output/PST_demo.mp4", fps=20) as writer:
        for filename in sorted(os.listdir(output_path)):
            image = imageio.imread(output_path + filename)
            writer.append_data(image)


if __name__ == "__main__":
    main()
