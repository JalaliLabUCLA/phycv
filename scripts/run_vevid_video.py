"""
Run VEVID on a video. 
All steps (load_img -> init_kernel -> apply_kernel) are performed seperately.
For fixed kernel, we only need to initialize the kernel once for the first frame 

"""

import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from phycv import VEVID_GPU


def main():
    # indicate the video to be processed
    vid = torchvision.io.read_video("./assets/input_videos/video_building.mp4")
    print("video loaded!")

    # get how many frames are in the video
    # create a empty array to store the VEViD output
    vid_frames = vid[0]
    length = vid_frames.shape[0]
    frame_h = vid_frames[0].shape[0]
    frame_w = vid_frames[0].shape[1]
    vevid_out_vid = np.zeros((length, 3, frame_h, frame_w))

    # indicate VEViD parameters
    S = 0.4
    T = 0.002
    b = 0.5
    G = 0.6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # run VEViD for each frame
    for i in range(length):
        frame = torch.permute(vid_frames[i], (2, 0, 1)) / 255.0
        frame = frame.to(device)
        # if it is the first frame, we have to initialized the kernel
        if i == 0:
            vevid = VEVID_GPU(device=device)
            vevid.load_img(img_array=frame)
            vevid.init_kernel(S, T)
            vevid.apply_kernel(b, G)
        # for rest of the frames, directly use the initialized kernel
        else:
            vevid.load_img(img_array=frame)
            vevid.apply_kernel(b, G)

        vevid_out_vid[i] = vevid.vevid_output.cpu().numpy()

    print("create video...")
    output_path = "./output/VEViD/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # save the results for each frame
    for i in range(length):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 16))
        ax1.imshow(vid_frames[i])
        ax1.axis("off")
        ax1.set_title("Original Video Frame", fontsize=16)
        ax2.imshow(np.transpose(vevid_out_vid[i], (1, 2, 0)))
        ax2.axis("off")
        ax2.set_title("Real-time Low-Light Enhancement by PhyCV", fontsize=18)
        idx = (4 - len(str(i))) * "0" + str(i)
        plt.savefig(os.path.join(output_path, f"{idx}.jpg"), bbox_inches="tight")
        plt.close()

    # create video from the processed frames
    with imageio.get_writer("./output/VEViD_demo.mp4", fps=25) as writer:
        for filename in sorted(os.listdir(output_path)):
            image = imageio.imread(output_path + filename)
            writer.append_data(image)


if __name__ == "__main__":
    main()
