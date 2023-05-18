"""
Run PST on a video. 
All steps (load_img -> init_kernel -> apply_kernel) are performed seperately.
For fixed kernel, we only need to initialize the kernel once for the first frame 

"""

import os

import imageio.v3 as iio
import numpy as np
import torch
import torchvision

from phycv import PST_GPU


def main():
    # indicate the video to be processed
    vid = torchvision.io.read_video("./assets/input_videos/video_nature.mp4")
    print("video loaded!")

    output_path = "./output/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

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
    thresh_min = 0
    thresh_max = 0.9
    morph_flag = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # run PST for each frame
    for i in range(length):
        frame = torch.permute(vid_frames[i], (2, 0, 1))
        frame = frame.to(device)
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
    # save the results for each frame
    concat_frames = []
    for i in range(length):
        raw_frame = vid_frames[i].numpy()
        pst_frame = (pst_out_vid[i][:, :, None] * 255).astype(np.uint8)
        pst_frame = np.repeat(pst_frame, 3, -1)
        concat_frame = np.concatenate((raw_frame, pst_frame), 1)
        concat_frames.append(concat_frame)

    iio.imwrite("output/PST_video_demo.mp4", concat_frames, fps=20)


if __name__ == "__main__":
    main()
