import json
import os
import shutil
import time
from math import radians, cos, sin

import numpy as np
import mediapipe as mp
import cv2
import moviepy.editor as mpy
from PIL import Image
from matplotlib import pyplot as plt
from pose_extractor_2 import *
# from EuclideanDistances import *
from Angle_functions import *
from Combined_functions import *

RESIZE = False
EXTRACTION_FLAG = False
EUCLIDEAN = False
ANGLE = True
COMBINED = False

EXERCISE = 'squats'
video_path_names = ['uservideos/squat0_wrong_3.mp4']
video_names = ['squat0_wrong_3']    # s1, r_s2

WORKING_FOLDER = EXERCISE + '_tester'   # squats_trainer
adj = [[1, 2], [0, 3], [0, 4], [1, 5], [2, 6], [3, 7], [4], [5], [0, 9, 10], [1, 8, 11], [8, 12], [9, 13],
       [10, 14], [11, 15], [12], [13]]


if __name__ == '__main__':
    if EXTRACTION_FLAG:
        empty_folders(WORKING_FOLDER)
        cap = [cv2.VideoCapture(i) for i in video_path_names]
        print('\nnames: ', video_path_names, '\nsize di 0: \n ', cap[0].get(cv2.CAP_PROP_FRAME_HEIGHT),
              cap[0].get(cv2.CAP_PROP_FRAME_WIDTH))

        if RESIZE:
            cap = resize_video(video_path_names)

            # print('\nnames: ', names, '\nsize di 0: \n ', cap[0].get(cv2.CAP_PROP_FRAME_HEIGHT), cap[0].get(cv2.CAP_PROP_FRAME_WIDTH) )

        frames = [None] * len(video_path_names)
        gray = [None] * len(video_path_names)
        ret = [None] * len(video_path_names)

        with mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:
            print(pose)
            iterator = 0
            while iterator < 600:
                iterator = skeleton_extraction(pose, cap, iterator, ret, gray, frames, video_path_names, video_names, WORKING_FOLDER, adj)
                # print(cap)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            for c in cap:
                if c is not None:
                    c.release()
            cv2.destroyAllWindows()

    # confrontino
    # euclidean_identify_repetitions()


    vis_err = True
    joint_th = pose_th = 1
    if EUCLIDEAN:
        rep_distance = repetitions_euclidean_distance(EXERCISE)
        identify_euclidean_errors(EXERCISE, rep_distance, joint_th, pose_th, vis_err)
    if ANGLE:
        rep_distance = repetitions_angles_distance(EXERCISE)
        identify_angles_errors(EXERCISE, rep_distance, joint_th, pose_th, vis_err)
    if COMBINED:
        rep_distance = repetitions_combined_distance(EXERCISE)
        identify_combined_errors(EXERCISE, rep_distance, joint_th, pose_th, vis_err)


