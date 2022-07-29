import time

import numpy as np
import mediapipe as mp
import cv2
import moviepy.editor as mpy

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


names = ['videos/r_s2.mp4', 'videos/r_s1.mp4']
window_titles = ['c', 'd']
#names = ['videos/s2.mp4', 'videos/s1.mp4']
#window_titles = ['s2', 's1']

cap = [cv2.VideoCapture(i) for i in names]
print('\nnames: ', names, '\nsize di 0: \n ', cap[0].get(cv2.CAP_PROP_FRAME_HEIGHT), cap[0].get(cv2.CAP_PROP_FRAME_WIDTH) )

RESIZE = False
if RESIZE:
    new_names = []
    for i in range(0, len(names)):
        cap = cv2.VideoCapture(names[i])
        clip = mpy.VideoFileClip(names[i])
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        sp = names[i].split('/')

        new_name = '/r_' + sp[1]
        new_name = sp[0] + new_name
        if height < width:
            clip_resized = clip.resize(height=360)
            clip_resized.write_videofile(new_name)
        else:
            clip_resized = clip.resize(width=360)
            clip_resized.write_videofile(new_name)

        new_names.append(new_name)
    names = new_names

    cap = [cv2.VideoCapture(i) for i in names]
    print('\nnames: ', names, '\nsize di 0: \n ', cap[0].get(cv2.CAP_PROP_FRAME_HEIGHT), cap[0].get(cv2.CAP_PROP_FRAME_WIDTH) )



frames = [None] * len(names)
gray = [None] * len(names)
ret = [None] * len(names)


def get_skeleton(cap):
    for i in enumerate(cap):
        cap = cap[i]
        #print('\ncap ', cap, cap[i])
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' inst ead of 'continue'.
                break
                # continue # per webcam
            #else:
             #   print('\nsuccess')

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break


def _gets(cap):

    for i, c in enumerate(cap):
        print('\n cap: ', cap[i], '\nc ', c)
        if c is not None:
            ret[i], frames[i] = c.read()
            #print('\nret[i] ', ret[i],'\nframes[i] ', frames[i])

    i = 0
    print('\nqui')
    print('\ncap ', cap, 'ret ', ret, '\nframes ', frames)
    for i, f in enumerate(frames):
        print('\ncap[i] ', cap[i], 'ret[i] ', ret[i])
        if ret[i] is True:
            print('\nret[i] is true ', ret[i] is True,'\nf ', f)
            # gray[i] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            f.flags.writeable = False

            results = pose.process(f)
            print('\nafter pose process: res', results)
            f.flags.writeable = True
            f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                f,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # left_ank_coo = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
            # right_ank_coo = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
            # #print('\n###### ', left_ank_coo, type(left_ank_coo))
            # #print('\n@@@@@@', list(left_ank_coo), type(list(left_ank_coo)))
            # left_xyz = np.array([left_ank_coo.x, left_ank_coo.y, left_ank_coo.z])
            # right_xyz = np.array([right_ank_coo.x, right_ank_coo.y, right_ank_coo.z])
            # dist = np.linalg.norm(left_xyz - right_xyz)
            # print('\n-> ', dist)
            cv2.imshow(window_titles[i], cv2.flip(f, 1))
            print('\nqui1')
            #time.sleep(0.006)
        print('\nstop ')
        #time.sleep(10)


with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    while True:
        _gets(cap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
          break


    for c in cap:
        if c is not None:
            c.release()
    cv2.destroyAllWindows()