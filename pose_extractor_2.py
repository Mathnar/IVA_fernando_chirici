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

RESIZE = False

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# names = ['videos/r_s2.mp4', 'videos/r_s2_1.mp4']
# window_titles = ['r_s2', 'r_s2_1']




# def get_skeleton(cap):
#     for i in enumerate(cap):
#         cap = cap[i]
#         # print('\ncap ', cap, cap[i])
#         while cap.isOpened():
#             success, image = cap.read()
#             if not success:
#                 print("Ignoring empty camera frame.")
#                 # If loading a video, use 'break' inst ead of 'continue'.
#                 break
#                 # continue # per webcam
#             # else:
#             #   print('\nsuccess')
#
#             # To improve performance, optionally mark the image as not writeable to
#             # pass by reference.
#             image.flags.writeable = False
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             results = pose.process(image)
#
#             # Draw the pose annotation on the image.
#             image.flags.writeable = True
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#             mp_drawing.draw_landmarks(
#                 image,
#                 results.pose_landmarks,
#                 mp_pose.POSE_CONNECTIONS,
#                 landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
#             # Flip the image horizontally for a selfie-view display.
#             cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
#             if cv2.waitKey(5) & 0xFF == 27:
#                 break


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def midpoint(x1, y1, z1, x2, y2, z2):
    return (x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2


def skeleton_extraction(pose, cap, iterator, ret, gray, frames, names, window_titles, SAVE_FOLDER, adj):
    for i, c in enumerate(cap):
        if c is not None:
            ret[i], frames[i] = c.read()
    xyz_array = np.zeros(shape=(1, 3))
    xyz_array = np.delete(xyz_array, 0, axis=0)
    for i, f in enumerate(frames):
        if ret[i]:
            f = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
            f.flags.writeable = False
            results = pose.process(f)
            f.flags.writeable = True
            f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            pose_land0 = results.pose_world_landmarks
            pose_land0_relative = results.pose_landmarks

            xyz_array = extract_array_from_landmarks(pose_land0, xyz_array)

            # print('\n\nxyz_array ', xyz_array,'\nremove_Arr', remove_arr,  len(xyz_array), len(remove_arr), type(xyz_array), type(remove_arr))

            # Rimuovo landmark che non mi interessano
            remove_arr = remove_landmarks(pose_land0)
            i, xyz_array = undesired_landmarks_removal(i, remove_arr, xyz_array)

            if len(xyz_array) < 16:
                print('\nFrame con troppi pochi landmarks')
                break
            #xyztest = xyz_array

            #debug print('\nArray dopo la rimozione dei landmark indesiderati:\n\n ', xyztest,
            #debug       '\nSize [expected 16]: ', len(xyz_array))
            # visualize_skeleton(xyztest)
            # print('exit')

            # START NORMALIZATION
            # Estrazione valore medio e normalizzazione vettore dei joints (centramento sul baricentro e normalizzazione)
            xyz_array = normalize_centering_and_size(pose_land0, xyz_array)
            #debug print('xyz norm ', xyz_array)

            # Normalizzazione next step
            # Calcolo matrice M per la normalizzazione dell'inquadratura
            xyz_array = normalize_framing(pose_land0, xyz_array)
            #debug print('\nFine normalizzazione \nxyz_array_final\n', xyz_array)

            iterator = visualize_or_save_skeleton(xyz_array, iterator, adj, SAVE_FOLDER, True)

            # print('\nexit')
            # exit()

            mp_drawing.draw_landmarks(
                f,
                pose_land0_relative,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # mp_drawing.plot_landmarks( @
            #     results.pose_landmarks, mp_pose.POSE_CONNECTIONS, name='a') @
            # img = Image.open('img/afoo.png') @

            # pose_land1 = results1.pose_landmarks
            # mp_drawing.draw_landmarks(
            #     f1,
            #     pose_land1,
            #     mp_pose.POSE_CONNECTIONS,
            #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # mp_drawing.plot_landmarks( @
            #     results1.pose_landmarks, mp_pose.POSE_CONNECTIONS, name='b') @
            # img1 = Image.open('img/bfoo.png') @

            # img_c = get_concat_h(img, img1).save('img/dst.png') @
            # frame = cv2.imread('img/dst.png') @
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) @
            # cv2.imshow('sd0', frame) @

            if len(window_titles) > 1:
                pointer = i
            else:
                pointer = 0
            # cv2.imshow(window_titles[pointer], cv2.flip(f, 1))
            break
        # print('\nstop')
    return iterator


def extract_array_from_landmarks(pose_land0, xyz_array):
    str_pose = str(pose_land0)
    #debug print('\nPose_landmarks as string: ', str_pose)
    landmark_coord = str_pose.split('landmark')
    landmark_coord = landmark_coord[1:]
    # Estraggo le xyz e metto in un array 3d
    #debug print('\nGet only landmarks: ', landmark_coord)
    for j in range(0, len(landmark_coord)):
        # print('\n',landmark_coord[i],'\n')
        xyz_array = get_xyz(xyz_array, landmark_coord[j])
        # print('xyz_array ', xyz_array)
    return xyz_array


def undesired_landmarks_removal(i, remove_arr, xyz_array):
    while i < len(xyz_array):
        #debug print('\nanalizzo ', xyz_array[i])
        if xyz_array[i] in remove_arr:
            #debug print('\nrimuovo ', xyz_array[i])
            xyz_array = np.delete(xyz_array, i, axis=0)
        else:
            i += 1
    # 0: left shoulder
    # 1: right shoulder
    # 2: left elbow
    # 3: right elbow
    # 4: left wrist
    # 5: right wrist
    # 6: left index
    # 7: right index
    # 8: left hip
    # 9: right hip
    # 10: left knee
    # 11: right knee
    # 12: left heel
    # 13: right heel
    # 14: left foot index
    # 15: right foot index
    # ############################final list
    return i, xyz_array


def normalize_centering_and_size(pose_land0, xyz_array):
    left_ank_coo = pose_land0.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ank_coo = pose_land0.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
    #print('\n###### ', left_ank_coo)
    #print('\n###### ', right_ank_coo)
    left_xyz = np.array([left_ank_coo.x, left_ank_coo.y, left_ank_coo.z])
    right_xyz = np.array([right_ank_coo.x, right_ank_coo.y, right_ank_coo.z])
    pelv = midpoint(left_ank_coo.x, left_ank_coo.y, left_ank_coo.z, right_ank_coo.x, right_ank_coo.y, right_ank_coo.z)
    #print('\nleft, ', left_xyz, '\nright ', right_xyz, '\npelv ', pelv, '\ntypes: ', type(left_xyz)
    #      , type(right_xyz), type(pelv))
    center_points = [left_xyz, right_xyz, pelv]
    # Estrazione vettore medio fianchi e torso
    mean_vector = np.zeros(dtype='float64', shape=(1, 3))  # [0 0 0]
    for point in center_points:
        mean_vector += point
    mean_vector = mean_vector / len(center_points)
    #print('\nbig: ', mean_vector, pelv, '\n\n')
    mean_vector = np.reshape(mean_vector, (3,))
    #print('\nMean_vector:\n', mean_vector, '\n_____________\n')
    for joint in xyz_array:
     #   print('\njoint before mean_vector: ', joint)
        joint -= mean_vector
      #  print('\njoint after mean_vector: ', joint)
    #print('\nwtf: ', np.linalg.norm(xyz_array))
    xyz_array = xyz_array / np.linalg.norm(xyz_array)
    return xyz_array


def normalize_framing(pose_land0, xyz_array):
    lcla = pose_land0.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    Jhl = np.transpose(np.array([lcla.x, lcla.y, lcla.z]))
    rcla = pose_land0.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    Jt = np.transpose(np.array([rcla.x, rcla.y, rcla.z]))
    #print('\nCoords transp \nlcla: ', Jhl, ' \nrcla: ', Jt)
    Jhl = Jhl / np.linalg.norm(Jhl)
    norm2 = np.linalg.norm(Jhl)
    #print('\n Jhl after norm ', Jhl, type(Jhl))
    tras = np.transpose(Jhl)
    temp = np.dot((tras / norm2), Jt).item()
    Jhl_ort = Jt - (temp * Jhl)
    #print('\n Jhl_ort ', Jhl_ort, type(Jhl_ort), type(np.array(Jhl_ort)))
    Jhl_ort = Jhl_ort / np.linalg.norm(Jhl_ort)
    cross_product_vector = np.cross(Jhl, Jhl_ort, axis=0)  # added reshape qua
    cross_product_vector = cross_product_vector / np.linalg.norm(cross_product_vector)
    # cross_product_vector = np.transpose(cross_product_vector)
    #print('\ncross_product_vector ', cross_product_vector, type(cross_product_vector))
    #print('\njhl..resh ', Jhl, Jhl.reshape(-1, 1))
    # exit()
    M = np.concatenate((Jhl.reshape(-1, 1), Jhl_ort.reshape(-1, 1), cross_product_vector.reshape(-1, 1)), axis=1)
    x_tilde = np.transpose(xyz_array)
    #print('\n-- ', x_tilde, '\n--- m ', M)
    x_tilde = np.dot(np.transpose(M), x_tilde)
    xyz_array = np.transpose(x_tilde)
    return xyz_array


# def visualize_skeleton_2(skeleton):
#     rotation_matrix = np.dot(np.dot([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]), [[cos(radians(-10)), 0, sin(radians(-10))], [0, 1, 0], [-sin(radians(-10)), 0, cos(radians(-10))]])
#     coords = np.dot(skeleton, rotation_matrix)
#     #coords = skeleton
#     edges = adj
#
#     # import tensorflow as tf
#     # with tf.Session() as sess:
#     #     # matrice adiacenza degli archi che collegano i joints (18x2)
#     #     skeleton["edges"] = sess.run(edges_tensor)
#     #
#     #     # coordinate dei joint dello skeleton (3D)(19x3)
#     #     skeleton["joint_coordinates"] = sess.run(poses_tensor)[0]  # joint_coordinates ha lo [0] perchè almeno viene rappresentato come una matrice (19x3)
#     #
#     #     # Nomi di ogni joint, sono nell'ordine in cui compaiono in joint_coordinates (19x1)
#     #     skeleton["joint_names"] = sess.run(joint_names_tensor)
#
#     # sess.close()
#
#     plt.switch_backend('TkAgg')
#     # noinspection PyUnresolvedReferences
#     from mpl_toolkits.mplot3d import Axes3D
#
#     # Matplotlib interprets the Z axis as vertical, but our pose
#     # has Y as the vertical axis.
#     # Therefore we do a 90 degree rotation around the horizontal (X) axis
#     #coords2 = coords.copy()
#     #coords[:, 1], coords[:, 2] = coords2[:, 2], -coords2[:, 1]
#
#     fig = plt.figure(figsize=(10, 5))
#
#     pose_ax = fig.add_subplot(1, 1, 1, projection='3d')
#     pose_ax.set_title('Prediction')
#     range_ = np.amax(np.abs(skeleton))
#     print('\nrange', range_)
#     pose_ax.set_xlim3d(-range_, range_)
#     pose_ax.set_ylim3d(-range_, range_)
#     pose_ax.set_zlim3d(-range_, range_)
#     plt.ylabel("y")
#     plt.xlabel("x")
#     for i_start in range(0, len(edges)):
#
#         for k in edges[i_start]:
#             x = [coords[i_start][0], coords[k][0]]
#             y = [coords[i_start][1], coords[k][1]]
#             z = [coords[i_start][2], coords[k][2]]
#
#             # pose_ax.plot(*zip(coords[i_start], coords[k]), marker='o', markersize=2)
#             pose_ax.scatter(coords[i_start][0], coords[i_start][1], coords[i_start][2], c='red', s=70)
#             pose_ax.plot(x, y, z, color='black')
#
#
#     pose_ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=2)
#
#     fig.tight_layout()
#     plt.draw()
#     plt.show()


def visualize_or_save_skeleton(skeleton, iterator, adj, SAVE_FOLDER, save=False):
    # print('\nsave:: ', save)
    rotation_matrix = np.dot(np.dot([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
                             [[cos(radians(-10)), 0, sin(radians(-10))], [0, 1, 0],
                              [-sin(radians(-10)), 0, cos(radians(-10))]])

    coords = np.dot(skeleton, rotation_matrix)
    edges = adj

    plt.switch_backend('TkAgg')
    # noinspection PyUnresolvedReferences
    from mpl_toolkits.mplot3d import Axes3D

    # Matplotlib interprets the Z axis as vertical, but our pose
    # has Y as the vertical axis.
    # Therefore we do a 90 degree rotation around the horizontal (X) axis
    # coords2 = coords.copy()
    # coords[:, 1], coords[:, 2] = coords2[:, 2], -coords2[:, 1]

    fig = plt.figure(figsize=(10, 5))

    pose_ax = fig.add_subplot(1, 1, 1, projection='3d')
    pose_ax.set_title('Prediction')
    range_ = np.amax(np.abs(skeleton))
    pose_ax.set_xlim3d(-range_, range_)
    pose_ax.set_ylim3d(-range_, range_)
    pose_ax.set_zlim3d(-0, range_)
    plt.ylabel("y")
    plt.xlabel("x")
    for i_start in range(0, len(edges)):

        for k in edges[i_start]:
            # x = [coords[i_start][0], coords[k][0]]
            # y = [coords[i_start][1], coords[k][1]]
            # z = [coords[i_start][2], coords[k][2]]

            # pose_ax.plot(*zip(coords[i_start], coords[k]), marker='o', markersize=2)
            pose_ax.scatter(coords[i_start][0], coords[i_start][1], coords[i_start][2], c='red', s=40)
            # pose_ax.plot(x, y, z, color='black')
            pose_ax.plot(*zip(coords[i_start], coords[k]), marker='o', markersize=2)

    pose_ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=2)

    fig.tight_layout()
    if save and (iterator % 5) == 0 or iterator == 0:
        # print('\nqua')
        plt.savefig(SAVE_FOLDER + '/' + str(iterator) + '.png')
        # salvo su txt coords
        with open(SAVE_FOLDER + '_coords/' + str(iterator) + '.txt', 'w') as f:
            f.write(str(skeleton))
        #exit()
    elif not save:
        plt.show()
    iterator += 1
    return iterator



def remove_landmarks(pose_land0):
    remove_list = []
    remove_list.append(pose_land0.landmark[mp_pose.PoseLandmark.NOSE])
    remove_list.append(pose_land0.landmark[mp_pose.PoseLandmark.RIGHT_EYE])
    remove_list.append(pose_land0.landmark[mp_pose.PoseLandmark.LEFT_EYE])
    remove_list.append(pose_land0.landmark[mp_pose.PoseLandmark.RIGHT_EYE_INNER])
    remove_list.append(pose_land0.landmark[mp_pose.PoseLandmark.LEFT_EYE_INNER])
    remove_list.append(pose_land0.landmark[mp_pose.PoseLandmark.RIGHT_EYE_OUTER])
    remove_list.append(pose_land0.landmark[mp_pose.PoseLandmark.LEFT_EYE_OUTER])
    remove_list.append(pose_land0.landmark[mp_pose.PoseLandmark.RIGHT_EAR])
    remove_list.append(pose_land0.landmark[mp_pose.PoseLandmark.LEFT_EAR])
    remove_list.append(pose_land0.landmark[mp_pose.PoseLandmark.MOUTH_LEFT])
    remove_list.append(pose_land0.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT])
    remove_list.append(pose_land0.landmark[mp_pose.PoseLandmark.RIGHT_THUMB])
    remove_list.append(pose_land0.landmark[mp_pose.PoseLandmark.LEFT_THUMB])
    remove_list.append(pose_land0.landmark[mp_pose.PoseLandmark.RIGHT_PINKY])
    remove_list.append(pose_land0.landmark[mp_pose.PoseLandmark.LEFT_PINKY])
    remove_list.append(pose_land0.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE])
    remove_list.append(pose_land0.landmark[mp_pose.PoseLandmark.LEFT_ANKLE])
    remo_ls = []
    for item in remove_list:
        remo_ls.append(np.array([item.x, item.y, item.z]))
    return np.array(remo_ls)


def get_xyz(xyz_array, str_pose):
    x_p = str_pose.split('{')
    x_p = x_p[1].split('}')
    x_p = (x_p[0].split('x:')[1]).split('y:')
    y_p = x_p[1].split('z:')
    z_p = y_p[1].split('vis')
    xx = x_p[0]
    yy = y_p[0]
    zz = z_p[0]
    xyz_array = np.vstack((xyz_array, (float(xx), float(yy), float(zz))))
    return xyz_array


# def ankle_size(results):
#     left_ank_coo = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
#     right_ank_coo = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
#     # print('\n###### ', left_ank_coo, type(left_ank_coo))
#     # print('\n@@@@@@', list(left_ank_coo), type(list(left_ank_coo)))
#     left_xyz = np.array([left_ank_coo.x, left_ank_coo.y, left_ank_coo.z])
#     right_xyz = np.array([right_ank_coo.x, right_ank_coo.y, right_ank_coo.z])
#     dist = np.linalg.norm(left_xyz - right_xyz)
#     return dist


def resize_video(names):
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
        return cap


def empty_folders(IMG_FOLDER):
    folder = IMG_FOLDER
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    folder = IMG_FOLDER + '_coords'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
