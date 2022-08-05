import shutil

import time
import json
from math import radians

from cmath import *
import numpy as np
from matplotlib import pyplot as plt
from video_creator import *


def find_coord(skeleton, pointer):
    return skeleton[pointer]


def find_vector(P, Q):
    vect = P - Q
    norm = np.linalg.norm(vect)
    return vect / norm


def from_angle_to_joint_index(angle):
    link = [0,4,1,3,5,1,0,9,8,9,8,9,8,11,10,8,9,13,12]
    return link[angle]


def get_coords_from_file(exercise, filename):
    filename = str(filename)
    with open(exercise + '_coords\\' + filename + '.txt', 'r') as f:
        lis = []
        for line in f:
            z = line.strip().strip('[').strip(']').split(' ')
            z = list(filter(None, z))
            lis.append((z[0], z[1], z[2]))
        return (np.array(lis)).astype(np.float)


def from_jointname_to_jointindex(name):
    link_n = [b'lsho', b'rsho', b'lelb', b'relb', b'lwri', b'rwri', b'lind',
              b'rind', b'lhip', b'rhip', b'lkne', b'rkne',b'lheel', b'rhell',
              b'lfind', b'rfind']
    return link_n.index(name)


def from_jointindex_to_jointname(index):
    link_i = [b'lsho', b'rsho', b'lelb', b'relb', b'lwri', b'rwri', b'lind',
              b'rind', b'lhip', b'rhip', b'lkne', b'rkne',b'lheel', b'rhell',
              b'lfind', b'rfind']
    return link_i[index]


# Individua le ripetizioni nei vari esercizi
def euclidean_identify_repetitions(exercise):
    skeleton_ref = get_coords_from_file(exercise, '0')
    distances = []
    n_frames = len([name for name in os.listdir(exercise)])

    for frame in range(0, (n_frames*5), 5):
        skeleton = get_coords_from_file(exercise, str(frame))
        distances.append(np.linalg.norm(skeleton_ref - skeleton))
        # '\nframe ', frame, ' / ', (n_frames*5-5))
        # carico tutte le altre coord e cerco distanza dal primio (ref) che si suppone sia la posizione iniziale
    #exit()

    # print('\ndist ', distances)

    candidate_PoI = []
    PoI = [(0, 0)]
    thr = np.mean(np.array(distances))
    for i in range(len(distances)):
        if distances[i] <= thr:
            candidate_PoI.append(i * 5)
    #print('\ncandidate_PoI', candidate_PoI)
    i = 0
    while i < len(candidate_PoI):
        temp = []
        j = 0
        while i + j + 1 < len(candidate_PoI) and candidate_PoI[i + j + 1] - candidate_PoI[i + j] <= 10:
            temp.append(candidate_PoI[i + j])
            j += 1
        temp.append(candidate_PoI[i + j])
        if len(temp) > 1:
            previous_el = int(PoI[len(PoI) - 1][1])
            PoI.append((previous_el, temp[int(len(
                temp) / 2)]))  # aggiungo alla lista PoI l'intervallo di una certa ripetizione (0-50, 50-90, ecc...)
            i += j
        else:
            previous_el = int(PoI[len(PoI) - 1][1])
            PoI.append((previous_el, temp[0]))
        i += 1
    # '\nPOI ', PoI)
    PoI.pop(0)
    # print('\nPOI ', PoI, PoI[0][1])
    if PoI[0][1] <= 30 and len(PoI)>1:  # vincolo che evita che il primo punto di interesse sia preso entro un secondo dall'inizio dell'esercizio
        PoI.pop(0)
    #    print('\nPOI ', PoI)

        PoI[0] = (0, PoI[0][1])
    #    print('\nòò ', PoI)
    if len(PoI) > 1:
        PoI.pop(len(PoI) - 1)
    previous_el = int(PoI[len(PoI) - 1][1])
    PoI.append((previous_el, n_frames * 5 - 5))
    return PoI



def identify_frame_errors(repetition_distance, thr_multiplier=1.5):
    error_list = []
    repetition_list = []
    distances, trainer_index, user_index = repetition_distance
    for triple in distances:
        user_repetition_num = triple[1]
        trainer_repetition_num = triple[0]
        path = triple[2][1]
        skeleton_distances = triple[2][2]
        thr = np.mean(skeleton_distances) * thr_multiplier
        for i in range(len(skeleton_distances)):
            if skeleton_distances[i] > thr:
                # print('\n ful ', skeleton_distances, '\nskel ', skeleton_distances[i], thr)
                # if i%3==0:
                #     exit()
                user_frame = user_index[user_repetition_num][path[i][1]]
                trainer_frame = trainer_index[trainer_repetition_num][path[i][0]]
                error_list.append((trainer_frame, user_frame))
                repetition_list.append(user_repetition_num)
    print("Errori commessi: " + str(len(error_list)))
    print("Nelle coppie di frame: " + str(error_list))
    # print('\n ff ', error_list[0], ' èèè ', error_list[0][1] )
    # exit()
    return error_list, repetition_list


def error_frame_higlight(joint_frame_error_bridge, exercise, adj, type):
    empty_folders('img_out_user', False)
    k=1
    if k:
        rotation_matrix = np.dot(np.dot([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
                                 [[cos(radians(-10)), 0, sin(radians(-10))], [0, 1, 0],
                                  [-sin(radians(-10)), 0, cos(radians(-10))]])
        frames_number = np.array(len([name for name in os.listdir(exercise + '_tester')]))


        joint_frame_error_bridge = np.array(joint_frame_error_bridge)
        range_ = 0
        for i in range(0, frames_number*5, 5):
            item = i
            s_t = 1
            error_flag = False
            list_of_errors = []
            correct_position = 0
            skeleton_user = get_coords_from_file(exercise + '_tester', str(item))

            array_of_broken_frames = np.unique(joint_frame_error_bridge[:, 1])
            if i in array_of_broken_frames:
                error_flag = True
                for broken_frame in joint_frame_error_bridge:
                    if i == int(broken_frame[1]):
                        list_of_errors.append(broken_frame[2])  # lista contenente i joint sbagliati i questo frame
                correct_position = joint_frame_error_bridge[0]


            coords = np.dot(skeleton_user, rotation_matrix)
            edges = adj

            plt.switch_backend('TkAgg')
            # noinspection PyUnresolvedReferences
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(10, 5))

            pose_ax = fig.add_subplot(1, 1, 1, projection='3d')
            pose_ax.set_title('Prediction')
            if range_ == 0:
                range_ = np.amax(np.abs(skeleton_user))
            pose_ax.set_xlim3d(-range_, range_)
            pose_ax.set_ylim3d(-range_, range_)
            pose_ax.set_zlim3d(-0, range_)
            plt.ylabel("y")
            plt.xlabel("x")
            for i_start in range(0, len(edges)):
                for k in edges[i_start]:
                    if error_flag and k in list_of_errors:
                        color = 'red'
                        s = 55
                    else:
                        color = 'blue'
                        s = 40
                    pose_ax.scatter(coords[i_start][0], coords[i_start][1], coords[i_start][2], c=color, s=s)
                    pose_ax.plot(*zip(coords[i_start], coords[k]), marker='o', markersize=2)

            pose_ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=2)
            cds = np.array(midpoint(coords[0][0], coords[0][1], coords[0][2], coords[1][0], coords[1][1], coords[1][2]))
            cds += ([0, 0, 0.05])
            pose_ax.scatter(cds[0], cds[1], cds[2], c='green', s=300)

            fig.tight_layout()
            if error_flag:
                plt.savefig('img_out_user' + '/' + str(item) + '.png')
                for r in range(0, 4):
                    plt.savefig('img_out_user' + '/' + str(item) + '.' + str(r) + '.png')
            else:
                plt.savefig('img_out_user' + '/' + str(item) + '.png')

    create_video_from_seq('img_out_user', type + '_output_video')



def empty_folders(folder, coords_flag):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    if coords_flag:
        folder = folder + '_coords'
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


def midpoint(x1, y1, z1, x2, y2, z2):
    return (x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2
