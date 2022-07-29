from math import radians

from cmath import cos, sin

import os

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import os, os.path
from utility import *




def retrive_euclidean_PoI_sequences(exercise):
    PoI = euclidean_identify_repetitions(exercise)
    print('\nnPoi: ', PoI)

    sequences = []
    index_list = []
    for tuple in PoI:
        sequence = []
        temp_index = []
        for frame in range(tuple[0], tuple[1], 5):
            sequence.append(get_coords_from_file(exercise, str(frame)))
            temp_index.append(frame)
        index_list.append(temp_index)
        sequences.append(sequence)
    print('\nseq ', sequences, '\nind ', index_list)
    return sequences, index_list


def sequence_euclidean_distance(S1, S2):
    S1 = np.reshape(S1, newshape=(len(S1), S1[0].shape[0] * S1[0].shape[1]))
    S2 = np.reshape(S2, newshape=(len(S2), S2[0].shape[0] * S2[0].shape[1]))
    sequence_dist = []
    dist, path = fastdtw(S1, S2, dist=euclidean)
    for idx in path:
        sequence_dist.append(np.linalg.norm(S1[idx[0]] - S2[idx[1]]))
    return dist / len(path), path, sequence_dist


def repetitions_euclidean_distance(exercise):
    user_sequences, user_index = retrive_euclidean_PoI_sequences(exercise + '_tester')
    trainer_sequences, trainer_index = retrive_euclidean_PoI_sequences(exercise + '_trainer')
    distances = []
    i = 0
    if len(user_sequences) > len(trainer_sequences):
        while i < len(trainer_sequences):
            distances.append((i, i, sequence_euclidean_distance(trainer_sequences[i], user_sequences[i])))
            i += 1
        while i < len(user_sequences):
            distances.append((0, i, sequence_euclidean_distance(trainer_sequences[0], user_sequences[i])))
            i += 1
    else:
        while i < len(user_sequences):
            distances.append((i, i, sequence_euclidean_distance(trainer_sequences[i], user_sequences[i])))
            i += 1
        while i < len(trainer_sequences):
            distances.append((i, 0, sequence_euclidean_distance(trainer_sequences[i], user_sequences[0])))
            i += 1
    return distances, trainer_index, user_index


def identify_euclidean_errors(exercise, repetition_distance, joint_thr_multiplier=1.0, frame_thr_multiplier=1.0,
                              visualize_errors_flag=True):
    frames_number = len([name for name in os.listdir(exercise + '_tester_coords')])
    joints_number = 15
    error_frame_list, repetition_error_list = identify_frame_errors(repetition_distance, frame_thr_multiplier)
    if len(error_frame_list) == 0:
        return
    print('\neu repetition_error_list ', repetition_error_list)
    # exit()
    joint_error_counter = np.zeros(shape=(np.max(repetition_error_list) + 1, get_coords_from_file(exercise + '_tester', 0).shape[
        0]))
    print('\nerror frame list ', error_frame_list)

    for j in range(len(error_frame_list)):
        frame_couple = error_frame_list[j]
        user_image = "squats_tester\\" + str(frame_couple[1]) + ".png"
        user_coordinates = get_coords_from_file('squats_tester', str(frame_couple[1]))

        trainer_image = "squats_trainer\\" + str(frame_couple[0]) + ".png"
        # if frame_couple[1] > len([name for name in os.listdir(exercise + '_trainer_coords')]):
        #     pointer = frame_couple[1]-()
        print('\n__iter: ', frame_couple[1])
        trainer_coordinates = get_coords_from_file('squats_trainer', str(frame_couple[0]))

        joint_distances = []
        for i in range(len(user_coordinates)):
            joint_distances.append((np.linalg.norm(user_coordinates[i] - trainer_coordinates[i]),
                                    i))  # memorizza distanza e indice coordinata
        thr = np.mean([tup[0] for tup in joint_distances]) * joint_thr_multiplier
        distances = sorted(joint_distances, key=lambda tup: tup[0], reverse=True)
        error_points = []
        print('\ndistances ', distances)
        # exit()
        for tuple in distances:
            if tuple[0] > thr:
                print('\ntuple[1] ', tuple[1])
                print('\nuser_coordinates[tuple[1]] ', user_coordinates[tuple[1]])
                coords_idx = tuple[1]#todo test
                print('\n np where ', np.where(user_coordinates[tuple[1]] == user_coordinates))
                error_points.append(tuple[1])
                print('\njoint_error_counter', joint_error_counter)
                print('\nrepetition_error_list[j]', repetition_error_list[j])
                print('\ncoords_idx', coords_idx)
                joint_error_counter[repetition_error_list[j]][coords_idx] += 1
        # if visualize_errors_flag:
        #     visualize_errors(trainer_image, user_image,
        #                      upper_body(user_sk, two_dim=True)[error_points], frame_couple)

    MCE = np.argmax(np.sum(joint_error_counter, axis=0))
    print("L'articolazione che è stata maggiormente sbagliata nel corso dell'esercizio " + exercise[:exercise.find(
        "_")] + " è: " +  'un altro segreto' + " (" + str(
        int(np.sum(joint_error_counter, axis=0)[MCE])) + ")\tSuccesso esercizio: " + str(
        round((1 - (np.sum(joint_error_counter)) / (frames_number * joints_number)) * 100, 2)) + "%")
    for i in range(joint_error_counter.shape[0]):
        MCE = np.argmax(joint_error_counter[i])
        print("L'articolazione che è stata maggiormente sbagliata nel corso della ripetizione " + str(i) + " è: " + 'un altro segreto '+ " (" + str(
            int(joint_error_counter[i][MCE])) + ")\tSuccesso ripetizione: " + str(round((1 - (
            np.sum(joint_error_counter[i])) / ((frames_number / len(
            np.unique(repetition_error_list))) * joints_number)) * 100, 2)) + "%")


#
# def visualize_errors(trainer_sk, user_sk, trainer_image, user_image, error_points, error_2d_points, frame_couple):
#     if len(error_points) == 0:
#         print("Joint Errors not detected in frame " + str(frame_couple[1]) + "! (Try to set a lower threshold.)")
#         return
#
#     rotation_matrix = np.dot(np.dot([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]), [[cos(radians(-10)), 0, sin(radians(-10))], [0, 1, 0], [-sin(radians(-10)), 0, cos(radians(-10))]])
#
#     coords1 = np.dot(trainer_sk["joint_coordinates"], rotation_matrix)
#     # coords1 = trainer_sk["joint_coordinates"]
#     edges1 = trainer_sk["edges"]
#     coords2 = np.dot(user_sk["joint_coordinates"], rotation_matrix)
#     # coords2 = user_sk["joint_coordinates"]
#     edges2 = user_sk["edges"]
#     error_points = np.dot(np.array(error_points), rotation_matrix)
#     # error_points = np.array(error_points)
#
#     matplotlib.use('Qt5Agg')
#     # noinspection PyUnresolvedReferences
#     #from mpl_toolkits.mplot3d import Axes3D
#
#     fig = plt.figure(figsize=(15, 15))
#
#     trainer_image = image_to_numpy(trainer_image)[0]
#     image_ax = fig.add_subplot(2, 2, 1)
#     image_ax.set_title('\nTRAINER')
#     image_ax.imshow(trainer_image)
#
#     user_image = image_to_numpy(user_image)[0]
#     image_ax = fig.add_subplot(2, 2, 2)
#     image_ax.set_title('\nUSER')
#     color = [[1, 0, 0]]
#     i = 1
#     for error in error_2d_points:
#         image_ax.scatter(error[0], error[1], marker='X', c=color, s=100)
#         color[0][1] += 1 / (2 ** i)
#         i += 1
#
#     image_ax.imshow(user_image)
#
#     pose_ax = fig.add_subplot(2, 2, 3, projection='3d')
#     pose_ax.set_title('\nFrame:' + str(frame_couple[0]))
#     range_ = np.amax(np.abs(coords1))
#     pose_ax.set_xlim3d(-range_, range_)
#     pose_ax.set_ylim3d(-range_, range_)
#     pose_ax.set_zlim3d(-range_, range_)
#     pose_ax.set_ylabel(ylabel='y')
#     pose_ax.set_xlabel("x")
#     pose_ax.set_zlabel("z")
#
#     for i_start, i_end in edges1:
#         pose_ax.plot(*zip(coords1[i_start], coords1[i_end]), marker='o', markersize=2)
#
#     pose_ax.scatter(coords1[:, 0], coords1[:, 1], coords1[:, 2], c='#0000ff', s=2)
#
#     pose_ax = fig.add_subplot(2, 2, 4, projection='3d')
#     pose_ax.set_title('\nFrame:' + str(frame_couple[1]))
#     range_ = np.amax(np.abs(coords2))
#     pose_ax.set_xlim3d(-range_, range_)
#     pose_ax.set_ylim3d(-range_, range_)
#     pose_ax.set_zlim3d(-range_, range_)
#     pose_ax.set_ylabel(ylabel='y')
#     pose_ax.set_xlabel("x")
#     pose_ax.set_zlabel("z")
#
#     for i_start, i_end in edges2:
#         pose_ax.plot(*zip(coords2[i_start], coords2[i_end]), marker='o', markersize=2)
#
#     pose_ax.scatter(coords2[:, 0], coords2[:, 1], coords2[:, 2], c='#0000ff', s=2)
#     color = [[1, 0, 0]]
#     i = 1
#     for error in error_points:
#         pose_ax.scatter(error[0], error[1], error[2], marker='X', c=color, s=50)
#         color[0][1] += 1 / (2 ** i)
#         i += 1
#     fig.tight_layout(pad=0.1, h_pad=0.01, w_pad=0.01, rect=(0, 0, 1, 1))
#
#     plt.show()
#     return
