
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import os.path
from utility import *


def retrive_euclidean_PoI_sequences(exercise):
    PoI = euclidean_identify_repetitions(exercise)
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
    try:
        trainer_sequences, trainer_index = retrive_euclidean_PoI_sequences(exercise + '_trainer')
    except FileNotFoundError:
        print("\nAttenzione!:\nhai avviato l'analisi di un tester senza prima riempire il dataser trainer!\nAnalizza prima un video trainer"
              " e non dimenticare di settare il flag 'trainer' = true nel file di configurazione!")
        exit()
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


def identify_euclidean_errors(exercise, repetition_distance, joint_thr_multiplier=1.5):
    frames_number = len([name for name in os.listdir(exercise + '_tester_coords')])
    joints_number = 15
    error_frame_list, repetition_error_list = identify_frame_errors(repetition_distance)
    if len(error_frame_list) == 0:
        return
    joint_error_counter = np.zeros(shape=(np.max(repetition_error_list) + 1, get_coords_from_file(exercise + '_tester', 0).shape[
        0]))
    joint_frame_error_bridge = []
    for j in range(len(error_frame_list)):
        frame_couple = error_frame_list[j]
        user_coordinates = get_coords_from_file(exercise + '_tester', str(frame_couple[1]))

        trainer_coordinates = get_coords_from_file(exercise + '_trainer', str(frame_couple[0]))
        joint_distances = []
        for i in range(len(user_coordinates)):
            joint_distances.append((np.linalg.norm(user_coordinates[i] - trainer_coordinates[i]),
                                    i))  # memorizza distanza e indice coordinata
        thr = np.mean([tup[0] for tup in joint_distances]) * joint_thr_multiplier
        distances = sorted(joint_distances, key=lambda tup: tup[0], reverse=True)
        error_points = []
        for tuple in distances:
            if tuple[0] > thr:
                coords_idx = tuple[1]
                joint_frame_error_bridge.append([error_frame_list[j][0], error_frame_list[j][1], coords_idx]) #0: trainer frame, 1:user frame, 2:joint sbagliato
                error_points.append(tuple[1])
                joint_error_counter[repetition_error_list[j]][coords_idx] += 1


    MCE = np.argmax(np.sum(joint_error_counter, axis=0))

    print("L'articolazione che è stata maggiormente sbagliata nel corso dell'esercizio " + exercise + " è: " + str(
        from_jointindex_to_jointname(MCE))[1:] + " (# di frame con joint mal posizionato: " + str(
        int(np.sum(joint_error_counter, axis=0)[MCE])) + ")\tSuccesso esercizio: " + str(
        round((1 - (np.sum(joint_error_counter)) / (frames_number * joints_number)) * 100, 2)) + "%")
    for i in range(joint_error_counter.shape[0]):
        MCE = np.argmax(joint_error_counter[i])
        print("L'articolazione che è stata maggiormente sbagliata nel corso della ripetizione " + str(i) + " è: " + str(
            from_jointindex_to_jointname(MCE)) + " (" + str(
            int(joint_error_counter[i][MCE])) + ")\tSuccesso ripetizione: " + str(round((1 - (
            np.sum(joint_error_counter[i])) / ((frames_number / len(
            np.unique(repetition_error_list))) * joints_number)) * 100, 2)) + "%")
    return joint_frame_error_bridge

