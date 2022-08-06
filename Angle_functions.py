from fastdtw import fastdtw
from math import acos
from utility import *

#####################     ANGLES FUNCTIONS       #######################

#  Calcola l'angolo in radianti tra i vettori u e v


def angle(skeleton, A, B, C):
    u = find_vector(find_coord(skeleton, A), find_coord(skeleton, B))
    v = find_vector(find_coord(skeleton, C), find_coord(skeleton, B))
    cos = np.inner(u, v).item() / (np.linalg.norm(u) * np.linalg.norm(v))
    return acos(cos)


def retrieve_angles(skeleton):
    angles = []
    index = []
    angles.append(angle(skeleton, 0, 1, 2))
    angles.append(angle(skeleton, 6, 4, 2))
    angles.append(angle(skeleton, 0, 2, 4))
    angles.append(angle(skeleton, 1, 0, 3))
    angles.append(angle(skeleton, 5, 3, 1))
    angles.append(angle(skeleton, 7, 5, 3))
    angles.append(angle(skeleton, 3, 1, 9))
    angles.append(angle(skeleton, 2, 0, 8))
    angles.append(angle(skeleton, 1, 9, 8))
    angles.append(angle(skeleton, 0, 8, 9))
    angles.append(angle(skeleton, 1, 9, 11))
    angles.append(angle(skeleton, 0, 8, 10))
    angles.append(angle(skeleton, 9, 11, 13))
    angles.append(angle(skeleton, 8, 10, 12))
    angles.append(angle(skeleton, 9, 8, 10))
    angles.append(angle(skeleton, 8, 9, 11))
    angles.append(angle(skeleton, 11, 13, 15))
    angles.append(angle(skeleton, 10, 12, 14))
    for joint in [b'lsho', b'rsho', b'lelb', b'relb', b'lwri', b'rwri', b'lind', b'rind', b'lhip', b'rhip', b'lkne', b'rkne', b'lheel', b'rhell', b'lfind', b'rfind']:
        index.append(joint)
    angles = np.array(angles)
    return angles, index


def angles_distance(anglesA, anglesB):
    return np.linalg.norm(anglesA - anglesB)


def retrive_angles_PoI_sequences(exercise):
    PoI = euclidean_identify_repetitions(exercise)
    sequences = []
    index_list = []
    for tuple in PoI:
        sequence = []
        temp_index = []
        for frame in range(tuple[0], tuple[1], 5):
            sequence.append(retrieve_angles(get_coords_from_file(exercise, str(frame)))[0])
            temp_index.append(frame)
        index_list.append(temp_index)
        sequences.append(sequence)
    return sequences, index_list


def repetitions_angles_distance(exercise):
    user_sequences, user_index = retrive_angles_PoI_sequences(exercise + '_tester')
    try:
        trainer_sequences, trainer_index = retrive_angles_PoI_sequences(exercise + '_trainer')
    except FileNotFoundError:
        print("\nAttenzione!:\nhai avviato l'analisi di un tester senza prima riempire il dataser trainer!\nAnalizza prima un video trainer"
              " e non dimenticare di settare il flag 'trainer' = true nel file di configurazione!")
        exit()
    distances = []
    i = 0
    if len(user_sequences) > len(trainer_sequences):
        while i < len(trainer_sequences):
            distances.append((i, i, sequence_angles_distance(trainer_sequences[i], user_sequences[i])))
            i += 1
        while i < len(user_sequences):
            distances.append((0, i, sequence_angles_distance(trainer_sequences[0], user_sequences[i])))
            i += 1
    else:
        while i < len(user_sequences):
            distances.append((i, i, sequence_angles_distance(trainer_sequences[i], user_sequences[i])))
            i += 1
        while i < len(trainer_sequences):
            distances.append((i, 0, sequence_angles_distance(trainer_sequences[i], user_sequences[0])))
            i += 1
    return distances, trainer_index, user_index


def sequence_angles_distance(S1, S2):
    sequence_dist = []
    dist, path = fastdtw(S1, S2, dist=angles_distance)
    for idx in path:
        sequence_dist.append(angles_distance(S1[idx[0]], S2[idx[1]]))
    return dist / len(path), path, sequence_dist


def identify_angles_errors(exercise, repetition_distance, joint_thr_multiplier=1.5):
    frames_number = len([name for name in os.listdir(exercise + '_tester_coords')])
    joints_number = 15
    error_frame_list, repetition_error_list = identify_frame_errors(repetition_distance)
    if len(error_frame_list) == 0:
        return
    joint_error_counter = np.zeros(
        shape=(np.max(repetition_error_list) + 1, 18))
    joint_frame_error_bridge = []
    for j in range(len(error_frame_list)):
        frame_couple = error_frame_list[j]
        user_coordinates = get_coords_from_file(exercise + '_tester', str(frame_couple[1]))
        user_angles, user_angles_index = retrieve_angles(user_coordinates)

        trainer_coordinates = get_coords_from_file(exercise + '_trainer', str(frame_couple[0]))
        trainer_angles, trainer_angles_index = retrieve_angles(trainer_coordinates)

        joint_distances = []
        for i in range(len(user_angles)):
            joint_distances.append((angles_distance(user_angles[i], trainer_angles[i]), i))
        thr = np.mean([tup[0] for tup in joint_distances]) * joint_thr_multiplier
        top_tier_distances = sorted(joint_distances, key=lambda tup: tup[0], reverse=True)

        error_points = []
        for tuple in top_tier_distances:
            if tuple[0] > thr:
                coords_idx = from_angle_to_joint_index(tuple[1])
                joint_frame_error_bridge.append([error_frame_list[j][0], error_frame_list[j][1], coords_idx]) #0: trainer frame, 1:user frame, 2:joint sbagliato
                error_points.append(from_angle_to_joint_index(tuple[1]))
                joint_error_counter[repetition_error_list[j]][coords_idx] += 1

    MCE = np.argmax(np.sum(joint_error_counter, axis=0))
    print("L'articolazione che è stata maggiormente sbagliata nel corso dell'esercizio " + exercise + " è: " + str(
        from_jointindex_to_jointname(MCE))[1:] + " (# di frame con joint mal posizionato: " + str(
        int(np.sum(joint_error_counter, axis=0)[MCE])) + ")\tSuccesso esercizio: " + str(
        round((1 - (np.sum(joint_error_counter)) / (frames_number * joints_number)) * 100, 2)) + "%")
    for i in range(joint_error_counter.shape[0]):
        MCE = np.argmax(joint_error_counter[i])
        print("L'articolazione che è stata maggiormente sbagliata nel corso della ripetizione " + str(i) + " è: " + str(
            from_jointindex_to_jointname(MCE))[1:] + " (# di frame joint mal posizionato: " + str(
            int(joint_error_counter[i][MCE])) + ")\tSuccesso ripetizione: " + str(round((1 - (
            np.sum(joint_error_counter[i])) / ((frames_number / len(
            np.unique(repetition_error_list))) * joints_number)) * 100, 2)) + "%")
    return joint_frame_error_bridge
