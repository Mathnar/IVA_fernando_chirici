from ai import cs
from cmath import acos, cos, sin
from fastdtw import fastdtw

from utility import *

# Converte coordinate cartesiane in coordinate sferiche
def carth_to_sphere(x, y, z):
    r, theta, phi = cs.cart2sp(x=x, y=y, z=z)
    return np.array([r, phi, theta])


# Aggiunge coordinate sferiche a skeleton
def skeleton_to_sphere(skeleton):
    skeleton_sp = list([])
    for i in range(len(skeleton)):
        skeleton_sp.append(carth_to_sphere(skeleton[i][0], skeleton[i][1], skeleton[i][2]))
    skeleton_sp = np.array(skeleton_sp)
    return skeleton_sp


# Distanza tra due punti definita autonomamente
def spheric_distance(A, B):
    rA = A[0]
    rB = B[0]
    phiA = A[1]
    phiB = B[1]
    thetaA = A[2]
    thetaB = B[2]
    if phiA == phiB and thetaA == thetaB:
        diff_an = 0
    else:
        diff_an = acos(cos(phiA - phiB) * cos(thetaA) * cos(thetaB) + sin(thetaA) * sin(thetaB))
    diff_r = np.linalg.norm(rA - rB)
    # [abs(rA-rB), abs(phiA-phiB), abs(tethaA-tethaB)]
    return diff_r + diff_an


# Distanza tra due skeleton
def spheric_skeleton_distance(skeletonA, skeletonB):
    diff = []
    for i in range(skeletonA.shape[0]):
        diff.append(spheric_distance(skeletonA[i], skeletonB[i]))
    return np.linalg.norm(diff)


def retrieve_sphere_PoI_sequences(exercise):
    PoI = euclidean_identify_repetitions(exercise)
    sequences = []
    index_list = []
    for tuple in PoI:
        sequence = []
        temp_index = []
        for frame in range(tuple[0], tuple[1], 5):
            skn = skeleton_to_sphere(get_coords_from_file(exercise, str(frame)))
            sequence.append(skn)
            temp_index.append(frame)
        index_list.append(temp_index)
        sequences.append(sequence)
    return sequences, index_list


def sequence_sphere_distance(S1, S2):
    sequence_dist = []
    dist, path = fastdtw(S1, S2, dist=spheric_skeleton_distance)
    for idx in path:
        sequence_dist.append(spheric_skeleton_distance(S1[idx[0]], S2[idx[1]]))
    return dist / len(path), path, sequence_dist


def repetitions_combined_distance(exercise):
    user_sequences, user_index = retrieve_sphere_PoI_sequences(exercise + '_tester')
    trainer_sequences, trainer_index = retrieve_sphere_PoI_sequences(exercise + '_trainer')
    distances = []
    i = 0
    if len(user_sequences) > len(trainer_sequences):
        while i < len(trainer_sequences):
            distances.append((i, i, sequence_sphere_distance(trainer_sequences[i], user_sequences[i])))
            i += 1
        while i < len(user_sequences):
            distances.append((0, i, sequence_sphere_distance(trainer_sequences[0], user_sequences[i])))
            i += 1
    else:
        while i < len(user_sequences):
            distances.append((i, i, sequence_sphere_distance(trainer_sequences[i], user_sequences[i])))
            i += 1
        while i < len(trainer_sequences):
            distances.append((i, 0, sequence_sphere_distance(trainer_sequences[i], user_sequences[0])))
            i += 1
    return distances, trainer_index, user_index


def identify_combined_errors(exercise, repetition_distance, joint_thr_multiplier=1.0, frame_thr_multiplier=1.0,
                             visualize_errors_flag=True):
    frames_number = len([name for name in os.listdir(exercise + '_tester_coords')])
    joints_number = 15
    error_frame_list, repetition_error_list = identify_frame_errors(repetition_distance, frame_thr_multiplier)
    if len(error_frame_list) == 0:
        return
    joint_error_counter = np.zeros(
        shape=(np.max(repetition_error_list) + 1, 18))

    joint_frame_error_bridge = []
    for j in range(len(error_frame_list)):
        frame_couple = error_frame_list[j]
        user_coordinates = get_coords_from_file(exercise + '_tester', str(frame_couple[1]))

        trainer_coordinates = get_coords_from_file(exercise + '_trainer', str(frame_couple[0]))
        joint_distances = []
        for i in range(len(user_coordinates)):
            joint_distances.append((spheric_distance(user_coordinates[i], trainer_coordinates[i]), i))
        thr = np.mean([tup[0] for tup in joint_distances]) * joint_thr_multiplier
        top_tier_distances = sorted(joint_distances, key=lambda tup: tup[0], reverse=True)
        error_points = []
        for tuple in top_tier_distances:
            if tuple[0] > thr:
                coords_idx = tuple[1]
                joint_frame_error_bridge.append([error_frame_list[j], coords_idx]) #0: trainer frame, 1:user frame, 2:joint sbagliato
                error_points.append(tuple[1])
                joint_error_counter[repetition_error_list[j]][coords_idx] += 1
    MCE = np.argmax(np.sum(joint_error_counter, axis=0))
    print("L'articolazione che è stata maggiormente sbagliata nel corso dell'esercizio " + exercise + " è: " + str(
        from_jointindex_to_jointname(MCE)) + " (" + str(
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
