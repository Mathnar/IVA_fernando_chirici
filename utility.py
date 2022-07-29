import os

import numpy as np


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
        print('\nframe ', frame, ' / ', (n_frames*5-5))
        # carico tutte le altre coord e cerco distanza dal primio (ref) che si suppone sia la posizione iniziale
    #exit()

    print('\ndist ', distances)

    candidate_PoI = []
    PoI = [(0, 0)]
    thr = np.mean(np.array(distances))
    for i in range(len(distances)):
        if distances[i] <= thr:
            candidate_PoI.append(i * 5)
    print('\ncandidate_PoI', candidate_PoI)
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
    print('\nPOI ', PoI)
    PoI.pop(0)
    print('\nPOI ', PoI, PoI[0][1])
    if PoI[0][1] <= 30 and len(PoI)>1:  # vincolo che evita che il primo punto di interesse sia preso entro un secondo dall'inizio dell'esercizio
        PoI.pop(0)
        print('\nPOI ', PoI)

        PoI[0] = (0, PoI[0][1])
        print('\nòò ', PoI)
    if len(PoI) > 1:
        PoI.pop(len(PoI) - 1)
    previous_el = int(PoI[len(PoI) - 1][1])
    PoI.append((previous_el, n_frames * 5 - 5))
    return PoI



def identify_frame_errors(repetition_distance, thr_multiplier=1.0):
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
                user_frame = user_index[user_repetition_num][path[i][1]]
                trainer_frame = trainer_index[trainer_repetition_num][path[i][0]]
                error_list.append((trainer_frame, user_frame))
                repetition_list.append(user_repetition_num)
    print("Errori commessi: " + str(len(error_list)))
    print("Nelle coppie di frame: " + str(error_list))
    return error_list, repetition_list
