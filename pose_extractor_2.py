import mediapipe as mp
from utility import *
from PIL import Image
from matplotlib import pyplot as plt

RESIZE = False

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst



def skeleton_extraction(pose, cap, iterator, ret, range_, frames, window_titles, SAVE_FOLDER, adj, taller):
    for i, c in enumerate(cap):
        if c is not None:
            ret[i], frames[i] = c.read()
            if iterator % 5 != 0:
                iterator += 1
                return iterator, range_

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

            # Rimuovo landmark che non mi interessano
            remove_arr = remove_landmarks(pose_land0)
            i, xyz_array = undesired_landmarks_removal(i, remove_arr, xyz_array)

            if len(xyz_array) < 16:
                print('\nFrame con troppi pochi landmarks')
                break

            # START NORMALIZATION
            # Estrazione valore medio e normalizzazione vettore dei joints (centramento sul baricentro e normalizzazione)
            xyz_array = normalize_centering_and_size(pose_land0, xyz_array)
            xyz_array = normalize_framing(pose_land0, xyz_array)
            iterator, range_ = visualize_or_save_skeleton(xyz_array, iterator, adj, SAVE_FOLDER, range_, True)
            mp_drawing.draw_landmarks(
                f,
                pose_land0_relative,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            if taller:
                cv2.namedWindow(str(window_titles[0]), cv2.WINDOW_NORMAL)
                imS = cv2.resize(cv2.flip(f, 1), (540, 960))  # Resize image
                cv2.imshow(window_titles[0], imS)  # Show image
            else:
                cv2.namedWindow(str(window_titles[0]), cv2.WINDOW_NORMAL)
                imS = cv2.resize(cv2.flip(f, 1), (960, 540))  # Resize image
                cv2.imshow(window_titles[0], imS)  # Show image
            break
    return iterator, range_


def extract_array_from_landmarks(pose_land0, xyz_array):
    str_pose = str(pose_land0)
    # debug print('\nPose_landmarks as string: ', str_pose)
    landmark_coord = str_pose.split('landmark')
    landmark_coord = landmark_coord[1:]
    # Estraggo le xyz e metto in un array 3d
    # debug print('\nGet only landmarks: ', landmark_coord)
    for j in range(0, len(landmark_coord)):
        # print('\n',landmark_coord[i],'\n')
        xyz_array = get_xyz(xyz_array, landmark_coord[j])
        # print('xyz_array ', xyz_array)
    return xyz_array


def undesired_landmarks_removal(i, remove_arr, xyz_array):
    while i < len(xyz_array):
        if xyz_array[i] in remove_arr:
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
    left_xyz = np.array([left_ank_coo.x, left_ank_coo.y, left_ank_coo.z])
    right_xyz = np.array([right_ank_coo.x, right_ank_coo.y, right_ank_coo.z])
    pelv = midpoint(left_ank_coo.x, left_ank_coo.y, left_ank_coo.z, right_ank_coo.x, right_ank_coo.y, right_ank_coo.z)
    center_points = [left_xyz, right_xyz, pelv]
    # Estrazione vettore medio fianchi e torso
    mean_vector = np.zeros(dtype='float64', shape=(1, 3))  # [0 0 0]
    for point in center_points:
        mean_vector += point
    mean_vector = mean_vector / len(center_points)
    mean_vector = np.reshape(mean_vector, (3,))

    for joint in xyz_array:
        joint -= mean_vector
    xyz_array = xyz_array / np.linalg.norm(xyz_array)
    return xyz_array


def normalize_framing(pose_land0, xyz_array):
    lcla = pose_land0.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    Jhl = np.transpose(np.array([lcla.x, lcla.y, lcla.z]))
    rcla = pose_land0.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    Jt = np.transpose(np.array([rcla.x, rcla.y, rcla.z]))
    Jhl = Jhl / np.linalg.norm(Jhl)
    norm2 = np.linalg.norm(Jhl)

    tras = np.transpose(Jhl)
    temp = np.dot((tras / norm2), Jt).item()
    Jhl_ort = Jt - (temp * Jhl)

    Jhl_ort = Jhl_ort / np.linalg.norm(Jhl_ort)
    cross_product_vector = np.cross(Jhl, Jhl_ort, axis=0)  # added reshape qua
    cross_product_vector = cross_product_vector / np.linalg.norm(cross_product_vector)

    M = np.concatenate((Jhl.reshape(-1, 1), Jhl_ort.reshape(-1, 1), cross_product_vector.reshape(-1, 1)), axis=1)
    x_tilde = np.transpose(xyz_array)
    x_tilde = np.dot(np.transpose(M), x_tilde)
    xyz_array = np.transpose(x_tilde)
    return xyz_array


def visualize_or_save_skeleton(skeleton, iterator, adj, SAVE_FOLDER, range_, save=False):
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

    fig = plt.figure(figsize=(10, 5))

    pose_ax = fig.add_subplot(1, 1, 1, projection='3d')
    pose_ax.set_title('Prediction')
    if range_ == 0:
        range_ = np.amax(np.abs(skeleton))
    pose_ax.set_xlim3d(-range_, range_)
    pose_ax.set_ylim3d(-range_, range_)
    pose_ax.set_zlim3d(-0, range_)
    plt.ylabel("y")
    plt.xlabel("x")
    for i_start in range(0, len(edges)):

        for k in edges[i_start]:
            pose_ax.scatter(coords[i_start][0], coords[i_start][1], coords[i_start][2], c='red', s=40)
            pose_ax.plot(*zip(coords[i_start], coords[k]), marker='o', markersize=2)

    pose_ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=2)
    cds = np.array(midpoint(coords[0][0], coords[0][1], coords[0][2], coords[1][0], coords[1][1], coords[1][2]))
    cds += ([0, 0, 0.05])
    pose_ax.scatter(cds[0], cds[1], cds[2], c='green', s=300)

    fig.tight_layout()
    if save:
        plt.savefig(SAVE_FOLDER + '/' + str(iterator) + '.png')
        with open(SAVE_FOLDER + '_coords/' + str(iterator) + '.txt', 'w') as f:
            f.write(str(skeleton))
    elif not save:
        plt.show()
    iterator += 1

    return iterator, range_


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


def get_cap(video_path):
    for i in range(0, len(video_path)):
        cap = cv2.VideoCapture(video_path[i])
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        if height < width:
            taller = False
        else:
            taller = True
    cap = [cv2.VideoCapture(i) for i in video_path]
    return cap, taller
