from pose_extractor_2 import *
from Euclidean_functions import *
from Angle_functions import *
from Combined_functions import *

config = json.load(open('config.json'))



TRAINER_FLAG = False if config['trainer'] == 'false' else True
if TRAINER_FLAG:
    subject = 'trainer'
else:
    subject = 'tester'
RESIZE = True
EXTRACTION_FLAG = True if config['extraction_flag'] == 'true' else False
EUCLIDEAN = config['EUCLIDEAN']
ANGLE = config['ANGULAR']
COMBINED = config['COMBINED']
POST_PROCESSING = True if config['trainer'] == 'false' else False
EXERCISE = config['exercise']
video_path = [config['video_path']]
video_name = [video_path[0].split('/')[1]]

WORKING_FOLDER = EXERCISE + '_' + subject
adj = [[1, 2], [0, 3], [0, 4], [1, 5], [2, 6], [3, 7], [4], [5], [0, 9, 10], [1, 8, 11], [8, 12], [9, 13],
       [10, 14], [11, 15], [12], [13]]


def get_video_size(cap):
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(length)
    return length


def create_dir(EXERCISE, subject):
    try:
        os.mkdir(EXERCISE + '_' + subject)
        os.mkdir(EXERCISE + '_' + subject + '_coords')
    except FileExistsError:
        print('\nLa cartella per questo esercizio esiste già e verrà sovrascritta ')


if __name__ == '__main__':
    t_start = time.time()
    if EXTRACTION_FLAG:
        create_dir(EXERCISE, subject)
        range_ = 0
        empty_folders(WORKING_FOLDER, True)
        cap, taller = get_cap(video_path)

        frames = [None] * len(video_path)
        gray = [None] * len(video_path)
        ret = [None] * len(video_path)

        with mp_pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                enable_segmentation=True,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7) as pose:
            iterator = 0

            leng = get_video_size(cap[0])
            t_proc = time.time()
            print('\nDurata del pre processing delle informazioni: ', t_proc-t_start, 's')
            while iterator < leng and iterator < 500:
                iterator, range_ = skeleton_extraction(pose, cap, iterator, ret, range_, frames, video_name, WORKING_FOLDER, adj, taller)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            for c in cap:
                if c is not None:
                    c.release()
            cv2.destroyAllWindows()

    t_post_p = time.time()
    if TRAINER_FLAG:
        print("\nAnalisi trainer completata! \nAvvia un'analisi tester settando dal config (1:si - 0:no) quale tipo di elaborazione vuoi effettuare!")
    if EXTRACTION_FLAG and not TRAINER_FLAG:
        print("\nDurata dell'estrazione degli skeletons: ", t_post_p-t_proc, "Tempo dall'inizio: ", t_post_p-t_start, 's')
    if POST_PROCESSING:
        print('\nInizio post processing')
        vis_err = True
        joint_th = pose_th = 1
        if EUCLIDEAN:
            print('\nSTART_EUCLIDEAN')
            rep_distance = repetitions_euclidean_distance(EXERCISE)
            joint_frame_error_bridge = identify_euclidean_errors(EXERCISE, rep_distance, joint_th, pose_th, vis_err)
            print("\nAttendere qualche secondo! E' in corso la generazione del video contenente tutte le informazioni di post processing."
                  "'\nPotrebbe volerci un po!")
            error_frame_higlight(joint_frame_error_bridge, EXERCISE, adj, 'euclidean')
        if ANGLE:
            print('\nSTART_ANGLE')
            rep_distance = repetitions_angles_distance(EXERCISE)
            joint_frame_error_bridge = identify_angles_errors(EXERCISE, rep_distance, joint_th, pose_th, vis_err)
            print("\nAttendere qualche secondo! E' in corso la generazione del video contenente tutte le informazioni di post processing."
                  "'\nPotrebbe volerci un po!")
            error_frame_higlight(joint_frame_error_bridge, EXERCISE, adj, 'angular')
        if COMBINED:
            print('\nSTART_COMBINED')
            rep_distance = repetitions_combined_distance(EXERCISE)
            joint_frame_error_bridge = identify_combined_errors(EXERCISE, rep_distance, joint_th, pose_th, vis_err)
            print("\nAttendere qualche secondo! E' in corso la generazione del video contenente tutte le informazioni di post processing."
                  "\nPotrebbe volerci un po!")
            error_frame_higlight(joint_frame_error_bridge, EXERCISE, adj, 'combined')
    t_end = time.time()
    print("Tempo dall'inizio: ", t_end-t_start, 's')
    if not TRAINER_FLAG:
        print("\nDurata del post processing: ", t_end - t_post_p, 's')







