import cv2
import os


def create_video_from_seq(image_sequence_folder, output_video_name):
    global height, width
    video_name = 'videos\\out\\'+output_video_name+'.avi'
    images = [img for img in os.listdir(image_sequence_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_sequence_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, 7, (width, height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_sequence_folder, image)))
    cv2.destroyAllWindows()
    video.release()
