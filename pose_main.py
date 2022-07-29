import mediapipe as mp
import cv2

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For webcam input:
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # per webcam

try:
    #cap = cv2.VideoCapture('videos/c.mp4')
    names = ['videos/c.mp4']
    window_titles = ['c']
    cap = [cv2.VideoCapture(i) for i in names]
except:
    print("Could not open video file")
    raise
#print(cap.grab())

for i in range(0,len(names)):
    if not cap[i].isOpened():
        print("Error opening video stream or file")


def get_skeleton(cap):
    for i, j in enumerate(cap):
        while cap[i].isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' inst ead of 'continue'.
                break
                # continue # per webcam

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break


with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    get_skeleton(cap)

cap.release()
cv2.destroyAllWindows()
