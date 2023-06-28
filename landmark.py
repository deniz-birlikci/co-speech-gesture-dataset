class Landmark():
    def __init__(self, cv2_timestamp, calc_timestamp, hand_landmarks, face_landmarks, pose_landmarks):
        self.cv2_timestamp = cv2_timestamp
        self.calc_timestamp = calc_timestamp
        self.hand_landmarks = hand_landmarks
        self.face_landmarks = face_landmarks
        self.pose_landmarks = pose_landmarks
        
class VideoLandmark():
    def __init__(self, video_path, landmarks):
        self.video_path = video_path
        self.landmarks = landmarks