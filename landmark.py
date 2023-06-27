class Landmark():
    def __init__(self, timestamp, hand_landmarks, face_landmarks, pose_landmarks):
        self.timestamp = timestamp
        self.hand_landmarks = hand_landmarks
        self.face_landmarks = face_landmarks
        self.pose_landmarks = pose_landmarks
        
class VideoLandmark():
    def __init__(self, video_path, landmarks):
        self.video_path = video_path
        self.landmarks = landmarks