import os
import cv2
import pickle
import mediapipe as mp
from tqdm import tqdm
from dotenv import load_dotenv
from landmark import Landmark, VideoLandmark
import multiprocessing
import time

# Load environment variables
load_dotenv()
# BASE_DIRECTORY = os.getenv('BASE_DIRECTORY')
BASE_DIRECTORY = "/work/mbirlikc/data/"

INPUT_DIRECTORY = os.path.join(BASE_DIRECTORY, r'Interviews')
OUTPUT_DIRECTORY = os.path.join(BASE_DIRECTORY, r'Processed Interviews')
MODELS_DIRECTORY = os.path.join(BASE_DIRECTORY, r'Models')

def get_hand_landmarker():
    mediapipe_model_path = os.path.join(MODELS_DIRECTORY, "hand_landmarker.task")

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a hand landmarker instance with the video mode:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=mediapipe_model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2
    )

    return HandLandmarker.create_from_options(options)

def get_face_landmarker(speaker_count=1):
    mediapipe_model_path = os.path.join(MODELS_DIRECTORY, 'face_landmarker.task')

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a face landmarker instance with the video mode:
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=mediapipe_model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=speaker_count,
        output_face_blendshapes=True
    )

    return FaceLandmarker.create_from_options(options)

def get_pose_landmarker(speaker_count=1):
    mediapipe_model_path = os.path.join(MODELS_DIRECTORY, 'pose_landmarker_lite.task')

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a pose landmarker instance with the video mode:
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=mediapipe_model_path,
                                #  delegate=mp.tasks.BaseOptions.Delegate.GPU
                                ),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=speaker_count,
    )

    return PoseLandmarker.create_from_options(options)

def landmark_video(video_path, speaker_count=1, tqdm_enabled=False, tqdm_position=0):
    landmarks = []
    
    with (get_hand_landmarker() as hand_landmarker,
         get_face_landmarker(speaker_count=speaker_count) as face_landmarker,
         get_pose_landmarker(speaker_count=speaker_count) as pose_landmarker,
    ):
        cap = cv2.VideoCapture(video_path)
        print("Successful capture for video_path:", video_path)

        # Load the frame rate of the video using OpenCV’s CAP_PROP_FPS
        # You’ll need it to calculate the timestamp for each frame.
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Get the total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate the time interval (in milliseconds) between each frame
        frame_interval = int(1000 / fps)

        # Initialize the timestamp variable
        timestamp_ms = 0

        # Loop through each frame in the video using VideoCapture#read()
        iterator = range(total_frames)
        if tqdm_enabled:
            video_id = video_path.split(".mp4")[0].split("@")[-1]
            iterator = tqdm(iterator, desc=f'Processing {tqdm_position}/{video_id}', unit='frame', position=tqdm_position)
        for _ in iterator:
            # Read each frame from the video using VideoCapture's read() method.
            ret, frame = cap.read()

            if not ret:
                break

            # Convert the frame received from OpenCV to a MediaPipe's Image object.
            mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            # Update the timestamp
            timestamp_ms += frame_interval

            # Get the current timestamp in milliseconds
            actual_timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            actual_frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)

            # Run each type of landmark
            hand_landmarker_result = hand_landmarker.detect_for_video(mp_frame, timestamp_ms)
            pose_landmarker_result = pose_landmarker.detect_for_video(mp_frame, timestamp_ms)
            face_landmarker_result = face_landmarker.detect_for_video(mp_frame, timestamp_ms)

            # Save each type of landmark output
            cur_landmark = Landmark(actual_timestamp_ms, hand_landmarker_result, face_landmarker_result, pose_landmarker_result)
            
            # Append the current landmark to the landmarks array
            landmarks.append(cur_landmark)

    # Release the video capture and destroy any open windows
    cap.release()

    # Save the landmark into a special video_landmark format
    video_landmark = VideoLandmark(video_path, landmarks)
    
    return video_landmark
        
def get_video_paths(categories):
    video_paths = []
    
    for input_category in categories:
        source_directory = os.path.join(INPUT_DIRECTORY, input_category)
        output_directory = os.path.join(OUTPUT_DIRECTORY, input_category)
        
        # Check if the output directory exists, if not, create it
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
            print("Creating directory:", output_directory)
        else:
            print("Directory already exists:", output_directory)
        
        # Get a list of all files in the source directory
        files = os.listdir(source_directory)
        
        # Filter out directories and obtain the full file paths in source directory
        files = [file_name for file_name in files if os.path.isfile(os.path.join(source_directory, file_name))]
        
        # For each file, check if it has already been assigned its 
        # own directory in output_directory. If not, create it.
        # We want each file to have its own directory since that's where
        # all the transcripts, audio_file, and QA data will be stored
        for file_name in files:
            # assert(file_name.endswith(".mp4")), "Error in logic"
            file_output_directory = os.path.join(output_directory, file_name.split(".mp4")[0])
            
            # Check if the file output directory exists, if not, create it
            if not os.path.exists(file_output_directory):
                os.makedirs(file_output_directory)
                print("\tCreating directory:", file_output_directory)
            
            # Add the file_path to the list of video_paths
            # represent it as a tuple of (input_category, video_path, video_output_directory)
            video_paths.append((input_category, os.path.join(source_directory, file_name), file_output_directory))
            
    return video_paths

def thread_landmark_fn(thread_input_tuple):
    category, video_path, output_directory = thread_input_tuple
    
    if category == "Two-Player":
        speaker_count = 2
    else:
        speaker_count = 1
        
    with lock:
        cur_tqdm_position = started.value
        started.value += 1
    
    # Retrieve the landmarks
    try:
        video_landmark = landmark_video(video_path, speaker_count=speaker_count, tqdm_enabled=True, tqdm_position=cur_tqdm_position)
    except Exception as e:
        print("Killed thread for video_path:", video_path, "due to error:", e)
        return
    
    # Designate the appropriate output path
    file_path = os.path.join(output_directory, "landmark.pkl")

    # Open the file in binary mode and use pickle.dump() to write the instance
    with open(file_path, "wb") as file_handle:
        pickle.dump(video_landmark, file_handle)
    
    ###################################
    # Update progress
    ###################################
    with lock:
        progress.value += 1
        current_progress = progress.value

    # Calculate elapsed time and average processing time
    elapsed_time = time.time() - start_time
    average_time_per_file = elapsed_time / current_progress

    # Calculate estimated remaining time
    remaining_files = len(video_files) - current_progress
    estimated_remaining_time = remaining_files * average_time_per_file

    # Create dynamic progress message
    progress_message = "Progress: {}/{}".format(current_progress, len(video_files))
    elapsed_time_message = "Elapsed Time: {:.2f}s".format(elapsed_time)
    avg_time_per_file_message = "Average Time per File: {:.2f}s".format(average_time_per_file)
    estimated_remaining_time_message = "Estimated Remaining Time: {:.2f}s".format(estimated_remaining_time)

    # Create final output message by concatenating all individual messages
    output_message = "\r{} | {} | {} | {}".format(progress_message, elapsed_time_message,
                                                   avg_time_per_file_message, estimated_remaining_time_message)
    
    # Print the dynamic output message
    print(output_message, end='', flush=True)
    
def clear_output():
    print("\r", end="")
        
# Write code that runs if .py file is run as a script
if __name__ == "__main__":
    categories = ["Two-Player"]
    # categories = ["One-Player", "Two-Player", "On-Court", "Need-Classification"]

    # get the list of video_files
    video_files = get_video_paths(categories)
    input("Continue? ")
    clear_output()
    
    # Shared variables
    started = multiprocessing.Value('i', 0)
    progress = multiprocessing.Value('i', 0)
    lock = multiprocessing.Lock()
    start_time = time.time()
    
    # run a pool with as many cpus as we have
    # cpu_count = multiprocessing.cpu_count()
    cpu_count = 20
    print("Working with {} cpus".format(cpu_count))
    with multiprocessing.Pool(processes=cpu_count) as pool:
        input("Continue? ")
        pool.map(thread_landmark_fn, video_files)
    
    print("\nProcessing complete.")  # Print newline after completion