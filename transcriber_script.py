import os
import time
import threading
import torch
import torch.cuda
import gpu_availability
from tqdm import tqdm
from functools import lru_cache
from multiprocessing import Pool, cpu_count
import whisper, torch, json, os
from functools import lru_cache

class WhisperTranscribing():
    def __init__(self, model):
        self.model = model
        
    @staticmethod
    @lru_cache
    def retrieve_model(model_desc, device):
        model = whisper.load_model(model_desc, device = device)
        return model

    def transcribe_file(self, audio_path):
        if not audio_path.endswith(".wav"):
            raise ValueError("The audio_path has to be path to a .wav file")

        result = self.model.transcribe(
            audio=audio_path, language='en', word_timestamps=True,
            initial_prompt="Umm, let me think like, hmm... Okay, here's what I'm, like, thinking."
        )

        file_directory = "/".join(audio_path.split("/")[:-1])

        # Store the text transcript
        transcribed_text = result["text"]
        transcript_file_path = os.path.join(file_directory, "transcript.txt")
        with open(transcript_file_path, "w") as outfile:
            outfile.write(transcribed_text)

        # Store the whisper inference
        inference_file_path = os.path.join(file_directory, "inference.json")
        with open(inference_file_path, "w") as outfile:
            json.dump(result, outfile, indent=4)

        return result

    def transcribe_directory(self, directory_path, avoid_entire_clip=False):
        for root, dirs, files in os.walk(directory_path):

            # If we want to avoid transcribing the whole clip,
            # and would just rather transcribe each QA session
            if avoid_entire_clip and root == directory_path:
                continue

            for file_name in files:
                if file_name.endswith(".wav"):
                    audio_path = os.path.join(root, file_name)
                    self.transcribe_file(audio_path)

    def __call__(self):
        if self.pipeline is not None:
            output_directory = self.pipeline.output_directory
        else:
            raise ValueError("Pipeline isn't connected. Instead, use the transcribe_directory function and manually provide the directory")

        self.transcribe_directory(output_directory)

class Bridge:
    def __init__(self):
        self.source_directories = []
        self.queue = []
        self.transcribed_directories = []

        self.processed_directory_count = 0

class Status:
    def __init__(self, bridge):
        self.bridge = bridge
        self.update()

    @staticmethod
    def float_to_percentage(number):
        percentage_string = "{:.2f}%".format(number * 100)
        return percentage_string
    
    def update(self):
        self.total_videos = 1706
        self.available_videos = self.bridge.processed_directory_count
        self.on_queue = len(set("/".join(dir.split("/")[:-1]) for dir in self.bridge.queue))
        self.transcribed = len(set("/".join(dir.split("/")[:-1]) for dir in self.bridge.transcribed_directories))

    def __repr__(self):
        self.update()

        avail_percent = self.float_to_percentage(self.available_videos / self.total_videos)
        queue_percent = self.float_to_percentage(self.on_queue / self.total_videos)
        transcribed_perc = self.float_to_percentage(self.transcribed / self.total_videos)

        return (f"{avail_percent} available | {queue_percent} on queue | {transcribed_perc} transcribed")

def get_all_source_directories():
    source_directories = []

    processed_directory = os.path.join(base_directory, "landmarked")

    for category in categories:
        category_directory = os.path.join(processed_directory, category)
        source_directories.extend([os.path.join(category_directory, subdir) for subdir in os.listdir(category_directory) if os.path.isdir(os.path.join(category_directory, subdir))])

    return source_directories

# Function to check if subdirectories haven't changed for 10 seconds
def check_subdirectories():
    processed_directories = set()

    while not stop_event.is_set():
        # Get all the potential source directories
        source_directories = set(get_all_source_directories())

        # Remove the ones we have already processed
        source_directories = source_directories - processed_directories
        bridge.source_directories = sorted(source_directories)

        # Iterate through the sources
        for directory in source_directories:
            subdirectories = [os.path.join(directory, subdir) for subdir in os.listdir(directory) if os.path.isdir(os.path.join(directory, subdir))]
            if not subdirectories:
                continue

            last_change_time = os.stat(directory).st_mtime
            current_time = time.time()

            # If subdirectories haven't changed for 60 seconds, remove the directory from source_directories
            if current_time - last_change_time > 60:
                processed_directories.add(directory)
                bridge.processed_directory_count += 1

        time.sleep(1)
        
def is_already_transcribed(directory):
    for root, _, files in os.walk(directory):
        if 'transcript.txt' in files:
            return True
    return False

# Function to run in the background and append new paths to the array
def background_function():
    processing_thread = threading.Thread(target=check_subdirectories)
    processing_thread.daemon = True
    processing_thread.start()

    visited = set()

    while not stop_event.is_set():
        for directory in bridge.source_directories:
            subdirectories = [subdir for subdir in os.listdir(directory) if os.path.isdir(os.path.join(directory, subdir))]
            for subdir in subdirectories:
                subdir_path = os.path.join(directory, subdir)
                if subdir_path not in visited:
                    visited.add(subdir_path)
                    if is_already_transcribed(subdir_path):
                        bridge.transcribed_directories.append(subdir_path)
                    else:
                        bridge.queue.append(subdir_path)

        time.sleep(1)

    
def transcribe_single_file(self, audio_path):
        try:
            # Determine the device (GPU or CPU) for this transcription
            device = torch.device("cuda", gpu_availability.available_gpus()[0]) if torch.cuda.is_available() else torch.device("cpu")
            
            # Load the model for this specific GPU
            model = WhisperTranscribing.retrieve_model("tiny.en", device)

            transcriber = WhisperTranscribing(model)
            return transcriber.transcribe_file(audio_path)
        except Exception as e:
            return f"Error processing {audio_path}: {e}"
    
def transcribing_function():
    # Determine the number of available GPUs
    num_gpus = torch.cuda.device_count()

    # Create a pool of processes equal to the number of GPUs
    pool = Pool(processes=num_gpus)

    while not stop_event.is_set():
        if bridge.queue:
            # Get the next path from the queue for transcription
            path_to_transcribe = bridge.queue.pop(0)

            # Use multiprocessing to transcribe the file concurrently on available GPUs
            results = pool.map(transcribe_single_file, [path_to_transcribe])

            # Add the transcribed file to the transcribed_directories list
            bridge.transcribed_directories.append(path_to_transcribe)
        else:
            # If the queue is empty, wait for new elements to be added
            time.sleep(1)
            
if __name__ == "__main__":
    # Set up the base directory
    base_directory = "/work/mbirlikc/data/landmarked/"
    
    # Set up the locations where search is going to take place
    categories = ["One-Player", "On-Court", "Need-Classification", "Two-Player"]
    
    # Set up the bridge for threads to communicate data over
    bridge = Bridge()

    # Set up status
    status = Status(bridge)

    # Set up TQDM
    progress_bar = tqdm(total=status.total_videos, desc="Processing", unit="iterations")

    # Event to signal the threads to stop
    stop_event = threading.Event()

    # Create and start the background thread
    background_thread = threading.Thread(target=background_function)
    background_thread.daemon = True
    background_thread.start()

    # Create and start the transcribing thread
    transcribing_thread = threading.Thread(target=transcribing_function)
    transcribing_thread.daemon = True
    transcribing_thread.start()

    # Keep the main thread running so the background threads continue to run
    while True:
        try:
            time.sleep(1)

            if status.transcribed >= status.total_videos:
                break
            
            while progress_bar.n < status.transcribed:
                progress_bar.update(1)

        except KeyboardInterrupt:
            break

    # Terminate the background threads when the user presses Ctrl+C
    # Set the stop_event to signal the threads to stop
    stop_event.set()
    progress_bar.close()

    # Wait for the background threads to terminate gracefully
    background_thread.join()
    transcribing_thread.join()
