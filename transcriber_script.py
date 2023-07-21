import os
import time
import threading
import torch
import torch.cuda
import gpu_availability
from tqdm import tqdm
from functools import lru_cache
from multiprocessing import Pool, cpu_count, Lock
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

def get_all_source_directories():
    source_directories = []

    processed_directory = os.path.join(base_directory, "landmarked")

    for category in categories:
        category_directory = os.path.join(processed_directory, category)
        source_directories.extend([os.path.join(category_directory, subdir) for subdir in os.listdir(category_directory) if os.path.isdir(os.path.join(category_directory, subdir))])

    return source_directories
        
def is_already_transcribed(directory):
    for root, _, files in os.walk(directory):
        if 'transcript.txt' in files:
            return True
    return False

# Function to run in the background and append new paths to the array
def get_transcribe_targets():
    source_directories = get_all_source_directories()
    targets = []

    for directory in source_directories:
        subdirectories = [subdir for subdir in os.listdir(directory) if os.path.isdir(os.path.join(directory, subdir))]
        for subdir in subdirectories:
            subdir_path = os.path.join(directory, subdir)
            if not is_already_transcribed(subdir_path):
                targets.append(subdir_path)
    
    return targets
        
def transcribing_function(queue, lock):
    # Determine the device (GPU or CPU) for this transcription
    with lock:
        if torch.cuda.is_available():
            gpu_id = gpu_availability.available_gpus()[0]
            device = torch.device("cuda", gpu_id)  
            print(f"Running with {gpu_id} | {device}")
        else:
            device = torch.device("cpu")
            print("Running with cpu")
    
    # Load the model for this specific GPU
    model = WhisperTranscribing.retrieve_model("tiny.en", device)

    transcriber = WhisperTranscribing(model)
    
    while True:
        with lock:
            if not queue:
                break
                
            # Remove and get the first argument from the queue
            path_to_transcribe = queue.pop(0)  
        
        progress_bar.set_description(f"Path: {path_to_transcribe}")
        transcriber.transcribe_directory(path_to_transcribe)

        with lock:
            progress_bar.update(1)
            
if __name__ == "__main__":
    # Set up the base directory
    base_directory = "/work/mbirlikc/data/"
    
    # Set up the locations where search is going to take place
    categories = ["One-Player", "On-Court", "Need-Classification", "Two-Player"]
    
    # Get all subdirectories that can be transcribed
    targets = sorted(get_transcribe_targets())

    # Set up TQDM
    progress_bar = tqdm(total=len(targets), desc="Processing", unit="iterations")

    # Determine the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Working with {num_gpus} gpus")

    # Create a pool of processes equal to the number of GPUs
    pool = Pool(processes=num_gpus)
    
    # The lock to synchronize access to the queue
    lock = Lock()
    
    try:
        # Start the workers asynchronously
        for _ in range(num_gpus):
            pool.apply_async(transcribing_function, args=(targets, lock))
            
        # Wait for all workers to finish
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        print("Terminating due to keyboard interrupt.")
        pool.terminate()
    finally:
        pool.close()
        pool.join()