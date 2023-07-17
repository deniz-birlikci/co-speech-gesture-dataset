import os
from tqdm.notebook import tqdm
import json

PARENT_DIRECTORY = "/work/mbirlikc/data/Processed Interviews/"

class Pipeline:
    def __init__(self, segments=[], parent_directory=PARENT_DIRECTORY, expect_already_existing_directory=True):
        self.segments = []
        self.expect_already_existing_directory = expect_already_existing_directory

        for module in segments:
            self.add_module(module)

        # Set up the output directory
        self.parent_directory = parent_directory
        
    def add_module(self, module):
        if not isinstance(module, Module):
            raise ValueError("Not the right instance")
        self.segments.append(module)

        # Have a backward connection to the pipeline itself
        module.pipeline = self

    def get_output_folder(self, file_path):
        category = file_path.split("/")[-2]
        filename = file_path.split("/")[-1].split(".mp4")[0]
        return os.path.join(self.parent_directory, category, filename)

    def create_output_directory(self, video_path):
        directory_path = self.get_output_folder(video_path)
        # Check if the directory exists
        if not os.path.exists(directory_path):
            # Create a directory
            os.mkdir(directory_path)
            
            if self.expect_already_existing_directory:
                raise Warning(f"Directory should have already existed: {directory_path}")

        return directory_path

    

    def forward_on_video_file(self, video_path):
        # Establishing the video path
        self.video_path = video_path

        # Setting output directory
        self.output_directory = self.create_output_directory(video_path)

        # Running each segment
        for module in self.segments:
            # Call each module
            result = module()

        return result

    def forward_on_directory(self, directory_path, logger_path=None, logger_overwrite=True):
        # Doing two passes to be able to use TQDM
        available_video_paths = []
        for root, dirs, files in os.walk(directory_path):
            for file_name in files:
                if file_name.endswith(".mp4"):
                    video_path = os.path.join(root, file_name)
                    available_video_paths.append(video_path)

        # Initialize the logger
        if not logger_overwrite and logger_path is not None:
            with open(logger_path) as json_file:
                existing_logger = json.load(json_file)
                already_parsed_paths = set(existing_logger.keys())

            # new_logger_path = logger_path.split(".json")[0] + "_cont.json"
            # with open(new_logger_path) as json_file:
            #     existing_logger = json.load(json_file)
            #     already_parsed_paths = already_parsed_paths | set(existing_logger.keys())

        else:
            already_parsed_paths = set()


        print(f"Found {len(already_parsed_paths)} parsed files already")
        video_paths = set(available_video_paths) - already_parsed_paths
        print(f"Paths to parse got down from {len(available_video_paths)} to {len(video_paths)} with the logger")

        progress_bar = tqdm(total=len(video_paths))

        logger = {}

        skipped_count = 0
        for video_path in video_paths:
            progress_bar.set_description(f"Processing file: {video_path.split('/')[-1]}")

            if video_path not in logger:
                try:
                    output = self.forward_on_video_file(video_path)
                    logger[video_path] = output
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except:
                    print(f"FAIL with {video_path}")
            else:
                skipped_count += 1

            progress_bar.update(1)

            # If there is a specified logger path, update the file
            if logger_path is not None:
                new_logger_path = logger_path.split(".json")[0] + "_cont_v2.json"
                with open(new_logger_path, "w+") as json_file:
                    json.dump(logger, json_file)

        progress_bar.close()

        return logger

    def __call__(self, path):
        if os.path.isfile(path) and path.endswith(".mp4"):
            print("Running pipeline for a video_path")
            return self.forward_on_video_file(path)
        elif os.path.isdir(path):
            print(f"Running pipeline for all files in directory {path}")
            return self.forward_on_directory(path)
        else:
            raise ValueError("The path does not exist or is not legal.")

class Module:
    def __init__(self):
        self.pipeline = None