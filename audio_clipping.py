from pipeline import Module

from pydub import AudioSegment
from moviepy.editor import VideoFileClip
import librosa
import soundfile as sf
import os


class QAClipping(Module):
    """
    This class is used to perform clipping of audio and video files based on QA segments.

    Attributes:
        temp_write_directory (str): The temporary write directory for storing the clipped files.
        clip_fname_segment_char (str): The character used to separate segments in the output file name.

    Methods:
        create_directory(directory_path):
            Creates a directory if it doesn't exist.
        parse_filename(path):
            Parses the filename from a given path.
        clip_audio(audio_path, start_ms, end_ms):
            Clips the audio file based on the specified start and end times in milliseconds.
        clip_video(video_path, start_ms, end_ms):
            Clips the video file based on the specified start and end times in milliseconds.
        create_clipping(video_path, audio_path, output_directory, output_file_name, start_ms, end_ms):
            Creates the clipping of both audio and video files and saves these clips in the output_directory.
        forward(video_path, audio_path, qa_segments, output_directory):
            Clips the audio and video files based on the QA segments and saves them to the output directory.
        __call__():
            Executes the forward pass of the module, clipping the audio and video files based on the pipeline.

    Arguments used from self.pipeline:
        audio_path (str): The path to the audio file to be clipped.
        video_path (str): The path to the video file to be clipped.
        qa_segments (list): A list of QA segments, where each segment is a dictionary with the following keys:
                            - "question" (dict): The start and end times of the question segment.
                                                 None if there is no question.
                            - "answer" (dict): The start and end times of the answer segment.
                            - "entire" (dict): The start and end times of the entire QA segment.
        output_directory (str): The directory where the clipped files will be saved.

    Arguments added to self.pipeline:
        None

    Usage:
        qaclipping = QAClipping()
        qaclipping(video_path, audio_path, qa_segments, output_directory)
    """
    def __init__(self, ):
        super().__init__()

        self.temp_write_directory = "/work/mbirlikc/temp"
        self.clip_fname_segment_char = "#"

    def create_directory(self, directory_path):
        # Check if the directory exists
        if not os.path.exists(directory_path):
            # Create a directory
            os.mkdir(directory_path)
        else:
            print("Directory already exists.")

    def parse_filename(self, path):
        return path.split("/")[-1].split(".mp4")[0]

    def clip_audio(self, audio_path, start_ms, end_ms):
        audio = AudioSegment.from_wav(audio_path)
        clipped_audio = audio[start_ms:end_ms]
        return clipped_audio

    def clip_video(self, video_path, start_ms, end_ms):
        video = VideoFileClip(video_path)
        clipped_video = video.subclip(start_ms / 1000, end_ms / 1000)
        return clipped_video

    def create_video_clipping(self, video_path, output_directory, output_file_name, start_ms, end_ms):
        output_path = os.path.join(output_directory, output_file_name)

        clipped_video = self.clip_video(video_path, start_ms, end_ms)
        clipped_video.write_videofile(f"{output_path}.mp4", codec="libx264", verbose=False, logger=None)
        
    def create_audio_clipping(self, audio_path, output_directory, output_file_name, start_ms, end_ms):
        output_path = os.path.join(output_directory, output_file_name)

        clipped_audio = self.clip_audio(audio_path, start_ms, end_ms)
        clipped_audio.export(f"{output_path}", format="wav")

    def forward(self, video_path, audio_path, qa_segments, output_directory):
        question_idx = -1

        # Go through each segment, and clip the question, the answer, and the
        # entire sequence seperately
        for segment in qa_segments:
            question_idx += 1

            question_tag = f"question {('0' + str(question_idx)) if (question_idx < 10) else str(question_idx)}"
            directory_question = os.path.join(output_directory, question_tag)
            self.create_directory(directory_question)

            for segment_type in segment:
                directory_segment = os.path.join(directory_question, segment_type)
                self.create_directory(directory_segment)

                file_name = self.parse_filename(video_path)
                # output_file_name = self.clip_fname_segment_char.join([file_name, question_tag, segment_type])
                output_file_name = "audio_segment.wav"

                start_ms, end_ms = segment[segment_type]["start"], segment[segment_type]["end"]

                self.create_audio_clipping(audio_path, directory_segment, output_file_name, start_ms, end_ms)
                # self.create_video_clipping(video_path, directory_segment, output_file_name, start_ms, end_ms)

    def __call__(self, *args):
        if self.pipeline is not None:
            audio_path = self.pipeline.audio_path
            video_path = self.pipeline.video_path
            qa_segments = self.pipeline.qa_segments
            output_directory = self.pipeline.output_directory
        elif len(args) == 4:
            video_path, audio_path, qa_segments, output_directory = args
        else:
            raise ValueError("Pipeline isn't connected. User has to feed in video_path, audio_path, qa_segments, output_directory as arguments.")

        self.forward(video_path, audio_path, qa_segments, output_directory)