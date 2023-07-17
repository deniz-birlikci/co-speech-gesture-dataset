from moviepy.editor import VideoFileClip
import librosa
import soundfile as sf
import os
from pipeline import Module

class AudioRetrieval(Module):
    '''
    This class is used to retrieve audio from a video file and perform audio
    processing tasks such as resampling.

    Args:
        None

    Attributes:
        audio_directory (str): The directory where the temporary audio files
            are stored.

    Methods:
        parse_filename(video_path): Parses the video path and returns the
            corresponding filename.
        get_audio_file(video_path): Retrieves the audio from the video and
            saves it as a WAV file.
        forward(video_path): Executes the forward pass of the module,
            extracting and processing the audio.
        __call__(video_path=None): Allows the instance to be called as a
            function, triggering the forward pass and assigning the audio path
            to the pipeline.

    Arguments used from self.pipeline:
        video_path (str): The path of the video file to process.

    Arguments added to self.pipeline:
        audio_path (str): The path of the audio file that has been processed.

    Usage:
        audio_retrieval = AudioRetrieval()
        audio_path = audio_retrieval(video_path)

    '''

    def __init__(self):
        super().__init__()
        self.temporary_audio_directory = "/work/mbirlikc/temp/"

    def parse_filename(self, video_path):
        assert(video_path.endswith(".mp4"))
        return video_path.split("/")[-1].split(".mp4")[0]

    def get_audio_file(self, video_path, audio_directory):
        video = VideoFileClip(video_path)
        audio = video.audio

        # Assign an audio path
        filename = self.parse_filename(video_path) + ".wav"
        audio_path = os.path.join(audio_directory, filename)

        # Get the audio from the video, store it in audio_path in wav format
        audio.write_audiofile(audio_path, codec='pcm_s16le', verbose=False, logger=None)  # Prints nothing

        # Load the audio file - we will need to resample its rate
        audio, current_sample_rate = librosa.load(audio_path, sr=None)

        # Resample the audio to the desired sample rate
        desired_sample_rate = 16000
        resampled_audio = librosa.resample(audio, orig_sr=current_sample_rate, target_sr=desired_sample_rate)

        # Save the resampled audio as a WAV file
        sf.write(audio_path, resampled_audio, desired_sample_rate)

        return audio_path

    def forward(self, video_path, audio_directory):
        self.audio_path = self.get_audio_file(video_path, audio_directory)
        return self.audio_path

    def __call__(self, video_path=None):
        if self.pipeline is not None:
            audio_directory = self.pipeline.output_directory
        else:
            audio_directory = self.temporary_audio_directory

        if video_path is None:
            if self.pipeline is not None:
                video_path = self.pipeline.video_path
            else:
                raise ValueError("Pipeline isn't connected. User has to specify video_path.")

        self.forward(video_path, audio_directory)

        # Assign the audio_path output to the pipeline itself
        if self.pipeline is not None:
            self.pipeline.audio_path = self.audio_path

        return self.audio_path
