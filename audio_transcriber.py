from pipeline import Module

import whisper, torch, json, os
from functools import lru_cache

class WhisperTranscribing(Module):
    def __init__(self, model_desc="small.en", probability_threshold = 0.45):
        super().__init__()

        self.probability_threshold = probability_threshold

        self.model = WhisperTranscribing.retrieve_model(model_desc)

    @staticmethod
    @lru_cache
    def retrieve_model(model_desc):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = whisper.load_model(model_desc, device = device)
        return model

    # TODO: In case we want to edit the text
    def apply_threshold(self):
        pass

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