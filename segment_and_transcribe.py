import os

BASE_DIRECTORY = "/work/mbirlikc/data/"
INPUT_DIRECTORY = os.path.join(BASE_DIRECTORY, r'Interviews')
OUTPUT_DIRECTORY = os.path.join(BASE_DIRECTORY, r'Processed Interviews')
MODELS_DIRECTORY = os.path.join(BASE_DIRECTORY, r'Models')

from pipeline import Pipeline
# from audio_diarization import AudioSegmentation
from audio_retrieval import AudioRetrieval
from audio_clipping import QAClipping
from audio_transcriber import WhisperTranscribing

pipeline = Pipeline([
    AudioRetrieval(),
    # AudioSegmentation(),
    # QAClipping(),
    # WhisperTranscribing()
], parent_directory=OUTPUT_DIRECTORY)

# Write code that runs if .py file is run as a script
if __name__ == "__main__":
    categories = ["One-Player", "On-Court", "Need-Classification", "Two-Player"]
    # Call pipeline with Input directory and category combined into a path
    for category in categories:
        pipeline(os.path.join(INPUT_DIRECTORY, category))