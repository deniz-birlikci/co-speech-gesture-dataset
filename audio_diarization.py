from pipeline import Module

import torch
import re
import os
from dotenv import load_dotenv
from pyannote.audio import Pipeline as pyannote_pipeline
from collections import Counter
from pydub import AudioSegment

class AudioSegmentation(Module):
    '''
    This class is used to perform speaker diarization on an audio file
    using a pretrained pipeline.

    Args:
        None

    Attributes:
        pipeline (Pipeline): The pretrained pipeline for speaker diarization.

    Methods:
        forward(audio_path):
            Executes the forward pass of the module, applying the pretrained
            pipeline for speaker diarization.
        __call__(audio_path=None):
            Allows the instance to be called as a function, triggering the
            forward pass and assigning the speaker segments to the pipeline.

    Arguments used from self.pipeline:
        audio_path (str): The path of the audio file to process.

    Arguments added to self.pipeline:
        qa_segments (list): A list of QA segments, where each segment is a
            dictionary with the following keys:
            - "question" (dict): The start and end times of the question segment.
                                 None if there is no question.
            - "answer" (dict): The start and end times of the answer segment.
            - "entire" (dict): The start and end times of the entire QA segment.

    Usage:
        audio_segmentation = AudioSegmentation()
        qa_segments = audio_segmentation(audio_path)
    '''

    load_dotenv('.env')

    # Access the environment variable
    HF_API_KEY = os.getenv('HF_API_KEY')

    def __init__(self, spacermilli = 2000):
        super().__init__()

        self.diarization_pipeline = pyannote_pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=AudioSegmentation.HF_API_KEY
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diarization_pipeline.to(self.device)

        self.spacermilli = spacermilli
        self.temp_write_directory = "/work/mbirlikc/temp/"

    @staticmethod
    def millisec(timeStr):
        spl = timeStr.split(":")
        s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
        return s

    @staticmethod
    def convert_milliseconds(milliseconds):
        seconds = int((milliseconds / 1000) % 60)
        minutes = int((milliseconds / (1000 * 60)) % 60)
        return f"{minutes:02d}:{seconds:02d}"

    @staticmethod
    def convert_qa_segments(qa_segments):
        '''
        Converts the start and end values inside QA segments to milliseconds.

        Args:
            qa_segments (list): A list of QA segments, where each segment 
                is a dictionary with "question", "answer", and "entire" keys.

        Returns:
            list: A new list of QA segments with start and end values 
                converted to milliseconds.

        Usage:
            converted_segments = convert_qa_segments(qa_segments)
        '''

        converted_segments = []
        conversion_fn = AudioSegmentation.convert_milliseconds

        for segment in qa_segments:
            converted_segment = {
                "question": None,
                "answer": {"start": None, "end": None},
                "entire": {"start": None, "end": None}
            }

            if segment["question"]:
                question_start = conversion_fn(segment["question"]["start"])
                question_end = conversion_fn(segment["question"]["end"])
                converted_segment["question"] = {"start": question_start, "end": question_end}

            answer_start = conversion_fn(segment["answer"]["start"])
            answer_end = conversion_fn(segment["answer"]["end"])
            converted_segment["answer"] = {"start": answer_start, "end": answer_end}

            entire_start = conversion_fn(segment["entire"]["start"])
            entire_end = conversion_fn(segment["entire"]["end"])
            converted_segment["entire"] = {"start": entire_start, "end": entire_end}

            converted_segments.append(converted_segment)

        return converted_segments

    # pyannote seems to miss the first 0.5 seconds of the audio, hence
    # why we are optionally adding a spacer
    def add_spacer(self, audio_path):
        spacer = AudioSegment.silent(duration=self.spacermilli)
        audio = AudioSegment.from_wav(audio_path)
        audio = spacer.append(audio, crossfade=0)

        temp_file_locn = os.path.join(self.temp_write_directory, 'input_prep.wav')
        audio.export(temp_file_locn, format='wav')

        return temp_file_locn

    def parse_speaker_times(self, diarization):
        # Parse speaker times
        speaker_times = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_times[(turn.start, turn.end)] = speaker
        return speaker_times

    def group_speaker_times(self, diarization):
        # Functions from:
        # https://colab.research.google.com/github/Majdoddin/nlp/blob/main/Pyannote_plays_and_Whisper_rhymes_v_2_0.ipynb#scrollTo=umQdzNFzcP2f

        speaker_groups = []
        cur_speaker = []
        last_end = 0

        for segment in str(diarization).splitlines():
            # If we are moving onto a different speaker
            if cur_speaker and (cur_speaker[0].split()[-1] != segment.split()[-1]):
                speaker_groups.append(cur_speaker)
                cur_speaker = []

            # We add the segment into the current speaker
            cur_speaker.append(segment)

            # We then find if the segment is engulfed by a previous segment
            segment_end = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=segment)[1]
            segment_end = self.millisec(segment_end)

            # If segment engulfed by a previous segment
            if (last_end > segment_end):
                # print("Engulfed segment")
                speaker_groups.append(cur_speaker)
                cur_speaker = []
            else:
                last_end = segment_end

        # If we have gone through all segments and haven't appended the current
        # speaker yet
        if cur_speaker:
            speaker_groups.append(cur_speaker)

        return speaker_groups

    def segment_speakers(self, groups):
        speaker_clips = []
        for speaker in groups:
            start = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=speaker[0])[0]
            end = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=speaker[-1])[1]
            start = self.millisec(start) - self.spacermilli
            end = self.millisec(end) - self.spacermilli

            speaker_id = speaker[0].split()[-1]
            speaker_clips.append([speaker_id, start, end])

        return speaker_clips

    def find_main_speaker(self, speaker_clips):
        counter = Counter()
        for speaker_id, start, end in speaker_clips:
            counter[speaker_id] += end - start

        try:
            return counter.most_common(1)[0][0]
        except:
            print("Error with counter", counter)

        return counter.most_common(1)[0][0]

    def segment_qa(self, speaker_clips):
        main_speaker = self.find_main_speaker(speaker_clips)

        qa_segments = []

        for idx, clip in enumerate(speaker_clips):
            speaker_id, start, end = clip

            if speaker_id == main_speaker:
                cur_qa_segment = {
                    "question" : None,
                    "answer" : {"start" : start, "end" : end},
                    "entire" : {"start" : start, "end" : end},
                }
                if idx > 0:
                    reporter_id, question_start, question_end = speaker_clips[idx - 1]

                    cur_qa_segment["question"] = {
                        "start" : question_start,
                        "end" : question_end
                    }
                    cur_qa_segment["entire"]["start"] = question_start

                qa_segments.append(cur_qa_segment)

        return qa_segments

    def forward(self, audio_path):
        # OPTIONAL: Add spacer at the beginning of audio
        if self.spacermilli > 0:
            audio_path = self.add_spacer(audio_path)

        # Apply pretrained pipeline
        diarization = self.diarization_pipeline(audio_path)

        # self.speaker_segments = self.parse_speaker_times(diarization)
        speaker_groups = self.group_speaker_times(diarization)
        speaker_clips = self.segment_speakers(speaker_groups)
        qa_segments = self.segment_qa(speaker_clips)

        self.qa_segments = qa_segments

        return qa_segments

    def __call__(self, audio_path=None):
        if audio_path is None:
            if self.pipeline is not None:
                audio_path = self.pipeline.audio_path
            else:
                raise ValueError("Pipeline isn't connected. User has to specify audio_path.")

        self.forward(audio_path)

        # Assign the audio_path output to the pipeline itself
        if self.pipeline is not None:
            self.pipeline.qa_segments = self.qa_segments

        return self.qa_segments
