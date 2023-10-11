import re
import librosa
import logging
import numpy as np
from typing import List, Dict, Union, Tuple


class KaraokeData:
    """Manages data and alignment for karaoke songs."""

    # Constants for alignment methods
    ALIGNMENT_METHODS = {
        "cross_correlation": "_align_cross_correlation",
        "onset_detection": "_align_onset_detection",
        "lyrics_data": "_align_lyrics_data",
        "start": "_align_start",
    }

    def __init__(self, original_audio: np.array, track_audio: np.array, raw_lyrics_data: str, sampling_rate: int):
        """Initializes the karaoke data object with audio and lyrics."""
        self.original_audio = original_audio
        self.track_audio = track_audio
        self.sampling_rate = sampling_rate
        self.current_position = 0
        self.initial_alignment_done = False
        self.lyrics_data = self._parse_lyrics(raw_lyrics_data)

    def get_lyrics(self, start_time: float, end_time: float) -> List[str]:
        """Fetches the lyrics between the specified start and end times."""
        lyrics_within_interval = []
        for entry in self.lyrics_data:
            if start_time <= entry["time"] <= end_time:
                lyrics_within_interval.append(entry["lyrics"])
        return lyrics_within_interval

    def _parse_lyrics(self, raw_lyrics: str) -> List[Dict[str, Union[float, str]]]:
        """Converts raw LRC lyrics into a structured format."""
        lines = raw_lyrics.split("\n")
        parsed_lyrics = []
        pattern = re.compile(r"\[(\d+):(\d+\.\d+)\](.+)")
        for line in lines:
            try:
                match = pattern.match(line)
                if match:
                    minutes, seconds, lyrics_text = match.groups()
                    time_seconds = float(minutes) * 60 + float(seconds)
                    parsed_lyrics.append({"time": time_seconds, "lyrics": lyrics_text.strip()})
                else:
                    logging.warning(f"Unexpected line format encountered - '{line}'")
            except Exception as e:
                logging.error(f"Error parsing line '{line}': {e}")
        return parsed_lyrics

    def align_audio(self, audio_chunk: np.array, method: str = "cross_correlation"):
        """Aligns the audio using the specified method."""
        if method not in self.ALIGNMENT_METHODS:
            raise ValueError(f"Invalid alignment method. Available methods: {list(self.ALIGNMENT_METHODS.keys())}")
        alignment_method = getattr(self, self.ALIGNMENT_METHODS[method])
        alignment_method(audio_chunk)

    def get_next_segment(self, audio_chunk_length: int) -> Tuple[np.array, np.array]:
        """
        Retrieve the next segment from the original audio and track audio based on the provided chunk length.
        """
        if not self.initial_alignment_done:
            raise ValueError("Initial alignment is required before accessing subsequent segments.")
        end_sample = self.current_position + audio_chunk_length
        original_segment = self.original_audio[self.current_position : min(end_sample, len(self.original_audio))]
        track_segment = self.track_audio[self.current_position : min(end_sample, len(self.track_audio))]
        self.current_position = min(end_sample, len(self.original_audio))
        return original_segment, track_segment

    def _get_audio_segment(self, audio: np.array, segment_length: float) -> np.array:
        """Extracts a segment of audio based on current position and segment length."""
        start_sample = librosa.time_to_samples(self.current_position, sr=self.sampling_rate)
        end_sample = start_sample + librosa.time_to_samples(segment_length, sr=self.sampling_rate)
        return audio[start_sample:end_sample]

    def _align_start(self, audio_chunk: np.array):
        """Aligns the audio starting at the beginning."""
        self.current_position = 0
        self.initial_alignment_done = True

    def _align_lyrics_data(self, audio_chunk: np.array):
        """Aligns the audio using the first entry in lyrics data."""
        if self.lyrics_data and not self.initial_alignment_done:
            start_time = self.lyrics_data[0]["time"]
            self.current_position = librosa.time_to_samples(start_time, sr=self.sampling_rate)
            self.initial_alignment_done = True

    def _align_onset_detection(self, audio_chunk: np.array):
        """Aligns the audio using onset detection."""
        if not self.initial_alignment_done:
            try:
                original_onsets = librosa.onset.onset_detect(y=self.original_audio, sr=self.sampling_rate)
                chunk_onsets = librosa.onset.onset_detect(y=audio_chunk, sr=self.sampling_rate)
                original_onset_samples = librosa.frames_to_samples(original_onsets)
                chunk_onset_samples = librosa.frames_to_samples(chunk_onsets)
                if original_onset_samples.size > 0 and chunk_onset_samples.size > 0:
                    offset = original_onset_samples[0] - chunk_onset_samples[0]
                    self.current_position = max(0, offset)
                else:
                    self.current_position = 0
                self.initial_alignment_done = True
            except Exception as e:
                logging.error(f"Error in onset detection alignment: {e}")

    def _align_cross_correlation(self, audio_chunk: np.array):
        """Aligns the audio using cross-correlation."""
        if not self.initial_alignment_done:
            try:
                cross_correlation = np.correlate(self.original_audio, audio_chunk, "valid")
                start_sample = np.argmax(cross_correlation)
                self.current_position = start_sample
                self.initial_alignment_done = True
            except Exception as e:
                logging.error(f"Error in cross-correlation alignment: {e}")

    def reset_alignment(self):
        """Resets the alignment state."""
        self.current_position = 0
        self.initial_alignment_done = False
