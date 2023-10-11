import librosa
from Levenshtein import distance as levenshtein_distance
from typing import Callable, Dict
import numpy as np
from karaoke_data import KaraokeData
import logging

# from dtaidistance import dtw
# from tslearn.metrics import dtw
# from fastdtw import fastdtw

# from scipy.spatial.distance import euclidean


class AudioScorer:
    """Handles audio scoring tasks."""

    def __init__(self, karaoke_data_instance: KaraokeData, transcriber: Callable):
        """Initialize with a karaoke data instance and a transcriber."""
        self.karaoke_data = karaoke_data_instance
        self.transcriber = transcriber

    def transcribe_audio(self, audio: np.ndarray) -> str:
        """Transcribes the provided audio."""
        return self.transcriber.transcribe(audio, self.karaoke_data.sampling_rate)

    def linguistic_accuracy_score(self, user_audio: np.ndarray, actual_lyrics: str) -> float:
        """Computes linguistic accuracy based on transcribed text."""
        try:
            user_transcription = self.transcribe_audio(user_audio).lower().strip()
            normalized_lyrics = actual_lyrics.lower().strip()
            distance = levenshtein_distance(user_transcription, normalized_lyrics)
            user_score = 1 - (distance / max(len(user_transcription), len(normalized_lyrics)))
            return user_score
        except Exception as e:
            logging.error(f"Error computing linguistic accuracy score: {e}")
            return 0.0  # Default score

    def compute_dtw_score(self, user_audio_features: np.ndarray, reference_audio_features: np.ndarray) -> float:
        """Computes DTW score between user and reference audio features."""
        try:
            distance = self.basic_dtw(user_audio_features, reference_audio_features)
            # distance = dtw(user_audio_features, reference_audio_features)
            # distance = dtw.distance(user_audio_features, reference_audio_features)

            # Normalize the distance
            normalized_distance = distance / (len(reference_audio_features) + len(user_audio_features))
            score = 1 / (1 + normalized_distance)

            return score
        except Exception as e:
            logging.error(f"Error computing DTW score: {e}")
            return 0.0  # Default score

    def basic_dtw(self, s, t):
        """
        Computes the Dynamic Time Warping distance between sequences s and t.
        :param s: Sequence s
        :param t: Sequence t
        :return: DTW distance
        """
        n, m = len(s), len(t)
        dtw_matrix = np.zeros((n + 1, m + 1))
        for i in range(n + 1):
            for j in range(m + 1):
                dtw_matrix[i, j] = np.inf
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(s[i - 1] - t[j - 1])
                dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])

        return dtw_matrix[n, m]

    def amplitude_matching_score(self, user_audio: np.ndarray, reference_audio: np.ndarray) -> float:
        """Compute amplitude matching score."""
        try:
            user_audio_1d = user_audio.flatten()
            reference_audio_1d = reference_audio.flatten()
            return self.compute_dtw_score(user_audio_1d, reference_audio_1d)
        except Exception as e:
            print(f"Error computing amplitude matching score: {e}")
            return 0.0  # Default score

    def pitch_matching_score(self, user_audio: np.ndarray, reference_audio: np.ndarray) -> float:
        """Compute pitch matching score."""
        try:
            user_pitch, _, _ = librosa.pyin(user_audio, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
            reference_pitch, _, _ = librosa.pyin(
                reference_audio, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
            )
            return self.compute_dtw_score(user_pitch, reference_pitch)
        except Exception as e:
            print(f"Error computing pitch matching score: {e}")
            return 0.0  # Default score

    def rhythm_score(self, user_audio: np.ndarray, reference_audio: np.ndarray) -> float:
        """Compute rhythm score."""
        try:
            user_onset_env = librosa.onset.onset_strength(y=user_audio)
            reference_onset_env = librosa.onset.onset_strength(y=reference_audio)
            return self.compute_dtw_score(user_onset_env, reference_onset_env)
        except Exception as e:
            print(f"Error computing rhythm score: {e}")
            return 0.0  # Default score

    def process_audio_chunk(self, audio_chunk: np.ndarray) -> Dict[str, float]:
        """Process an audio chunk and compute scores."""
        scoring_functions = {
            "linguistic_score": self.linguistic_accuracy_score,
            "amplitude_score": self.amplitude_matching_score,
            "pitch_score": self.pitch_matching_score,
            "rhythm_score": self.rhythm_score,
        }

        scores = {}
        for score_name, func in scoring_functions.items():
            try:
                if score_name == "linguistic_score":
                    scores[score_name] = func(audio_chunk, self.karaoke_data.get_next_lyrics())
                else:
                    scores[score_name] = func(audio_chunk)
            except Exception as e:
                print(f"Error computing {score_name}: {e}")
                scores[score_name] = 0.0  # Default score

        return scores
