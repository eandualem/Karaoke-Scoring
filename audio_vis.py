import librosa
import numpy as np
import librosa.display
from scipy.signal import welch
import IPython.display as ipd
import matplotlib.pyplot as plt
from IPython.display import display


class AudioVis:
    """Class for audio visualizations."""

    def play_audio(self, samples: np.array, sr: int = 22000) -> None:
        """Plays the provided audio samples."""
        display(ipd.Audio(samples, rate=sr))

    def _extract_audio_segment(
        self, signal: np.array, sr: int, start_time: float = None, end_time: float = None
    ) -> np.array:
        """Extracts a portion of the audio signal based on start and end times."""
        if start_time:
            start_sample = int(start_time * sr)
            if start_sample >= len(signal):
                raise ValueError("Start time is beyond audio duration.")
        else:
            start_sample = None

        if end_time:
            end_sample = int(end_time * sr)
            if end_sample >= len(signal):
                raise ValueError("End time is beyond audio duration.")
        else:
            end_sample = None

        return signal[start_sample:end_sample]

    def wav_plot(
        self,
        signal: np.array,
        sr: int = 22000,
        start_time: float = None,
        end_time: float = None,
        title: str = "Audio Signal",
    ) -> None:
        """Plots the waveform of the audio signal."""
        signal = self._extract_audio_segment(signal, sr, start_time, end_time)
        plt.figure(figsize=(15, 4))
        plt.figure(figsize=(15, 4))
        plt.plot(signal)
        plt.title(title)
        plt.ylabel("Amplitude")
        plt.xlabel("Time (s)")
        plt.tight_layout()
        plt.show()

    def plot_spectrogram(
        self,
        signal: np.array,
        sr: int = 22000,
        start_time: float = None,
        end_time: float = None,
        title: str = "Spectrogram",
    ) -> None:
        """Displays a spectrogram of the audio signal."""
        signal = self._extract_audio_segment(signal, sr, start_time, end_time)
        X = librosa.stft(signal)
        Xdb = librosa.amplitude_to_db(abs(X))
        plt.figure(figsize=(15, 5))
        librosa.display.specshow(Xdb, sr=sr, x_axis="time", y_axis="hz")
        plt.colorbar()
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def plot_log_spectrogram(
        self,
        signal: np.array,
        sr: int = 22000,
        start_time: float = None,
        end_time: float = None,
        title: str = "Log Spectrogram",
    ) -> None:
        """Displays a logarithmic spectrogram of the audio signal."""
        signal = self._extract_audio_segment(signal, sr, start_time, end_time)
        X = librosa.stft(signal)
        Xdb = librosa.amplitude_to_db(abs(X))
        plt.figure(figsize=(15, 5))
        librosa.display.specshow(Xdb, sr=sr, x_axis="time", y_axis="log")
        plt.colorbar()
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def plot_mfcc(
        self,
        signal: np.array,
        sr: int = 22000,
        start_time: float = None,
        end_time: float = None,
        title: str = "MFCC",
        n_mfcc: int = 13,
    ) -> None:
        """Visualizes the MFCC of the audio signal."""
        signal = self._extract_audio_segment(signal, sr, start_time, end_time)
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
        plt.figure(figsize=(15, 5))
        librosa.display.specshow(mfccs, sr=sr, x_axis="time")
        plt.colorbar()
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def plot_psd(
        self,
        signal: np.array,
        sr: int = 22000,
        start_time: float = None,
        end_time: float = None,
        title: str = "Power Spectral Density",
    ) -> None:
        """Plots the power spectral density of the audio signal."""
        signal = self._extract_audio_segment(signal, sr, start_time, end_time)
        freqs, psd = welch(signal, fs=sr)
        plt.figure(figsize=(15, 5))
        plt.semilogy(freqs, psd)
        plt.title(title)
        plt.xlabel("Frequency")
        plt.ylabel("Power")
        plt.tight_layout()
        plt.show()
