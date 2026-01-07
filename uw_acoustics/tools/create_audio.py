import numpy as np
from scipy.io import wavfile
from scipy.signal import sawtooth

def generate_tone(
    filename="tone.wav",
    frequency=1000.0,     # Hz
    amplitude=0.5,        # 0.0 - 1.0
    duration=2.0,         # seconds
    fs=48000              # sampling rate (Hz)
):
    """
    Generate a sine wave audio file.

    Parameters
    ----------
    filename : str
        Output WAV file name
    frequency : float
        Tone frequency in Hz
    amplitude : float
        Signal amplitude (0.0 to 1.0)
    duration : float
        Duration in seconds
    fs : int
        Sampling frequency in Hz
    """

    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * frequency * t)

    # Convert to 16-bit PCM
    signal_int16 = np.int16(signal / np.max(np.abs(signal)) * 32767)

    wavfile.write(filename, fs, signal_int16)
    print(f"Saved {filename} | f={frequency} Hz, A={amplitude}, fs={fs} Hz")


def generate_triangle_wave(
    filename="triangle.wav",
    frequency=1000.0,     # Hz
    amplitude=0.5,        # 0.0 – 1.0
    duration=2.0,         # seconds
    fs=48000              # sampling rate (Hz)
):
    """
    Generate a triangle wave audio file.

    Parameters
    ----------
    filename : str
        Output WAV file name
    frequency : float
        Wave frequency in Hz
    amplitude : float
        Signal amplitude (0.0 to 1.0)
    duration : float
        Duration in seconds
    fs : int
        Sampling frequency in Hz
    """

    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Triangle wave: sawtooth with width=0.5
    signal = amplitude * sawtooth(2 * np.pi * frequency * t, width=0.5)

    # Convert to 16-bit PCM
    signal_int16 = np.int16(signal / np.max(np.abs(signal)) * 32767)

    wavfile.write(filename, fs, signal_int16)
    print(f"Saved {filename} | triangle wave | f={frequency} Hz, fs={fs} Hz")



if __name__ == "__main__":
    # generate_tone(
    #     filename="tone_1kHz.wav",
    #     frequency=1000.0,
    #     amplitude=0.8,
    #     duration=3.0
    # )

    generate_triangle_wave(
        filename="assets/triangle_1kHz.wav",
        frequency=1000.0,
        amplitude=0.8,
        duration=3.0
    )
