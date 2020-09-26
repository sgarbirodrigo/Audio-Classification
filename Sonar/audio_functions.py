############### Functions ###############
"""
get_sample:
gets the audio data from the microphone
inputs: audio stream and PyAudio object
outputs: int16 array
"""
from matplotlib.mlab import specgram

import mic_read


class AudioCollect:
    def get_sample(self, stream, pa):
        data = mic_read.get_data(stream, pa)
        return data

class PrepareToVisualize:
    """
    get_specgram:
    takes the FFT to create a spectrogram of the given audio signal
    input: audio signal, sampling rate
    output: 2D Spectrogram Array, Frequency Array, Bin Array
    see matplotlib.mlab.specgram documentation for help
    """
    def get_specgram(self,signal, rate):
        arr2D, freqs, bins = specgram(signal, Fs=rate, NFFT=nfft, mode='psd')
        return arr2D, freqs, bins
