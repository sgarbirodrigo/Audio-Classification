import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

from kapre.time_frequency import Melspectrogram
from pandas import np

cwd = os.getcwd()
filename = 'test_sound_whale.wav'
y, sr = librosa.load("{}/{}".format(cwd, filename))
print("sr:", sr)

# trim silent edges
whale_song, _ = librosa.effects.trim(y)

##normal wave
# librosa.display.waveplot(whale_song, sr=sr)
# plt.tight_layout()

n_fft = 2048
##FFTT
# = np.abs(librosa.stft(whale_song[:n_fft], n_fft=n_fft, hop_length=n_fft+1))
# plt.plot(D)
# plt.show()

##FFTT vs Time
hop_length = 160
D = np.abs(librosa.stft(whale_song, n_fft=n_fft, hop_length=hop_length))
# librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear')
# plt.colorbar()

# DB = librosa.amplitude_to_db(D, ref=np.max)
# librosa.display.specshow(DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
# plt.colorbar(format='%+2.0f dB')

n_mels = 128
mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
librosa.display.specshow(mel, sr=sr, hop_length=hop_length, x_axis='linear',
                         fmin=0.0, fmax=sr/2)

plt.ylabel('Mel filter')
plt.colorbar()
plt.title('1. Our filter bank for converting from Hz to mels.')

plt.show()
