"""
run_specgram.py

Created By Alexander Yared (akyared@gmail.com)
Main Script for the Live Spectrogram project, a real time spectrogram
visualization tool
Dependencies: matplotlib, numpy and the mic_read.py module

"""

############### Import Libraries ###############
import asyncio
import datetime
import time
from tkinter.ttk import Progressbar

from tensorflow.keras.models import load_model
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.mlab import window_hanning, specgram
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
import numpy as np
from tkinter import *
from pandas import DataFrame
from matplotlib import style, ticker
############### Import Modules ###############
from scipy import ndimage
from scipy.constants import degree
from matplotlib import dates

import mic_read

############### Constants ###############
# SAMPLES_PER_FRAME = 10 #Number of mic reads concatenated within a single window
from clean import downsample_mono
from generalTools import load_dict

SAMPLES_PER_FRAME = 20
nfft = 1024  # 256#1024 #NFFT value for spectrogram
overlap = 0  # 512 #overlap value for spectrogram
rate = mic_read.RATE  # sampling rate
model = load_model("models/lstm.h5",
                   custom_objects={'Melspectrogram': Melspectrogram,
                                   'Normalization2D': Normalization2D})

root = Tk()
############### Functions ###############
"""
get_sample:
gets the audio data from the microphone
inputs: audio stream and PyAudio object
outputs: int16 array
"""


def get_sample(stream, pa):
    data = mic_read.get_data(stream, pa)
    return data


"""
get_specgram:
takes the FFT to create a spectrogram of the given audio signal
input: audio signal, sampling rate
output: 2D Spectrogram Array, Frequency Array, Bin Array
see matplotlib.mlab.specgram documentation for help
"""


def get_specgram(signal, rate):
    arr2D, freqs, bins = specgram(signal, Fs=rate, NFFT=nfft, mode='psd')
    return arr2D, freqs, bins


"""
update_fig:
updates the image, just adds on samples at the start until the maximum size is
reached, at which point it 'scrolls' horizontally by determining how much of the
data needs to stay, shifting it left, and appending the new data. 
inputs: iteration number
outputs: updated image
"""


async def run_predict(data, n):
    step = mic_read.CHUNK_SIZE
    batch = []

    wav = data.astype(np.float32, order='F')
    for i in range(0, wav.shape[0], step):
        sample = wav[i:i + step]
        sample = sample.reshape(1, -1)
        if sample.shape[0] < step:
            tmp = np.zeros(shape=(1, step), dtype=np.float32)
            tmp[:, :sample.shape[1]] = sample.flatten()
            sample = tmp
        batch.append(sample)
    X_batch = np.array(batch, dtype=np.float32)
    y_pred = model.predict(X_batch)

    for i in range(len(y_pred[0])):
        print(predict_labels[0])
        print(predict_labels[0][i])
        print(predict_labels[0][i]["text"])
        print(predict_labels)
        predict_labels[0][i]["text"] = y_pred[0][i]
        predict_labels[1][i]["value"] = int(100 * y_pred[0][i])


async def visual(data, n):
    arr2D, freqs, bins = get_specgram(data, rate)
    im_data = im.get_array()
    # if n < SAMPLES_PER_FRAME:
    # keep_block = arr2D.shape[1] * (SAMPLES_PER_FRAME - 1)
    if n < SAMPLES_PER_FRAME:
        im_data = np.hstack((im_data, arr2D))
        im.set_array(im_data)
    else:
        keep_block = arr2D.shape[1] * (SAMPLES_PER_FRAME - 1)
        im_data = np.delete(im_data, np.s_[:-keep_block], 1)
        im_data = np.hstack((im_data, arr2D))
        im.set_array(im_data)

    # Get the image data array shape (Freq bins, Time Steps)
    # shape = im_data.shape

    # Find the CW spectrum peak - look across all time steps
    # f = int(np.argmax(im_data[:]) / shape[1])

    return im,


async def process(data, n):
    process_spectogram = asyncio.create_task(visual(data, n))
    process_predict = asyncio.create_task(run_predict(data, n))

    await process_spectogram
    await process_predict


def update_fig(n):
    data = get_sample(stream, pa)
    asyncio.run(process(data, n))
    return im,


def main():
    global stream
    global pa
    global im
    global fig
    ############### Initialize Plot ###############

    fig = plt.figure()

    ############### GUI ########################

    """
    Launch the stream and the original spectrogram
    """
    stream, pa = mic_read.open_mic()
    data = get_sample(stream, pa)
    arr2D, freqs, bins = get_specgram(data, rate)
    # print("first array:",arr2D)

    """
    Setup the plot paramters
    """
    extent = (bins[0], bins[-1] * SAMPLES_PER_FRAME, freqs[-1], freqs[0])
    # print(max(freqs),max(bins))
    # arr2D = ndimage.rotate(arr2D,3.14)
    im = plt.imshow(arr2D, aspect='auto', extent=extent, norm=LogNorm(vmin=1 / pow(10, 10), vmax=pow(10, 8)), )

    # im = ndimage.rotate(im,angle=degree*90)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    # plt.title('Real Time Spectogram')
    plt.gca().invert_yaxis()
    plt.setp(plt.gca().get_xticklabels(), rotation='horizontal', fontsize=10)
    plt.setp(plt.gca().get_yticklabels(), rotation=0, fontsize=10)
    # plt.setp(, rotation='vertical', fontsize=10)
    plt.gca().invert_xaxis()
    # plt.gca().set_axis_off()
    plt.subplots_adjust(top=0.92, right=1, bottom=0.08, left=.08,
                        hspace=0, wspace=5)
    ############### Animate ###############

    plotcanvas = FigureCanvasTkAgg(fig, middle_bar)
    plotcanvas.get_tk_widget().pack(fill=BOTH, expand=True)
    cbar = plt.colorbar()
    cbar.set_label('Frequency Power (dB)')

    anim = animation.FuncAnimation(fig, update_fig, blit=False,
                                   interval=mic_read.CHUNK_SIZE / 1000, cache_frame_data=True)
    try:
        root.mainloop()
    except:
        print("Plot Closed")

    ############### Terminate ###############
    stream.stop_stream()
    stream.close()
    pa.terminate()
    print("Program Terminated")


if __name__ == "__main__":
    root.title('Classificação Sonar - Inteligência Artificial')
    root.state('zoomed')
    root.config(background='#ffffff')
    globaViewMaisGlobal = Frame(root, bg="white")
    globaViewMaisGlobal.pack(fill=BOTH, expand=True)
    globaViewMaisGlobal.columnconfigure(0, weight=1)
    globaViewMaisGlobal.columnconfigure(1, weight=1)
    globaViewMaisGlobal.rowconfigure(0, weight=1)
    coluna1 = Frame(globaViewMaisGlobal)
    coluna1.grid(column=0, row=0, sticky="nsew")
    coluna2 = Frame(globaViewMaisGlobal, bg="brown")
    coluna2.grid(column=1, row=0, sticky="nsew")

    leftPanel = Frame(coluna1, bg="black")
    leftPanel.pack(fill=BOTH, expand=True)
    leftPanel.columnconfigure(0, weight=1)
    leftPanel.rowconfigure([0, 1, 2], weight=1)
    top_bar = Frame(leftPanel, bg="yellow")
    top_bar.grid(row=0, sticky="nsew")
    middle_bar = Frame(leftPanel, height=200, bg="red")
    middle_bar.grid(row=1, sticky="nsew")
    bottom_bar = Frame(leftPanel, bg="blue")
    bottom_bar.grid(row=2, sticky="nsew")

    rightPanel = Frame(coluna2, bg="grey")
    #rightPanel.pack(fill=BOTH, expand=True)
    rightPanel.pack(fill=BOTH)

    classesLabels = load_dict("classes")
    predict_labels = [[],[]]
    for i in range(len(classesLabels)):
        Label(rightPanel, text=classesLabels[i], width=20).grid(column=0, row=i, sticky="nsew")
        label = Label(rightPanel, text="0.0", width=20)
        label.grid(column=1, row=i, sticky="nsew")
        predict_labels[0].append(label)
        progressBar = Progressbar(rightPanel, orient=HORIZONTAL, length=100, mode='determinate', value=0)
        progressBar.grid(column=2, row=i, sticky="nsew")
        predict_labels[1].append(progressBar)

    """classes = Frame(rightPanel, bg="yellow")
    classes.grid(row=0, column=0, sticky="nsew")
    classes.columnconfigure(0, weight=1)
    classes.rowconfigure(0, weight=1)"""

    w = 1400
    h = 800
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()
    x = (ws / 2) - (w / 2)
    y = (hs / 2) - (h / 2)
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))

    main()
