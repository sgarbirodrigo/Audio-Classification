from tkinter import *
import argparse
from tkinter import filedialog

from clean import split_wavs

root = Tk()
w =800
h = 800
ws = root.winfo_screenwidth()
hs = root.winfo_screenheight()
x = (ws / 2) - (w / 2)
y = (hs / 2) - (h / 2)
root.geometry('%dx%d+%d+%d' % (w, h, x, y))



class Clean:
    def __init__(self,root):
        self.selected_folder = "/Volumes/My Passport/HD externo/1 - Projetos/6 - Sonar/TestDataset"
        self.root = root
        print("clean")

    def call_clean(self):
        parser = argparse.ArgumentParser(description='Cleaning audio data')
        #print("selected folder:",self.selected_folder)
        parser.add_argument('--src_root', type=str, default=self.selected_folder,
                            help='directory of audio files in total duration')

        upperDirectory = self.selected_folder.replace(self.selected_folder.split("/")[-1],"")
        parser.add_argument('--dst_root', type=str, default="{}/cleaned_dataset".format(upperDirectory),
                            help='directory to put audio files split by delta_time')
        parser.add_argument('--delta_time', '-dt', type=float, default=1.0,
                            help='time in seconds to sample audio')
        parser.add_argument('--sr', type=int, default=16000,
                            help='rate to downsample audio')
        parser.add_argument('--fn', type=str, default='6__10_07_13_marDeCangas_Entra16',
                            help='file to plot over time to check magnitude')
        parser.add_argument('--threshold', type=str, default=20,
                            help='threshold magnitude for np.int16 dtype')
        args, _ = parser.parse_known_args()
        split_wavs(args)

    def select_folder(self):
        self.selected_folder = filedialog.askdirectory()
        self.folder_path_label.config(text=self.selected_folder)

    def draw_widgets(self):
        self.folder_path_label = Label(self.root, text="Directory: {}".format(self.selected_folder))
        self.folder_path_label.pack()

        self.btn_directory = Button(root, text="Select Audio Folder", command=self.select_folder)
        self.btn_directory.pack()

        self.btn_clean = Button(root, text="Prepare Audio Files", command=self.call_clean)
        self.btn_clean.pack()


clean = Clean(root)
clean.draw_widgets()
root.mainloop()