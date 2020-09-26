import os
import warnings
from glob import glob
from typing import Any

import tensorflow as tf
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D
from pandas import np
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow_core.python.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow_core.python.keras.utils.np_utils import to_categorical


class AI:
    def __init__(self,src_root,sample_rate, delta_time,batch_size, n_fdt):
        self.src_root = src_root
        self.sample_rate = sample_rate
        self.delta_time = delta_time
        self.batch_size = batch_size
        self.number_of_classes = len(os.listdir(self.src_root))
        self.n_fdt = n_fdt
        self.test_set_percentage = 0.1

    class Model:
        def __init__(self, sample_rate, delta_time, n_fdt, number_of_classes):
            self.sample_rate = sample_rate
            self.delta_time = delta_time
            self.model_type = "lstm"  # 'conv1d','conv2d'
            self.number_of_classes = number_of_classes
            self.n_fdt = n_fdt
            self.test_set_percentage = 0.1

        def LSTM_Model(self):
            i = layers.Input(shape=(1, int(self.sample_rate * self.delta_time)), name='input')
            x = Melspectrogram(n_dft=self.n_fdt, n_hop=160,
                               padding='same', sr=self.sample_rate, n_mels=128,
                               fmin=0.0, fmax=self.sample_rate/2, power_melgram=2.0,
                               return_decibel_melgram=True, trainable_fb=False,
                               trainable_kernel=False,
                               name='melbands')(i)
            x = Normalization2D(str_axis='batch', name='batch_norm')(x)
            x = layers.Permute((2, 1, 3), name='permute')(x)
            x = TimeDistributed(layers.Reshape((-1,)), name='reshape')(x)
            s = TimeDistributed(layers.Dense(64, activation='tanh'),
                                name='td_dense_tanh')(x)
            x = layers.Bidirectional(layers.LSTM(32, return_sequences=True),
                                     name='bidirectional_lstm')(s)
            x = layers.concatenate([s, x], axis=2, name='skip_connection')
            x = layers.Dense(64, activation='relu', name='dense_1_relu')(x)
            x = layers.MaxPooling1D(name='max_pool_1d')(x)
            x = layers.Dense(32, activation='relu', name='dense_2_relu')(x)
            x = layers.Flatten(name='flatten')(x)
            x = layers.Dropout(rate=0.2, name='dropout')(x)
            x = layers.Dense(32, activation='relu',
                             activity_regularizer=l2(0.001),
                             name='dense_3_relu')(x)
            o = layers.Dense(self.number_of_classes, activation='softmax', name='softmax')(x)

            model = Model(inputs=i, outputs=o, name='long_short_term_memory')
            model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            return model

    def train(self):
        csv_path = os.path.join('logs', 'lstm_history.csv')
        wav_paths = glob('{}/**'.format(self.src_root), recursive=True)
        wav_paths = [x.replace(os.sep, '/') for x in wav_paths if '.wav' in x]
        classes = sorted(os.listdir(self.src_root))
        le = LabelEncoder()
        le.fit(classes)
        labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
        labels = le.transform(labels)

        wav_train, wav_val, label_train, label_val = train_test_split(wav_paths,
                                                                      labels,
                                                                      test_size=self.test_set_percentage,
                                                                      random_state=0)

        assert len(label_train) >= self.batch_size, 'Number of train samples must be >= batch_size'
        if len(set(label_train)) != self.number_of_classes:
            warnings.warn('Found {}/{} classes in training data. Increase data size or change random_state.'.format(
                len(set(label_train)), self.number_of_classes))
        if len(set(label_val)) != self.number_of_classes:
            warnings.warn('Found {}/{} classes in validation data. Increase data size or change random_state.'.format(
                len(set(label_val)), self.number_of_classes))

        tg = DataGenerator(wav_train, label_train, self.sample_rate, self.delta_time,
                           self.number_of_classes, batch_size=self.batch_size)

        vg = DataGenerator(wav_val, label_val, self.sample_rate, self.delta_time,
                           self.number_of_classes, batch_size=self.batch_size)

        cp = ModelCheckpoint('models/lstm.h5', monitor='val_loss',
                             save_best_only=True, save_weights_only=False,
                             mode='auto', save_freq='epoch', verbose=1)
        csv_logger = CSVLogger(csv_path, append=False)

        model = AI.Model(sample_rate=self.sample_rate, delta_time=self.delta_time, n_fdt=self.n_fdt,
                         number_of_classes=self.number_of_classes).LSTM_Model()

        model.fit(tg, validation_data=vg,
                  epochs=30, verbose=1, workers=2,
                  callbacks=[csv_logger, cp])

    def predict(self, data):
        None


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, wav_paths, labels, sr, dt, n_classes,
                 batch_size=32, shuffle=True):
        self.wav_paths = wav_paths
        self.labels = labels
        self.sr = sr
        self.dt = dt
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.wav_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        wav_paths = [self.wav_paths[k] for k in indexes]
        labels = [self.labels[k] for k in indexes]

        # generate a batch of time data
        X = np.empty((self.batch_size, 1, int(self.sr * self.dt)), dtype=np.float32)
        Y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)

        for i, (path, label) in enumerate(zip(wav_paths, labels)):
            rate, wav = wavfile.read(path)
            X[i,] = wav.reshape(1, -1)
            Y[i,] = to_categorical(label, num_classes=self.n_classes)

        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.wav_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)
