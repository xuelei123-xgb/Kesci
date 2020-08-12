
import numpy as np
import glob
import os

def time_shift_spectrogram(spectrogram,axis=0):
    """ Shift a spectrogram along the time axis in the spectral-domain at random
    """
    nb_cols = spectrogram.shape[axis]
    nb_shifts = np.random.randint(0, nb_cols)

    return np.roll(spectrogram, nb_shifts, axis=axis)



def same_class_augmentation(data1, data2):
    """ Perform same class augmentation of the wave by loading a random segment
    from the class_dir and additively combine the wave with that segment.
    """
    alpha = np.random.rand()
    wave = (1.0-alpha)*data1 + alpha*data2
    return wave

