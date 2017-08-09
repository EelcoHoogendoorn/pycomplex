
import os
import sys
import contextlib
import subprocess
import matplotlib.pyplot as plt


def save_animation(path, frames, overwrite=False):
    if path is not None:
        try:
            os.mkdir(path)
        except:
            if not overwrite:
                raise

    for i in range(frames):
        yield i
        if path is not None:
            plt.savefig(os.path.join(path, f'frame_{i}.png'))
            plt.close()
            print(f'saved frame {i}')
