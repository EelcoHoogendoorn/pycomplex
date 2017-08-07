
import os
import sys
import contextlib
import subprocess
import matplotlib.pyplot as plt


def save_animation(path, frames):
    os.mkdir(path)

    for i in range(frames):
        yield i
        plt.savefig(os.path.join(path, f'frame_{i}.png'))
        plt.close()
        print(f'saved frame {i}')
