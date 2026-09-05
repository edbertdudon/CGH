"""
Every 2D FFT / IFFT call in this project goes through this counter.

The point: the technical reference doc's "14 FFTs/frame" and "~1.49 GFLOP
per 2D FFT" numbers are *planning estimates*, not measurements. This module
makes it possible to replace the estimate with an exact count from a real
run -- how many FFT calls did it actually take to hit acceptable image
quality, for a given algorithm and depth complexity.
"""
import numpy as np


class FFTCounter:
    def __init__(self):
        self.count = 0
        self.log = []  # (op, shape) per call, useful for debugging

    def reset(self):
        self.count = 0
        self.log = []

    def fft2(self, x):
        self.count += 1
        self.log.append(("fft2", x.shape))
        return np.fft.fft2(x)

    def ifft2(self, x):
        self.count += 1
        self.log.append(("ifft2", x.shape))
        return np.fft.ifft2(x)


# Single shared instance used by propagation.py. Reset it at the start of
# any experiment you want an isolated count for.
counter = FFTCounter()
