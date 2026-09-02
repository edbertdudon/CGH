"""
Same idea as fft_counter.py, ported to torch so GPU runs also report an
exact, measured FFT count instead of an assumed one.
"""
import torch


class FFTCounterTorch:
    def __init__(self):
        self.count = 0

    def reset(self):
        self.count = 0

    def fft2(self, x):
        self.count += 1
        return torch.fft.fft2(x)

    def ifft2(self, x):
        self.count += 1
        return torch.fft.ifft2(x)


counter = FFTCounterTorch()
