"""Independent Python implementation of the specified standard MT19937-64 stream."""
import numpy as np

MASK = (1 << 64) - 1


def splitmix64(value):
    value = (value + 0x9E3779B97F4A7C15) & MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK
    return value ^ (value >> 31)


class MT64:
    def __init__(self, seed):
        self.state = [seed]
        for index in range(1, 312):
            previous = self.state[-1]
            self.state.append((6364136223846793005 * (previous ^ (previous >> 62)) + index) & MASK)
        self.index = 312

    def next(self):
        if self.index == 312:
            for i in range(312):
                x = (self.state[i] & 0xFFFFFFFF80000000) | (self.state[(i+1) % 312] & 0x7FFFFFFF)
                self.state[i] = self.state[(i+156) % 312] ^ (x >> 1) ^ (0xB5026F5AA96619E9 if x & 1 else 0)
            self.index = 0
        x = self.state[self.index]
        self.index += 1
        x ^= (x >> 29) & 0x5555555555555555
        x ^= (x << 17) & 0x71D67FFFEDA60000
        x ^= (x << 37) & 0xFFF7EEE000000000
        x ^= x >> 43
        return x


def probabilities(base, alpha, size, seed, stream, shot):
    generator = MT64(splitmix64(seed ^ splitmix64(shot) ^ splitmix64((stream + 0xD1B54A32D192ED03) & MASK)))
    output = np.tile(base, (size, 1))
    if alpha:
        for member in range(1, size):
            xi = np.asarray([2 * ((generator.next() >> 11) * 2**-53) - 1 for _ in base])
            output[member] = np.clip(base * (1 + alpha * xi), 1e-14, 1 - 1e-14)
    return output
