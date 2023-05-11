from collections.abc import Callable
import numpy as np
import cv2


class WavePropagation:
    def __init__(
        self,
        grid: np.ndarray,
        t_max: int,
        velocity: np.ndarray,
        transducer_pos: tuple[int, int],
        source: Callable[[int, tuple[int, int]], np.ndarray],
        gating: int = 350,
    ):
        if grid.ndim != 2:
            raise Exception("Grid must have two spatial dimensions.")

        self.G = np.tile(grid, (3, 1, 1))
        self.t = t_max
        self.C = velocity
        self.tp = transducer_pos
        self.s = source
        self.gating = gating
        self.a_scan = np.zeros(t_max - gating)

    def laplassian(self, i):
        G = self.G[i]
        lap = np.pad(np.zeros_like(G), (4,))

        lap[0:-8, 4:-4] -= 9 * G
        lap[1:-7, 4:-4] += 128 * G
        lap[2:-6, 4:-4] -= 1008 * G
        lap[3:-5, 4:-4] += 8064 * G
        lap[4:-4, 4:-4] -= 14350 * G
        lap[5:-3, 4:-4] += 8064 * G
        lap[6:-2, 4:-4] -= 1008 * G
        lap[7:-1, 4:-4] += 128 * G
        lap[8:, 4:-4] -= 9 * G

        lap[4:-4, 0:-8] -= 9 * G
        lap[4:-4, 1:-7] += 128 * G
        lap[4:-4, 2:-6] -= 1008 * G
        lap[4:-4, 3:-5] += 8064 * G
        lap[4:-4, 4:-4] -= 14350 * G
        lap[4:-4, 5:-3] += 8064 * G
        lap[4:-4, 6:-2] -= 1008 * G
        lap[4:-4, 7:-1] += 128 * G
        lap[4:-4, 8:] -= 9 * G

        return lap[4:-4, 4:-4] / 5040

    def simulate(self, video_out: bool) -> np.ndarray:
        if video_out:
            out = cv2.VideoWriter(
                "simulation.mp4",
                cv2.VideoWriter_fourcc(*"mp4v"),
                24,
                self.G.shape[1:],
                isColor=False,
            )

            frames = np.tile(self.G[0], (self.t - self.gating, 1, 1))

        for i in range(self.t):
            self.G[-1] = (
                self.laplassian(-2) * np.power(self.C, 2)
                + 2 * self.G[-2]
                - self.G[-3]
                + self.s(i, self.tp)
            )

            if i >= self.gating:
                idx = i - self.gating

                self.a_scan[idx] = self.G[-1, self.tp[0], self.tp[1]]

                if video_out:
                    frames[idx] = self.G[-1]

            self.G = np.roll(self.G, -1, axis=0)

        if video_out:
            self._write_frames(out, frames)

        return self.G

    def _write_frames(self, out, frames):
        normalized_grid = (
            (frames - np.min(frames)) * 255 / (np.max(frames) - np.min(frames))
        )

        for frame in cv2.convertScaleAbs(normalized_grid):
            out.write(frame)

        out.release()


def sinusoidal_source(freq, t):
    return np.cos(freq * 2 * np.pi * t)


if __name__ == "__main__":
    N = 120
    T = 1000
    c = 0.3

    G = np.zeros((N, N))
    C = np.full((N, N), c)

    mid = int(N / 2)

    def source(t: int, coords: tuple[int, int]):
        freq = 1 / 50
        s = np.zeros_like(G)
        s[coords] = sinusoidal_source(freq, t)
        return s

    simulation = WavePropagation(G, T, C, (mid, mid), source)
    simulation.simulate(video_out=True)
