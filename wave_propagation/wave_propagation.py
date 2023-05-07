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
        self.a_scan = np.zeros(t_max)

    def laplassian(self, i):
        G_pad = np.pad(self.G[i], (1,))
        return (
            G_pad[1:-1, 2:] + G_pad[1:-1, :-2] + G_pad[2:, 1:-1] + G_pad[:-2, 1:-1]
        ) - 4 * self.G[i]

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

            self.a_scan[i] = self.G[-1, self.tp[0], self.tp[1]]

            if video_out and i >= self.gating:
                frames[i - self.gating] = self.G[-1]

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
