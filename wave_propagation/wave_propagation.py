import numpy as np
import cv2


class WavePropagation:
    def __init__(
        self,
        grid: np.ndarray,
        time: np.ndarray,
        velocity: np.ndarray,
        transducer_pos: list,
        source: np.ndarray,
    ):
        self.G = grid
        self.t = time
        self.C = velocity
        self.tp = transducer_pos
        self.s = source

    def laplassian(self, i):
        G_pad = np.pad(self.G[i], (1,))
        return (
            G_pad[1:-1, 2:] + G_pad[1:-1, :-2] + G_pad[2:, 1:-1] + G_pad[:-2, 1:-1]
        ) - 4 * self.G[i]

    def simulate(self) -> np.ndarray:
        for i in self.t[:-1]:
            self.G[i + 1] = (
                self.laplassian(i) * np.power(self.C, 2)
                + 2 * self.G[i]
                - self.G[i - 1]
                + self.s[i, :]
            )

        return self.G

    def a_scan(self) -> np.ndarray:
        return self.G[:, self.tp[0], self.tp[1]]

    def export_video(self, name: str, fps: int, gating: int):
        normalized_grid = (
            (self.G - np.min(self.G)) * 255 / (np.max(self.G) - np.min(self.G))
        )
        frames = cv2.convertScaleAbs(normalized_grid)

        out = cv2.VideoWriter(
            name,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            self.G.shape[1:],
            isColor=False,
        )

        for frame in frames[gating:]:
            out.write(frame)
        out.release()


def sinusoidal_source(freq, t):
    return np.cos(freq * 2 * np.pi * t)


if __name__ == "__main__":
    N = 120
    T = 1000
    c = 0.3

    G = np.zeros((T, N, N))
    C = np.full((N, N), c)
    t = np.arange(T)
    s = np.zeros_like(G)

    mid = int(N / 2)
    freq = 1 / 50
    sinusoidal = sinusoidal_source(freq, t)
    s[:, mid, mid] = sinusoidal
    simulation = WavePropagation(G, t, C, [mid, mid], s)
    simulation.simulate()
    a_scan = simulation.a_scan()
    import matplotlib.pyplot as plt

    plt.plot(t, a_scan)
    plt.show()
    simulation.export_video("kaka.mp4", 24, 0)
