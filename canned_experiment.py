from argparse import ArgumentParser
import numpy as np


def _get_args():
    parser = ArgumentParser(
        prog='Canned ultrasound propagation simulation')
    parser.add_argument('output', metavar='O', action='store_const')

    return parser.parse_args()


def laplassian(G):
    G_pad = np.pad(G, (1,))
    return (G_pad[1:-1, 2:] + G_pad[1:-1, :-2] +
            G_pad[2:, 1:-1] + G_pad[:-2, 1:-1]) - 4 * G


def gaussian_source(t, mu, sigma):
    return np.exp(-0.5 * ((t - mu)**2 / sigma**2)
                  / (sigma * np.sqrt(2 * np.pi)))


def sinusoidal_source(t, freq):
    return np.cos(freq * 2 * np.pi * t)


if __name__ == "__main__":
    args = _get_args()
    output_file = args.output

    N = 300     # grid size
    T = 500     # time limit
    Tc = 32     # transdutor count
    c = 0.3     # baseline propagation velocity

    G = np.zeros((T, N, N))     # space grid
    C = np.full((N, N), c)      # velocities matrix
    t = np.arange(T)            # time axis
    s = np.zeros_like(G)        # sources matrix

    mid = int(N / 2)
    C[mid, mid] = 0.6 * c

    Tx = np.linspace(0, N, num=Tc, dtype=int)
    y = int(N/10)
    Ty = np.full(Tc, y)

    freq = 1/50
    gaussian = gaussian_source(t, 4 / freq, len(t) / 150)
    sinusoidal = sinusoidal_source(t, freq)

    s[:, Tx][Ty] = gaussian * sinusoidal

    for i in t[:-1]:
        G[i+1] = laplassian(G[i]) * np.power(C, 2) + 2 * G[i] - G[i-1] + s[i]

    np.save(output_file, G)
