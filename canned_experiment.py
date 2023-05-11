from concurrent.futures import ThreadPoolExecutor
from wave_propagation import WavePropagation
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np


def _get_args():
    parser = ArgumentParser(
        prog="canned_experiment", description="Ultrasound propagation simulation"
    )
    parser.add_argument("output", help="B-scan output path")
    parser.add_argument(
        "--dry-run", action="store_true", help="Don't perform any simulation"
    )
    parser.add_argument("--mu", default=100, help="Gaussian mu parameter", type=float)
    parser.add_argument(
        "--sigma", default=4, help="Gaussian sigma parameter", type=float
    )
    parser.add_argument(
        "--show-source", help="Show gaussian source without delay", action="store_true"
    )
    parser.add_argument(
        "--show-setup", help="Show simulation setup", action="store_true"
    )
    parser.add_argument(
        "--preview", help="Show B-scan at the end of simulation", action="store_true"
    )
    parser.add_argument("--grid-size", default=300, help="Square grid size", type=int)
    parser.add_argument(
        "--time-limit", default=600, help="Instants of time count", type=int
    )
    parser.add_argument(
        "--transducer-count",
        default=100,
        help="Transducer array size. transducers are evenly distributed in a horizontal line",
        type=int,
    )
    parser.add_argument(
        "--transducer-offset",
        default=0.33,
        help="Transducer array y-axis offset. Percentage of N",
        type=float,
    )
    parser.add_argument(
        "--reflector-offset",
        default=0.5,
        help="Reflector array y-axis offset. Percentage of N",
        type=float,
    )
    parser.add_argument(
        "--period",
        default=40,
        help="Period of the gaussian source frequency",
        type=int,
    )
    parser.add_argument(
        "--velocity",
        default=0.35,
        help="Base wave propagation velocity. Should range from 0.2 to 0.5",
        type=float,
    )
    parser.add_argument(
        "--obstacle-damping",
        default=0,
        help="Propagation velocity through the obstacle. Percentage of VELOCITY",
        type=float,
    )
    parser.add_argument(
        "--activation-delay",
        default=0,
        help="transducer activation delay from left to right.",
        type=int,
    )
    parser.add_argument("--gating", default=200, help="Gating threshold", type=int)
    parser.add_argument(
        "--video-out",
        action="store_true",
        help="Generate simulation videos",
    )
    # parser.add_argument(
    #     "--fps",
    #     default=24,
    #     help="Simulation video FPS. Ignored if VIDEO-OUT is not provided",
    #     type=int,
    # )

    return parser.parse_args()


def gaussian_source(t, mu, sigma):
    return np.exp(-0.5 * ((t - mu) ** 2 / sigma**2) / (sigma * np.sqrt(2 * np.pi)))


def sinusoidal_source(t, freq):
    return np.cos(freq * 2 * np.pi * t)


if __name__ == "__main__":
    args = _get_args()

    N = args.grid_size
    T = args.time_limit
    Tc = args.transducer_count
    c = args.velocity
    damping = args.obstacle_damping
    delay = args.activation_delay
    transducer_y_offset = args.transducer_offset
    reflector_y_offset = args.reflector_offset
    mu = args.mu
    sigma = args.sigma

    # transducer x positions
    Tx = int(N / 4) + np.linspace(0, int((N - 1) / 2), num=Tc, dtype=int)
    mid = int(N / 2)
    half = int(Tc / 2)
    Tx = np.arange(mid - half, mid + half, 1)
    Ty = int(transducer_y_offset * N)  # transducer y positions
    t = np.arange(T)  # time axis
    delays = np.arange(Tc) * delay

    freq = 1 / args.period
    gaussian = gaussian_source(t, mu, sigma)
    sinusoidal = sinusoidal_source(t, freq)
    source = gaussian * sinusoidal

    def _source(t: int, coords: tuple[int, int]):
        s = np.zeros((N, N))
        s[coords] = source[t]
        return s

    if args.show_source:
        plt.plot(t, source)
        plt.show()

    simulations = []

    if not args.dry_run:
        gating_threshold = 0 if not args.gating else args.gating

        for i, delay in enumerate(delays):
            G = np.zeros((N, N))  # space grid
            C = np.full((N, N), c)  # velocities matrix
            s = np.zeros((T, N, N))

            mid = int(N / 2)
            C[mid, int(N * reflector_y_offset)] = damping * c

            simulation_source = np.hstack((np.zeros(delay), source))[:T]
            simulation = WavePropagation(
                G, T, C, (Ty, Tx[i]), _source, gating_threshold
            )
            simulations.append(simulation)

        B_scan = np.zeros((T - gating_threshold, Tc))

        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [
                executor.submit(
                    simulation.simulate, True if args.video_out and i == 0 else False
                )
                for i, simulation in enumerate(simulations)
            ]

            for i, future in enumerate(futures):
                future.result()
                B_scan[:, i] = simulations[i].a_scan

        np.save(args.output, B_scan)

        if args.preview:
            plt.imshow(np.log10(abs(B_scan) + 1), aspect="auto")
            plt.show()
