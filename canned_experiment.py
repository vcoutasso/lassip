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
    parser.add_argument("--mu", default=200, help="Gaussian mu parameter", type=float)
    parser.add_argument(
        "--sigma", default=6.67, help="Gaussian sigma parameter", type=float
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
    parser.add_argument("--grid-size", default=120, help="Square grid size", type=int)
    parser.add_argument(
        "--time-limit", default=1000, help="Instants of time count", type=int
    )
    parser.add_argument(
        "--transducer-count",
        default=32,
        help="Transducer array size. transducers are evenly distributed in a horizontal line",
        type=int,
    )
    parser.add_argument(
        "--transducer-offset",
        default=0.2,
        help="Transducer array y-axis offset. Percentage of N",
        type=float,
    )
    parser.add_argument(
        "--frequency-reciprocal",
        default=50,
        help="Reciprocal of the gaussian source frequency",
        type=int,
    )
    parser.add_argument(
        "--velocity",
        default=0.3,
        help="Base wave propagation velocity. Should range from 0.2 to 0.5",
        type=float,
    )
    parser.add_argument(
        "--obstacle-damping",
        default=0.3,
        help="Propagation velocity through the obstacle. Percentage of VELOCITY",
        type=float,
    )
    parser.add_argument(
        "--activation-delay",
        default=0,
        help="transducer activation delay from left to right.",
        type=int,
    )
    parser.add_argument("--gating", help="Gating threshold", type=int)
    parser.add_argument(
        "--video-out",
        action="store_true",
        help="Generate simulation videos",
    )
    parser.add_argument(
        "--fps",
        default=24,
        help="Simulation video FPS. Ignored if VIDEO-OUT is not provided",
        type=int,
    )

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
    y_offset = args.transducer_offset
    mu = args.mu
    sigma = args.sigma

    Tx = int(N / 4) + np.linspace(
        0, int((N - 1) / 2), num=Tc, dtype=int
    )  # transducer x positions
    mid = int(N / 2)
    half = int(Tc / 2)
    Tx = np.arange(mid - half, mid + half, 1)
    Ty = int(y_offset * N)  # transducer y positions
    t = np.arange(T)  # time axis
    delays = np.arange(Tc) * delay

    freq = 1 / args.frequency_reciprocal
    gaussian = gaussian_source(t, mu, sigma)
    sinusoidal = sinusoidal_source(t, freq)
    source = gaussian * sinusoidal

    if args.show_source:
        plt.plot(t, source)
        plt.show()

    simulations = []

    if not args.dry_run:
        for i, delay in enumerate(delays):
            G = np.zeros((T, N, N))  # space grid
            t = np.arange(T)  # time axis
            C = np.full((N, N), c)  # velocities matrix
            s = np.zeros_like(G)

            mid = int(N / 2)
            C[mid, mid] = damping * c

            simulation_source = np.hstack((np.zeros(delay), source))[:T]
            s[:, Ty, Tx[i]] = simulation_source
            simulation = WavePropagation(G, t, C, [Ty, Tx[i]], s)
            simulations.append(simulation)

        gating_cutoff = 0 if not args.gating else args.gating
        B_scan = np.zeros((T - gating_cutoff, Tc))

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(simulation.simulate) for simulation in simulations
            ]

            for i, future in enumerate(futures):
                future.result()
                B_scan[:, i] = simulations[i].a_scan()[gating_cutoff:]

        np.save(args.output, B_scan)

        if args.preview:
            plt.imshow(B_scan, aspect="auto")
            plt.show()

        if args.video_out:
            with ThreadPoolExecutor(max_workers=Tc) as executor:
                filenames = [f"{args.output}{i}.mp4" for i in range(Tc)]
                futures = [
                    executor.submit(
                        simulation.export_video, filename, args.fps, gating_cutoff
                    )
                    for simulation, filename in zip(simulations, filenames)
                ]

                for i, future in enumerate(futures):
                    future.result()
