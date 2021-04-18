"""Experiment runner for noise ratio
"""

import argparse
import numpy as np
import sys
from run_experiment import main as screen_main


screen_parser = argparse.ArgumentParser(description='Runs a screen on noise_ratio')
screen_parser.add_argument('--session_type', type=str, default='both', choices=['train', 'infer', 'both'],
                        help='type session to run (train, infer, or both')
screen_parser.add_argument('--data', type=str, default='ecg', choices=['ecg', 'gesture', 'power_demand', 'space_shuttle', 'respiration', 'nyc_taxi'],
                        help='type of the dataset (ecg, gesture, power_demand, space_shuttle, respiration, nyc_taxi')
screen_parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='cuda or cpu')
screen_parser.add_argument('--noise_interval', type=float, default=0.0005,
                        help='noise interval')
screen_parser.add_argument('--noise_ratio', type=float, default=0.05,
                            help='noise ratio (float between 0 and 1)')
screen_parser.add_argument('--runs', type=int, default='5', help='Number of runs')
screen_args = screen_parser.parse_args()

for n in range(int(screen_args.runs)):
    seed = np.random.randint(1, 10000000)

    print(f'----- Run {n} -----')
    sys.argv.extend(['--save_str', f'{screen_args.data}_run_{str(n)}',
                     '--seed', str(seed)])
    screen_main()

    idx = sys.argv.index('--save_str')
    del sys.argv[idx+1]
    del sys.argv[idx]

    idx = sys.argv.index('--seed')
    del sys.argv[idx+1]
    del sys.argv[idx]