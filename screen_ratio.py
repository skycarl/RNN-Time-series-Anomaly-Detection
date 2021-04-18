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
screen_args = screen_parser.parse_args()

ratios = np.linspace(0.0, 1, 11)

for rat in ratios:
    print(f'----- Running noise_ratio = {str(np.round(rat, 4))} -----')
    sys.argv.extend(['--noise_ratio', str(np.round(rat, 4)), '--save_str', f'nr_{str(np.round(rat, 4))}'])
    screen_main()

    idx = sys.argv.index('--noise_ratio')
    del sys.argv[idx+1]
    del sys.argv[idx]

    idx = sys.argv.index('--save_str')
    del sys.argv[idx+1]
    del sys.argv[idx]