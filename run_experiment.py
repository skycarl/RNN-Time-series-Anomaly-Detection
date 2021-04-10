"""Run a single experiment for training, inference, or both on the specified dataset
"""

import sys
import os
import glob
import argparse
from train import main as train_main
from predict import main as predict_main
from pathlib import Path

def main():

    run_parser = argparse.ArgumentParser(description='Run training, inference, or both')
    run_parser.add_argument('--session_type', type=str, default='both', choices=['train', 'infer', 'both'],
                            help='type session to run (train, infer, or both')
    run_parser.add_argument('--data', type=str, default='ecg', choices=['ecg', 'gesture', 'power_demand', 'space_shuttle', 'respiration', 'nyc_taxi'],
                            help='type of the dataset (ecg, gesture, power_demand, space_shuttle, respiration, nyc_taxi')
    run_parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                            help='cuda or cpu')
    run_parser.add_argument('--noise_ratio', type=float, default=0.05,
                            help='noise ratio (float between 0 and 1)')
    run_parser.add_argument('--noise_interval', type=float, default=0.0005,
                            help='noise interval')
    run_parser.add_argument('--save_str', type=str, default=None,
                            help='subdir to store results in')
    run_args = run_parser.parse_args()

    # Run training, if specified
    if run_args.session_type in ['train', 'both']:
        arg_path = Path('save', run_args.save_str)
        arg_path.mkdir(parents=True, exist_ok=True)
        with open(os.path.join(arg_path, 'experiment_args.txt'), 'w') as f:
            f.write('\n'.join(sys.argv[1:]))

        train_paths = glob.glob(os.path.join(f'dataset/{run_args.data}/labeled/train', '*.pkl'))
        train_files = list(map(os.path.basename, train_paths))

        for pkl in train_files:
            print(f'----- Training on {pkl} -----')
            sys.argv.extend(['--filename', pkl])
            train_main()

            idx = sys.argv.index('--filename')
            del sys.argv[idx+1]
            del sys.argv[idx]

    # Run inference, if specified
    if run_args.session_type in ['infer', 'both']:
        test_paths = glob.glob(os.path.join(f'dataset/{run_args.data}/labeled/test', '*.pkl'))
        test_files = list(map(os.path.basename, test_paths))

        arg_path = Path('result', run_args.save_str)
        arg_path.mkdir(parents=True, exist_ok=True)
        with open(os.path.join(arg_path, 'experiment_args.txt'), 'w') as f:
            f.write('\n'.join(sys.argv[1:]))
        
        for pkl in test_files:
            print(f'----- Running inference on {pkl} -----')
            sys.argv.extend(['--filename', pkl])
            predict_main()

            idx = sys.argv.index('--filename')
            del sys.argv[idx+1]
            del sys.argv[idx]

if __name__ == '__main__':
    main()
