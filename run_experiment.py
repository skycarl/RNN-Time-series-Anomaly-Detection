"""Runs training, inference, or both on the specified dataset
"""

import sys
import os
import glob
import argparse
from train import main as train_main
from predict import main as predict_main
from pathlib import Path


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
                        help='subdir in result/ to store results in')
run_args = run_parser.parse_args()

if run_args.session_type in ['train', 'both']:
    arg_path = Path('save', run_args.save_str)
    arg_path.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(arg_path, 'experiment_args.txt'), 'w') as f:
        f.write('\n'.join(sys.argv[1:]))

if run_args.session_type in ['infer', 'both']:
    arg_path = Path('result', run_args.save_str)
    arg_path.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(arg_path, 'experiment_args.txt'), 'w') as f:
        f.write('\n'.join(sys.argv[1:]))

# Run training, if specified
if run_args.session_type in ['train', 'both']:
    train_paths = glob.glob(os.path.join(f'dataset/{run_args.data}/labeled/train', '*.pkl'))
    train_files = list(map(os.path.basename, train_paths))

    for pkl in train_files:
        print(f'----- Training on {pkl} -----')
        sys.argv.extend(['--data', run_args.data, '--filename', pkl, '--save_fig', '--device', run_args.device,
                         '--save_str', run_args.save_str])
        train_main()

# Run inference, if specified
if run_args.session_type in ['infer', 'both']:
    test_paths = glob.glob(os.path.join(f'dataset/{run_args.data}/labeled/test', '*.pkl'))
    test_files = list(map(os.path.basename, test_paths))
    
    for pkl in test_files:
        print(f'----- Running inference on {pkl} -----')
        sys.argv.extend(['--data', run_args.data, '--filename', pkl, '--save_fig', '--device', run_args.device,
                         '--save_str', run_args.save_str])
        predict_main()
