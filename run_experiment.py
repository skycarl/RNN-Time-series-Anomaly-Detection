"""Runs training, inference, or both on the specified dataset
"""

import sys
import os
import glob
import argparse
from train import main as train_main
from predict import main as predict_main


run_parser = argparse.ArgumentParser(description='Run training, inference, or both')
run_parser.add_argument('--session_type', type=str, default='both', choices=['train', 'infer', 'both'],
                        help='type session to run (train, infer, or both')
run_parser.add_argument('--data', type=str, default='ecg', choices=['ecg', 'gesture', 'power_demand', 'space_shuttle', 'respiration', 'nyc_taxi'],
                        help='type of the dataset (ecg, gesture, power_demand, space_shuttle, respiration, nyc_taxi')
run_parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='cuda or cpu')
run_args = run_parser.parse_args()

# Run training, if specified
if run_args.session_type in ['train', 'both']:
    train_paths = glob.glob(os.path.join(f'dataset/{run_args.data}/labeled/train', '*.pkl'))
    train_files = list(map(os.path.basename, train_paths))

    for pkl in train_files:
        print(f'----- Training on {pkl} -----')
        sys.argv.extend(['--data', run_args.data, '--filename', pkl, '--save_fig', '--device', run_args.device])
        train_main()

# Run inference, if specified
if run_args.session_type in ['infer', 'both']:
    test_paths = glob.glob(os.path.join(f'dataset/{run_args.data}/labeled/test', '*.pkl'))
    test_files = list(map(os.path.basename, test_paths))
    
    for pkl in test_files:
        print(f'----- Running inference on {pkl} -----')
        sys.argv.extend(['--data', run_args.data, '--filename', pkl, '--save_fig', '--device', run_args.device])
        predict_main()
