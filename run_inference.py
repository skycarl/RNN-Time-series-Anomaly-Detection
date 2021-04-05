"""Runs inference on the specified dataset
"""

import sys
import os
import glob
import argparse
from predict import main


run_parser = argparse.ArgumentParser(description='Run inference')
run_parser.add_argument('--data', type=str, default='ecg',
                        help='type of the dataset (ecg, gesture, power_demand, space_shuttle, respiration, nyc_taxi')
run_parser.add_argument('--device', type=str, default='cuda',
                        help='cuda or cpu')
run_args = run_parser.parse_args()


pkl_paths = glob.glob(os.path.join(f'dataset/{run_args.data}/labeled/test', '*.pkl'))
pkl_files = list(map(os.path.basename, pkl_paths))

for pkl in pkl_files:
    print(f'----- Running inference on {pkl} -----')
    sys.argv.extend(['--data', run_args.data, '--filename', pkl, '--save_fig', '--device', run_args.device])
    main()
