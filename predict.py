import argparse
import torch
import pickle
import preprocess_data
from torch import optim
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from anomalyDetector import fit_norm_distribution_param
from anomalyDetector import anomalyScore
from anomalyDetector import get_precision_recall
import sys
import os

def main():
    """Run inference.""" 

    from model import model   

    parser = argparse.ArgumentParser(description='PyTorch RNN Anomaly Detection Model')
    parser.add_argument('--prediction_window_size', type=int, default=10,
                        help='prediction_window_size')
    parser.add_argument('--data', type=str, default='ecg',
                        help='type of the dataset (ecg, gesture, power_demand, space_shuttle, respiration, nyc_taxi')
    parser.add_argument('--filename', type=str, default='chfdb_chf13_45590.pkl',
                        help='filename of the dataset')
    parser.add_argument('--save_fig', action='store_true', default=True,
                        help='save results as figures')
    parser.add_argument('--compensate', action='store_true',
                        help='compensate anomaly score using anomaly score esimation')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='beta value for f-beta score')
    parser.add_argument('--device', type=str, default='cuda',
                        help='cuda or cpu')
    parser.add_argument('--save_str', type=str, default=None,
                        help='subdir that the saved checkpoint comes from')
    parser.add_argument('--res_str', type=str, default=None,
                        help='subdir to store results in')
    parser.add_argument('--session_type', type=str, default='both', choices=['train', 'infer', 'both'],
                            help='type session to run (train, infer, or both')
    parser.add_argument('--noise_ratio', type=float, default=0.05,
                        help='noise ratio (float between 0 and 1)')
    parser.add_argument('--noise_interval', type=float, default=0.0005,
                        help='noise interval')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')

    args_ = parser.parse_args()

    arg_path = Path('result', args_.data, args_.save_str)
    arg_path.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(arg_path, 'predict_args.txt'), 'w') as f:
        f.write('\n'.join(sys.argv[1:]))

    print('-' * 89)
    print("=> loading checkpoint ")

    if args_.save_str is None:
        load_str = str(Path('save',args_.data,'checkpoint', args_.filename).with_suffix('.pth'))
    else:
        load_str = str(Path('save', args_.data, args_.save_str, 'checkpoint', args_.filename).with_suffix('.pth'))
    
    if args_.device == 'cpu':
        checkpoint = torch.load(load_str, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(load_str)
    
    args = checkpoint['args']
    args.prediction_window_size= args_.prediction_window_size
    args.beta = args_.beta
    args.save_fig = args_.save_fig
    args.compensate = args_.compensate
    args.device = args_.device
    args.save_str = args_.save_str
    args.res_str = args_.res_str
    args.seed = args_.seed
    print("=> loaded checkpoint")

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    print(f'Inference seed = {args.seed}')

    ###############################################################################
    # Load data
    ###############################################################################
    TimeseriesData = preprocess_data.PickleDataLoad(data_type=args.data,filename=args.filename, augment_test_data=False)
    train_dataset = TimeseriesData.batchify(args,TimeseriesData.trainData[:TimeseriesData.length], bsz=1)
    test_dataset = TimeseriesData.batchify(args,TimeseriesData.testData, bsz=1)

    ###############################################################################
    # Build the model
    ###############################################################################
    nfeatures = TimeseriesData.trainData.size(-1)
    model = model.RNNPredictor(rnn_type = args.model,
                            enc_inp_size=nfeatures,
                            rnn_inp_size = args.emsize,
                            rnn_hid_size = args.nhid,
                            dec_out_size=nfeatures,
                            nlayers = args.nlayers,
                            res_connection=args.res_connection).to(args.device)
    model.load_state_dict(checkpoint['state_dict'])

    scores, predicted_scores, precisions, recalls, f_betas = list(), list(), list(), list(), list()
    targets, mean_predictions, oneStep_predictions, Nstep_predictions = list(), list(), list(), list()
    try:
        # For each channel in the dataset
        for channel_idx in range(nfeatures):
            ''' 1. Load mean and covariance if they are pre-calculated, if not calculate them. '''
            # Mean and covariance are calculated on train dataset.
            if 'means' in checkpoint.keys() and 'covs' in checkpoint.keys():
                print('=> loading pre-calculated mean and covariance')
                mean, cov = checkpoint['means'][channel_idx], checkpoint['covs'][channel_idx]
            else:
                print('=> calculating mean and covariance')
                mean, cov = fit_norm_distribution_param(args, model, train_dataset, channel_idx=channel_idx)

            ''' 2. Train anomaly score predictor using support vector regression (SVR). (Optional) '''
            # An anomaly score predictor is trained
            # given hidden layer output and the corresponding anomaly score on train dataset.
            # Predicted anomaly scores on test dataset can be used for the baseline of the adaptive threshold.
            if args.compensate:
                print('=> training an SVR as anomaly score predictor')
                train_score, _, _, hiddens, _ = anomalyScore(args, model, train_dataset, mean, cov, channel_idx=channel_idx)
                score_predictor = GridSearchCV(SVR(), cv=5,param_grid={"C": [1e0, 1e1, 1e2],"gamma": np.logspace(-1, 1, 3)})
                score_predictor.fit(torch.cat(hiddens,dim=0).numpy(), train_score.cpu().numpy())
            else:
                score_predictor=None

            ''' 3. Calculate anomaly scores'''
            # Anomaly scores are calculated on the test dataset
            # given the mean and the covariance calculated on the train dataset
            print('=> calculating anomaly scores')
            score, sorted_prediction, sorted_error, _, predicted_score = anomalyScore(args, model, test_dataset, mean, cov,
                                                                                    score_predictor=score_predictor,
                                                                                    channel_idx=channel_idx)

            ''' 4. Evaluate the result '''
            # The obtained anomaly scores are evaluated by measuring precision, recall, and f_beta scores
            # The precision, recall, f_beta scores are are calculated repeatedly,
            # sampling the threshold from 1 to the maximum anomaly score value, either equidistantly or logarithmically.
            print('=> calculating precision, recall, and f_beta')
            precision, recall, f_beta = get_precision_recall(args, score, num_samples=1000, beta=args.beta,
                                                            label=TimeseriesData.testLabel.to(args.device))
            print('data: ',args.data,' filename: ',args.filename,
                ' f-beta (no compensation): ', f_beta.max().item(),' beta: ',args.beta)
            if args.compensate:
                precision, recall, f_beta = get_precision_recall(args, score, num_samples=1000, beta=args.beta,
                                                                label=TimeseriesData.testLabel.to(args.device),
                                                                predicted_score=predicted_score)
                print('data: ',args.data,' filename: ',args.filename,
                    ' f-beta    (compensation): ', f_beta.max().item(),' beta: ',args.beta)


            target = preprocess_data.reconstruct(test_dataset.cpu()[:, 0, channel_idx],
                                                TimeseriesData.mean[channel_idx],
                                                TimeseriesData.std[channel_idx]).numpy()
            mean_prediction = preprocess_data.reconstruct(sorted_prediction.mean(dim=1).cpu(),
                                                        TimeseriesData.mean[channel_idx],
                                                        TimeseriesData.std[channel_idx]).numpy()
            oneStep_prediction = preprocess_data.reconstruct(sorted_prediction[:, -1].cpu(),
                                                            TimeseriesData.mean[channel_idx],
                                                            TimeseriesData.std[channel_idx]).numpy()
            Nstep_prediction = preprocess_data.reconstruct(sorted_prediction[:, 0].cpu(),
                                                        TimeseriesData.mean[channel_idx],
                                                        TimeseriesData.std[channel_idx]).numpy()
            sorted_errors_mean = sorted_error.abs().mean(dim=1).cpu()
            sorted_errors_mean *= TimeseriesData.std[channel_idx]
            sorted_errors_mean = sorted_errors_mean.numpy()
            score = score.cpu()
            scores.append(score), targets.append(target), predicted_scores.append(predicted_score)
            mean_predictions.append(mean_prediction), oneStep_predictions.append(oneStep_prediction)
            Nstep_predictions.append(Nstep_prediction)
            precisions.append(precision), recalls.append(recall), f_betas.append(f_beta)


            if args.save_fig:
                if args.res_str is None:
                    save_dir = Path('result', args.data, args.filename).with_suffix('').joinpath('fig_detection')
                else:
                    save_dir = Path('result', args.data, args.save_str, args.res_str, args.filename).with_suffix('').joinpath('fig_detection')
                save_dir.mkdir(parents=True,exist_ok=True)
                plt.plot(precision.cpu().numpy(),label='precision')
                plt.plot(recall.cpu().numpy(),label='recall')
                plt.plot(f_beta.cpu().numpy(), label='f1')
                plt.legend()
                plt.xlabel('Threshold (log scale)')
                plt.ylabel('Value')
                plt.title('Anomaly Detection on ' + args.data + ' Dataset', fontsize=18, fontweight='bold')
                plt.savefig(str(save_dir.joinpath('fig_f_beta_channel'+str(channel_idx)).with_suffix('.png')))
                plt.close()


                fig, ax1 = plt.subplots(figsize=(15,5))
                ax1.plot(target,label='Target',
                        color='black',  marker='.', linestyle='--', markersize=1, linewidth=0.5)
                ax1.plot(mean_prediction, label='Mean predictions',
                        color='purple', marker='.', linestyle='--', markersize=1, linewidth=0.5)
                ax1.plot(oneStep_prediction, label='1-step predictions',
                        color='green', marker='.', linestyle='--', markersize=1, linewidth=0.5)
                ax1.plot(Nstep_prediction, label=str(args.prediction_window_size) + '-step predictions',
                        color='blue', marker='.', linestyle='--', markersize=1, linewidth=0.5)
                ax1.plot(sorted_errors_mean,label='Absolute mean prediction errors',
                        color='orange', marker='.', linestyle='--', markersize=1, linewidth=1.0)
                ax1.legend(loc='upper left')
                ax1.set_ylabel('Value',fontsize=15)
                ax1.set_xlabel('Index',fontsize=15)
                ax2 = ax1.twinx()
                ax2.plot(score.numpy().reshape(-1, 1), label='Anomaly scores from \nmultivariate normal distribution',
                        color='red', marker='.', linestyle='--', markersize=1, linewidth=1)
                if args.compensate:
                    ax2.plot(predicted_score, label='Predicted anomaly scores from SVR',
                            color='cyan', marker='.', linestyle='--', markersize=1, linewidth=1)
                ax2.legend(loc='upper right')
                ax2.set_ylabel('anomaly score',fontsize=15)
                plt.title('Anomaly Detection on ' + args.data + ' Dataset', fontsize=18, fontweight='bold')
                plt.tight_layout()
                plt.xlim([0,len(test_dataset)])
                plt.savefig(str(save_dir.joinpath('fig_scores_channel'+str(channel_idx)).with_suffix('.png')))
                plt.close()

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from inference early')


    print('=> saving the results as pickle extensions')
    if args.res_str is None:
        save_dir = Path('result', args.data, args.filename).with_suffix('')
    else:
        save_dir = Path('result', args.data, args.save_str, args.res_str, args.filename).with_suffix('')
    save_dir.mkdir(parents=True, exist_ok=True)
    pickle.dump(targets, open(str(save_dir.joinpath('target.pkl')),'wb'))
    pickle.dump(mean_predictions, open(str(save_dir.joinpath('mean_predictions.pkl')),'wb'))
    pickle.dump(oneStep_predictions, open(str(save_dir.joinpath('oneStep_predictions.pkl')),'wb'))
    pickle.dump(Nstep_predictions, open(str(save_dir.joinpath('Nstep_predictions.pkl')),'wb'))
    pickle.dump(scores, open(str(save_dir.joinpath('score.pkl')),'wb'))
    pickle.dump(predicted_scores, open(str(save_dir.joinpath('predicted_scores.pkl')),'wb'))
    pickle.dump(precisions, open(str(save_dir.joinpath('precision.pkl')),'wb'))
    pickle.dump(recalls, open(str(save_dir.joinpath('recall.pkl')),'wb'))
    pickle.dump(f_betas, open(str(save_dir.joinpath('f_beta.pkl')),'wb'))
    print('-' * 89)

if __name__ == '__main__':
    main()
