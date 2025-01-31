import numpy as np
import pandas as pd
import torch 
import argparse
import os

from xgboost import XGBRegressor
from scipy.stats import ttest_1samp

import warnings
warnings.filterwarnings("ignore")

from data import PPCI

def get_parser():
    parser = argparse.ArgumentParser(description='ISTAnt')
    parser.add_argument('--sc', type=str, default="experiment_easy", help='Splitting criteria')
    #parser.add_argument('--video_length', type=int, default=10, help='Filter Video length')
    parser.add_argument('--num_epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--force', type=bool, default=False, help='Force training')
    parser.add_argument('--task', type=str, default="or", help='Task')
    return parser

def AIPW(dataset, treatment=2, control=1, pred=False):
    settings = pd.read_csv(f'{dataset.data_dir}/experiments_settings.csv')#.dropna()
    settings = settings[settings["Valid"]==1]
    settings['Position'] = settings.apply(lambda x: x["Experiment"][1], axis=1).astype(int)
    settings['Experiment'] = settings.apply(lambda x: ord(x["Experiment"][0]) - ord('a'), axis=1).astype(int)

    settings = settings[settings["Treatment"].isin([treatment, control])]
    T = settings["Treatment"].replace({treatment: 1, control: 0})
    Y = settings.apply(lambda x: dataset.supervised['Y_hat' if pred else "Y"][(dataset.supervised['source_data']["experiment"] == x["Experiment"]) & (dataset.supervised['source_data']["position"] == x["Position"])].sum().item(), axis=1)
    covariates = ["Position X", "Position Y", "Hour", "Date"]
    W = settings[covariates]
    if "Annotator" in settings.columns:
        W["Annotator"] = settings["Annotator"].astype('category').cat.codes
    W["Date"] = W["Date"].astype('category').cat.codes
    W = torch.tensor(W.values, dtype=torch.float32)
    T = torch.tensor(T.values, dtype=torch.float32).squeeze()
    Y = torch.tensor(Y.values, dtype=torch.float32).squeeze()

    ps = T.mean().item()
    N = len(T)
    model_outcome = XGBRegressor()
    model_outcome.fit(X = torch.cat((W, T.reshape(N, 1)), dim=1), y = Y)
    mu0 = model_outcome.predict(torch.cat((W, torch.zeros(N, 1)), dim=1)) #
    mu1 = model_outcome.predict(torch.cat((W, torch.ones(N, 1)), dim=1)) #Y[(T==1)[:,0]].mean().numpy()
    ite = mu1-mu0 + T.numpy() * (Y.numpy()-mu1) / ps - (1-T.numpy()) * (Y.numpy()-mu0) / (1-ps) 
    ATE = ite.mean()
    ATE_std = np.sqrt(ite.var()/N)
    p_value = ttest_1samp(ite, 0, alternative='greater')[1]
    return ATE, ATE_std, p_value

# def get_summary(dataset, video_length=40):
#     exps = set(np.array(dataset.supervised['source_data']["experiment"]))
#     poss = set(np.array(dataset.supervised['source_data']["position"]))
#     i = 0
#     df = pd.DataFrame(columns=['T', 'Z1', 'Z2', 'Y', 'Y_hat'])
#     for exp in exps:
#         for pos in poss:
#             mask = (dataset.supervised['source_data']["experiment"] == exp) & (dataset.supervised['source_data']["position"] == pos) & (dataset.supervised['source_data']["exp_minute"]<video_length)
#             if mask.sum() == 0:
#                 continue
#             df.loc[i] = {'T': dataset.supervised['source_data']["treatment"][mask][0].item(), 
#                               'Z1': exp, 
#                               'Z2': pos, 
#                               'Y': dataset.supervised['Y'][mask].sum().item(),
#                               'Y_hat': dataset.supervised["Y_hat"][mask].sum().item() if "Y_hat" in dataset.supervised.keys() else np.nan}
#             i += 1
#     return df

# def treatment_effect(df, C=1, T=2, verbose=False, exp="Reference"):
#     ATE = df[df['T'] == T]['Y'].mean() - df[df['T'] == C]['Y'].mean()
#     ATE_std = np.sqrt(df[df['T'] == T]['Y'].var()/df[df['T'] == T].shape[0] + df[df['T'] == C]['Y'].var()/df[df['T'] == C].shape[0])
#     PPATE = df[df['T'] == T]['Y_hat'].mean() - df[df['T'] == C]['Y_hat'].mean()
#     PPATE_std = np.sqrt(df[df['T'] == T]['Y_hat'].var()/df[df['T'] == T].shape[0] + df[df['T'] == C]['Y_hat'].var()/df[df['T'] == C].shape[0])
#     if verbose:
#         print(f"{exp} - ATE: {ATE:.2f} (std: {ATE_std:.2f}), PPATE: {PPATE:.2f} (std: {PPATE_std:.2f})")
#     return ATE, ATE_std, PPATE, PPATE_std

def accuracy(dataset, verbose=False, exp="Reference"):
    y = dataset.supervised["Y"]
    y_hat = np.round(dataset.supervised["Y_hat"])
    tp = np.array((y == 1) & (y_hat == 1)).sum()
    tn = np.array((y == 0) & (y_hat == 0)).sum()
    fp = np.array((y == 0) & (y_hat == 1)).sum()
    fn = np.array((y == 1) & (y_hat == 0)).sum()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bacc = (sensitivity + specificity) / 2
    acc = (tp + tn) / (tp + tn + fp + fn)
    if verbose:
        print(f"{exp} - Accuracy: {acc:.2f}, Balanced Accuracy: {bacc:.2f}")
    return acc, bacc

def main(args):
    print(f"Splitting criteria: {args.sc}")
    encoders = ["dino", "clip","clip_large","vit","vit_large"]
    results = pd.DataFrame(columns=['method','encoder','lr','seed','ATE_ref', 'ATE_std_ref', 'ATE_p_value_ref', 'PPATE_ref', 'PPATE_std_ref', 'PPATE_p_value_ref', 'ATE_tar', 'ATE_std_tar', 'ATE_p_value_tar', 'PPATE_tar', 'PPATE_std_tar', 'PPATE_p_value_tar', 'acc_ref', 'bacc_ref', 'acc_tar', 'bacc_tar'])
    i = 0
    for encoder in encoders:
        reference = PPCI(encoder = encoder,
                    token = "class",
                    task = args.task,
                    split_criteria = args.sc,
                    environment = "supervised",
                    batch_size = 256,
                    num_proc = 4,
                    verbose = True,
                    data_dir = 'data/istant_lq',
                    results_dir = 'results/istant_lq')
        target = PPCI(encoder = encoder,
                    token = "class",
                    task = args.task,
                    split_criteria = args.sc,
                    environment = "supervised",
                    batch_size = 256,
                    num_proc = 4,
                    verbose = True,
                    data_dir = 'data/istant_hq',
                    results_dir = 'results/istant_hq')

        for method in ["ERM","vREx","DERM"]:
            for lr in [0.005,0.0005, 0.00005]:
                for seed in range(1):
                    print(f"Encoder: {encoder}, Method: {method}, LR: {lr}, Seed: {seed}")
                    reference.train(add_pred_env="supervised", 
                            hidden_layers = 2,
                            hidden_nodes = 256,
                            batch_size = 256,
                            lr = lr,
                            seed = seed,
                            num_epochs = args.num_epochs,
                            save = True,
                            verbose = True,
                            force = args.force,
                            method = method)
                    target.results_dir = 'results/istant_lq'
                    target.train(add_pred_env="supervised", 
                                hidden_layers = 2,
                                hidden_nodes = 256,
                                batch_size = 256,
                                lr = lr,
                                seed = seed,
                                num_epochs = args.num_epochs,
                                save = False,
                                verbose = True,
                                method = method)
                    target.results_dir = 'results/istant_hq'
                    
                    # reference_df = get_summary(reference, video_length=args.video_length)
                    # target_df = get_summary(target, video_length=args.video_length)

                    acc_ref, bacc_ref = accuracy(reference, verbose=True, exp="Reference")
                    ATE_ref, ATE_std_ref, ATE_p_value_ref = AIPW(reference, pred=False)
                    PPATE_ref, PPATE_std_ref, PPATE_p_value_ref = AIPW(reference, pred=True)
                    
                    acc_tar, bacc_tar = accuracy(target, verbose=True, exp="Target")
                    ATE_tar, ATE_std_tar, ATE_p_value_tar = AIPW(target, pred=False)
                    PPATE_tar, PPATE_std_tar, PPATE_p_value_tar = AIPW(target, pred=True)

                    i += 1
                    results.loc[i] = [method, encoder, lr, seed, ATE_ref, ATE_std_ref, ATE_p_value_ref, PPATE_ref, PPATE_std_ref, PPATE_p_value_ref, ATE_tar, ATE_std_tar, ATE_p_value_tar, PPATE_tar, PPATE_std_tar, PPATE_p_value_tar, acc_ref, bacc_ref, acc_tar, bacc_tar]
        if not os.path.exists(f'results/generalization/{args.task}/'):
            os.makedirs(f'results/generalization/{args.task}')
        results.to_csv(f"results/generalization/{args.task}/{args.sc}.csv", index=False)

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)