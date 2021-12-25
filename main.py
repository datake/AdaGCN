import logging
from ppnp.pytorch import PPNP
from ppnp.pytorch.training import train_model, train_model_AdaGCN
from ppnp.pytorch.earlystopping import stopping_args
from ppnp.pytorch.propagation import PPRExact, PPRPowerIteration
from ppnp.data.io import load_dataset
import pandas as pd
import seaborn as sns
import time
import argparse
import numpy as np
import torch
from models import GCN, AdaGCN

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='citeseer', help='cora_ml, citeseer, pubmed, ms_academic')
parser.add_argument('--model', type=str, default='AdaGCN', help='GCN, PPNP, APPNP, AdaGCN')
parser.add_argument('--trainsize', type=int, default=20, help='each class.')
parser.add_argument('--test', type=int, default=1, help='test or not')
parser.add_argument('--niter', type=int, default=1, help='iteration per seed')
parser.add_argument('--nseed', type=int, default=5, help='number of seeds')
# AdaGCN
parser.add_argument('--layers', type=int, default=2, help='Number of layers.')
parser.add_argument('--hid_AdaGCN', type=int, default=20, help='Number of hidden units. default:800')
parser.add_argument('--lr_AdaGCN', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for optimizer(all layers).')
parser.add_argument('--reg', type=float, default=5e-3, help='Weight decay on the 1st layer.')
parser.add_argument('--dropoutadj_AdaGCN', type=float, default=0.0, help='mixed dropout for adj in AdaGCN.')
parser.add_argument('--dropoutadj_GCN', type=float, default=0.0, help='mixed dropout for adj in GCN.')
parser.add_argument('--dropout', type=float, default=0.0, help='ordinary dropout for GCN')
parser.add_argument('--max', type=int, default=500, help='max epoch in early stopping')
parser.add_argument('--patience', type=int, default=300, help='patience in early stopping')
parser.add_argument('--early', type=int, default=0, help='whether early stopping is used')
args = parser.parse_args()
print(args)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
stopping_args['max_epochs'] = args.max
stopping_args['patience'] = args.patience
EARLY = True if args.early == 1 else False
logging.basicConfig(
        format='%(asctime)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO)

# (1) load data
graph_name = args.dataset
graph = load_dataset(graph_name)
graph.standardize(select_lcc=True)

# (2) define models: PPNP, APPNP, f: H —> Z
ALPHA = 0.2 if graph_name == 'ms_academic' else 0.1
if args.model in ['PPNP', 'APPNP']:
    model_class0 = PPNP
elif args.model == 'GCN':
    model_class0 = GCN
else:
    model_class0 = AdaGCN
if args.model == 'PPNP':
    model_prop = PPRExact(graph.adj_matrix, alpha=ALPHA)
elif args.model == 'APPNP':
    model_prop = PPRPowerIteration(graph.adj_matrix, alpha=ALPHA, niter=10) # alpha=0.2 for MS
else:
    model_prop = None

# (3) train parameters
NKNOW = 5000 if graph_name == 'ms_academic' else 1500
idx_split_args = {'ntrain_per_class': args.trainsize, 'nstopping': 500, 'nknown': NKNOW, 'seed': 1} # seed: 2413340114
reg_lambda = args.reg
learning_rate = 0.01

model_args = {
    'hiddenunits': [64],
    'drop_prob': args.dropout,
    'propagation': model_prop,  # propagation involves sparse Tensor\
    'lr': learning_rate if args.model == 'GCN' else args.lr_AdaGCN,
    'hid_AdaGCN': args.hid_AdaGCN,
    'layers': args.layers,
    'dropoutadj_GCN': args.dropoutadj_GCN,
    'dropoutadj_AdaGCN': args.dropoutadj_AdaGCN,
    'weight_decay': args.weight_decay,
}

test = True if args.test == 1 else False

# (4)set seeds
test_seeds = [
        2144199730,  794209841, 2985733717, 2282690970, 1901557222,
        2009332812, 2266730407,  635625077, 3538425002,  960893189,
        497096336, 3940842554, 3594628340,  948012117, 3305901371,
        3644534211, 2297033685, 4092258879, 2590091101, 1694925034]
val_seeds = [
        2413340114, 3258769933, 1789234713, 2222151463, 2813247115,
        1920426428, 4272044734, 2092442742, 841404887, 2188879532,
        646784207, 1633698412, 2256863076,  374355442,  289680769,
        4281139389, 4263036964,  900418539,  119332950, 1628837138]

if test:
    seeds = test_seeds[:args.nseed]
else:
    seeds = val_seeds[:args.nseed]

# (5) train
niter_per_seed = args.niter # random splitting for each seed, default 5
save_result = False
print_interval = 100

results = []
used_seeds = []
niter_tot = niter_per_seed * len(seeds) # 5 * 20
i_tot = 0
for seed in seeds:
    idx_split_args['seed'] = seed # split depends on seed
    for _ in range(niter_per_seed):
        i_tot += 1
        logging_string = f"Iteration {i_tot} of {niter_tot}"
        logging.log(22, logging_string + "\n                     " + '-' * len(logging_string))
        # train model
        if args.model == 'AdaGCN':
            result = train_model_AdaGCN(graph_name, args.model, model_class0, graph, model_args, learning_rate, reg_lambda,
                                        idx_split_args, stopping_args, test, device, None, print_interval, EARLY)
        else:
            result = train_model(graph_name, args.model, model_class0, graph, model_args, learning_rate, reg_lambda,
                                 idx_split_args, stopping_args, test, device, None, print_interval, EARLY)
        # return results
        results.append({})
        results[-1]['stopping_accuracy'] = result['early_stopping']['accuracy']
        results[-1]['stopping_f1_score'] = result['early_stopping']['f1_score']
        results[-1]['valtest_accuracy'] = result['valtest']['accuracy']
        results[-1]['valtest_f1_score'] = result['valtest']['f1_score']
        results[-1]['runtime'] = result['runtime']
        results[-1]['runtime_perepoch'] = result['runtime_perepoch']
        results[-1]['split_seed'] = seed
#
# (6) evaluation
result_df = pd.DataFrame(results)
result_df.head()

def calc_uncertainty(values: np.ndarray, n_boot: int = 1000, ci: int = 95) -> dict:
    stats = {}
    stats['mean'] = values.mean()
    boots_series = sns.algorithms.bootstrap(values, func=np.mean, n_boot=n_boot)
    stats['CI'] = sns.utils.ci(boots_series, ci)
    stats['uncertainty'] = np.max(np.abs(stats['CI'] - stats['mean']))
    return stats

stopping_acc = calc_uncertainty(result_df['stopping_accuracy'])
stopping_f1 = calc_uncertainty(result_df['stopping_f1_score'])
valtest_acc = calc_uncertainty(result_df['valtest_accuracy'])
valtest_f1 = calc_uncertainty(result_df['valtest_f1_score'])
runtime = calc_uncertainty(result_df['runtime'])
runtime_perepoch = calc_uncertainty(result_df['runtime_perepoch'])

print("{}\n"
      "Early stopping: Accuracy: {:.2f} ± {:.2f}%, "
      "F1 score: {:.4f} ± {:.4f}\n"
      "{}: Accuracy: {:.2f} ± {:.2f}%, "
      "F1 score: {:.4f} ± {:.4f}\n"
      "Runtime: {:.3f} ± {:.3f} sec, per epoch: {:.2f} ± {:.2f}ms"
      .format(
          args.model,
          stopping_acc['mean'] * 100,
          stopping_acc['uncertainty'] * 100,
          stopping_f1['mean'],
          stopping_f1['uncertainty'],
          'Test' if test else 'Validation',
          valtest_acc['mean'] * 100,
          valtest_acc['uncertainty'] * 100,
          valtest_f1['mean'],
          valtest_f1['uncertainty'],
          runtime['mean'],
          runtime['uncertainty'],
          runtime_perepoch['mean'] * 1e3,
          runtime_perepoch['uncertainty'] * 1e3,
      ))

for i in result_df['valtest_accuracy']:
    print('{:.6f}'.format(i), end=',')
print()
