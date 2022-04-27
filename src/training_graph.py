import torch
import torch.nn.functional as F
import torch.optim as optim
import pickle as pkl
import os

import time
import gzip
import numpy as np
import torch.optim as optim
from dataset.reddit_user_dataset import RedditUserDataset

from argparse import ArgumentParser
import datetime
import time
from os.path import  join
import json
from tqdm import tqdm
from utils.metrics import *
from utils.utils import *
from utils.loss_fct import *
from utils.train_utils import save_checkpoint
from model import GatClassification
from constants import *
import sys


def loss_fn(output, targets, samples_per_cls, no_of_classes=2):
    beta = 0.9999
    gamma = 2.0
    loss_type = "softmax"

    return CB_loss(targets, output, samples_per_cls, no_of_classes, loss_type, beta, gamma)

def get_samples_per_class(labels):
    return torch.bincount(labels).tolist()

def merge_samples(samples):
    for sample in tqdm(samples):
        temp = torch.mean(sample.features, dim=0, keepdim=True)

        temp_tuples = set()
        source = []
        target = []

        for graph_data in sample.graph_data:
            for i in range(len(graph_data[0])):
                connection = (graph_data[0][i], graph_data[1][i])

                if connection not in temp_tuples:
                    source.append(connection[0])
                    target.append(connection[1])
                    temp_tuples.add(connection)

        edges = torch.cat([torch.tensor(source).unsqueeze(-1), torch.tensor(target).unsqueeze(-1)], dim=1).permute(1, 0)
        sample.window = 1
        sample.graph_data = [edges]
        sample.features = temp

def train(epoch):
    #@TODO: Split both loops into train and evaluation into the model.
    t = time.time()
    model.train()
    #optimizer.zero_grad() #TODO: This is not inside the samples loop. Are gradients supposed to be accumulated? I think it is a bug.

    acc_train = []
    accuracy_val = []
    accuracy_test = []
    losses_train = []
    losses_val = []
    losses_test = []

    for sample in train_samples:  #
        train_sample = None
        if args.lazy_loading:
            train_sample = pkl.load(gzip.open(sample, 'rb'))
        else:
            train_sample = sample
        start = time.time()
        optimizer.zero_grad()
        train_features = train_sample.features.to(DEVICE)
        train_label = train_sample.labels.to(DEVICE)
        all_graphs = [graph.to(DEVICE) for graph in train_sample.graph_data]
        time_steps = train_sample.window
        adj = train_sample.adj
        end = time.time()
        compute = end - start
        start = time.time()
        output = model(all_graphs, train_features, time_steps, adj)
        #loss_train = F.nll_loss(output, train_label)
        loss_train = loss_fn(output, train_label,  get_samples_per_class(train_label))
        acc_train.append(accuracy(output, train_label).detach().cpu().numpy())
        loss_train.backward()
        optimizer.step()
        end = time.time()
        losses_train.append(np.mean(loss_train.detach().cpu().numpy()))

    model.eval()
    with torch.no_grad():
        for sample in val_samples:
            val_sample = None
            if args.lazy_loading:
                val_sample = pkl.load(gzip.open(sample, 'rb'))
            else:
                val_sample = sample
            val_features = val_sample.features.to(DEVICE)
            val_label = val_sample.labels.to(DEVICE)
            all_graphs = [graph.to(DEVICE) for graph in val_sample.graph_data]
            time_steps = val_sample.window
            adj = val_sample.adj

            output = model(all_graphs, val_features, time_steps, adj)
            #loss_val = F.nll_loss(output, val_label)
            loss_val = loss_fn(output, val_label,  get_samples_per_class(val_label))
            accuracy_val.append(accuracy(output, val_label).detach().cpu().numpy())
            losses_val.append(np.mean(loss_val.detach().cpu().numpy()))

    
        for sample in test_samples:
            test_sample = None
            if args.lazy_loading:
                test_sample = pkl.load(gzip.open(sample, 'rb'))
            else:
                test_sample = sample
            test_features = test_sample.features.to(DEVICE)
            test_label = test_sample.labels.to(DEVICE)
            all_graphs = [graph.to(DEVICE) for graph in test_sample.graph_data]
            time_steps = test_sample.window
            adj = test_sample.adj

            output = model(all_graphs, test_features, time_steps, adj)
            #loss_test = F.nll_loss(output, test_label)
            loss_test = loss_fn(output, test_label,  get_samples_per_class(test_label))
            accuracy_test.append(accuracy(output, test_label).detach().cpu().numpy())
            losses_test.append(np.mean(loss_test.detach().cpu().numpy()))

    metrics = {'train_acc': np.mean(np.array(acc_train)),
               'val_acc': np.mean(np.array(accuracy_val)),
               'test_acc': np.mean(np.array(accuracy_test)),
               'train_loss': float(np.mean(np.array(losses_train))),
               'val_loss': float(np.mean(np.array(losses_val))),
               'test_loss': float(np.mean(np.array(losses_test)))}

    print(metrics)
    return metrics

parser = ArgumentParser()
parser.add_argument("--max_epochs", dest="max_epochs", default=300, type=int)
parser.add_argument("--sample_dir", dest="sample_dir", type=str, required=True)
parser.add_argument("--checkpoint_dir", dest="checkpoint_dir", type=str, required=True)
parser.add_argument("--learning_rate", dest="learning_rate", default=5e-6, type=float)
parser.add_argument("--weight_decay", dest="weight_decay", default=1e-3, type=float)
parser.add_argument("--patience", dest="patience", default=10, type=int)
parser.add_argument("--run_id", dest="run_id", default='no_id_given')
parser.add_argument("--nhid_graph", dest="nhid_graph", default=64, type=int)
parser.add_argument("--nhid", dest="nhid", default=64, type=int)
parser.add_argument("--users_dim", dest="users_dim", default=768, type=int)
parser.add_argument("--result_dir", dest="result_dir", default="data/results", type=str)
parser.add_argument("--nheads", dest="nheads", default=8, type=int)
parser.add_argument("--dropout", dest="dropout", default=0.2, type=float)
parser.add_argument("--model_seed" , dest="model_seed", type=int, default=1234)
parser.add_argument("--comment", dest="comment", type=str, default="")
parser.add_argument("--lazy_loading", dest="lazy_loading", type=str2bool, required=False, default=False)
parser.add_argument("--gnn", dest="gnn", default='gat', type=str)


args = parser.parse_args()

###################################### Creation of model input data ####################################################

#descriptor = pkl.load(gzip.open(os.path.join(args.sample_dir, 'dataset_descriptor.data'), 'rb'))
descriptor = json.load(open(os.path.join(args.sample_dir, 'dataset_descriptor.json'), 'r'))

train_path = os.path.join(args.sample_dir, 'train_samples/')
n_train = descriptor['n_train_samples']
print(n_train)
val_path = os.path.join(args.sample_dir, 'val_samples/')
n_val = descriptor['n_val_samples']
print(n_val)
test_path = os.path.join(args.sample_dir, 'test_samples/')
n_test = descriptor['n_test_samples']
print(n_test)

########################################################################################################################

torch.manual_seed(SEED)
max_epochs = args.max_epochs
l_r = args.learning_rate
weight_decay = args.weight_decay
users_dim = args.users_dim
early_stopping_patience = args.patience
dropout = args.dropout
nheads = args.nheads
print("Weight decay: {}".format(weight_decay))
print("Learning rate: {}".format(l_r))

model = GatClassification(nfeat=users_dim, nhid_graph=args.nhid_graph, nhid=args.nhid, nclass=2, dropout=dropout,
                nheads=nheads, gnn_name=args.gnn).to(DEVICE)


optimizer = optim.Adam(model.parameters(),
                       lr=l_r,
                       weight_decay=weight_decay)

if args.lazy_loading:
    print('Lazy loading active...')
# Loading data
train_samples = []
for n in tqdm(range(n_train)):
    if args.lazy_loading:
        train_samples.append(join(train_path, 'sample_' + str(n) + '.data'))
    else:
        train_samples.append(pkl.load(gzip.open(join(train_path, 'sample_' + str(n) + '.data'), 'rb')))

val_samples = []
for n in range(n_val):
    if args.lazy_loading:
        val_samples.append(join(val_path, 'sample_' + str(n) + '.data'))
    else:
        val_samples.append(pkl.load(gzip.open(join(val_path, 'sample_' + str(n) + '.data'), 'rb')))

test_samples = []
for n in range(n_test):
    if args.lazy_loading:
        test_samples.append(join(test_path, 'sample_' + str(n) + '.data'))
    else:
        test_samples.append(pkl.load(gzip.open(join(test_path, 'sample_' + str(n) + '.data'), 'rb')))


top_val_acc = 0
no_improvement_epochs = 0
best_res = {}
conf = []
accs = {}
best_epoch_index = -1
train_preds = {}
val_preds = {}
test_preds = {}
best_model = None
TIMESTAMP = str(datetime.datetime.now()).replace(" ", "_").replace(".", ":")
best_filename = TIMESTAMP + '-best_model.tar'
checkpoint_dir = args.checkpoint_dir

for i in range(max_epochs):
    print('Iteration: ' + str(i))
    metrics = train(i)
    accs[i] = metrics
    if metrics['val_acc'] >= top_val_acc:
        top_val_acc = metrics['val_acc']
        no_improvement_epochs = 0
        best_res = metrics
        save_checkpoint({
            'epoch': i + 1,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'metrics': metrics}, 
            checkpoint=checkpoint_dir, name=best_filename
        )
        
    else:
        no_improvement_epochs += 1

    if no_improvement_epochs >= early_stopping_patience:
        print('Early stopping triggered')
        print('Best val results:')
        print(best_res)
        early_stopped = True
        break

bestModel_path = os.path.join(checkpoint_dir, best_filename)
checkpoint = torch.load(bestModel_path, map_location=DEVICE)
print("Checkpoint was in epoch {}".format(checkpoint['epoch']))
best_model = model
best_model.load_state_dict(checkpoint['state_dict'])
best_model.to(DEVICE)

best_model.eval()
accuracy_test = []
losses_test = []

with torch.no_grad():
    for sample in test_samples:
        test_sample = None
        if args.lazy_loading:
            test_sample = pkl.load(gzip.open(sample, 'rb'))
        else:
            test_sample = sample
        test_features = test_sample.features.to(DEVICE)
        test_label = test_sample.labels.to(DEVICE)
        all_graphs = [graph.to(DEVICE) for graph in test_sample.graph_data]
        time_steps = test_sample.window
        adj = test_sample.adj

        output = best_model(all_graphs, test_features, time_steps, adj)
        loss_test = F.nll_loss(output, test_label)
        accuracy_test.append(accuracy(output, test_label).detach().cpu().numpy())
        losses_test.append(np.mean(loss_test.detach().cpu().numpy()))
        gold =output.max(1)[1].type_as(test_label).detach().cpu().numpy()
        test_metrics = print_metrics(gold, test_label.cpu().numpy())
        

test_accuracy = np.mean(np.array(accuracy_test))
test_loss = float(np.mean(np.array(losses_test)))

best_res['test_loss'] = test_loss
best_res['test_acc'] = test_accuracy

conf = calc_confusion_matrices(best_model, train_samples, val_samples, test_samples)
train_preds, val_preds, test_preds = calc_prediction_map(best_model, train_samples, val_samples, test_samples)


train_accs_flattened = []
val_accs_flattened = []
test_accs_flattened = []

train_losses_flattened = []
val_losses_flattened = []
test_losses_flattened = []

for timestep, metric_dict in accs.items():
    train_accs_flattened.append(metric_dict['train_acc'])
    val_accs_flattened.append(metric_dict['val_acc'])
    test_accs_flattened.append(metric_dict['test_acc'])
    train_losses_flattened.append(metric_dict['train_loss'])
    val_losses_flattened.append(metric_dict['val_loss'])
    test_losses_flattened.append(metric_dict['test_loss'])

id = args.run_id + "_" + TIMESTAMP

result_obj = {'id': id}
result_obj['type'] = type(model).__name__
result_obj['timestamp'] = TIMESTAMP
result_obj['epochs'] = i
result_obj['learning_rate'] = l_r
result_obj['weight_decay'] = weight_decay
result_obj['graph_nhid'] = args.nhid_graph
result_obj['nhid'] = args.nhid
result_obj['dropout'] = dropout
result_obj['nheads'] = nheads
result_obj['model_seed'] = SEED
result_obj['max_epochs'] = max_epochs
result_obj['patience'] = early_stopping_patience
result_obj['gnn_type'] = args.gnn
result_obj['gnn'] = args.gnn
result_obj['best_model_path'] = bestModel_path
result_obj['test_metrics'] = test_metrics

for key, val in descriptor.items():
    if isinstance(val, dict):
        result_obj[key] = {k: convert_value(v) for k,v in val.items()}
    else:
        result_obj[key] = val

result_obj['samples_dir'] = args.sample_dir

best_res['epoch'] = best_epoch_index
result_obj['best_epoch'] = best_res

result_obj['hist_train_acc'] = train_accs_flattened
result_obj['hist_val_acc'] = val_accs_flattened
result_obj['hist_test_acc'] = test_accs_flattened
result_obj['hist_train_loss'] = train_losses_flattened
result_obj['hist_val_loss'] = val_losses_flattened
result_obj['hist_test_loss'] = test_losses_flattened

result_obj['train_conf'] = str(conf[0]).replace('\n', '')
result_obj['val_conf'] = str(conf[1]).replace('\n', '')
result_obj['test_conf'] = str(conf[2]).replace('\n', '')
result_obj['comment'] = args.comment

result_obj['train_predictions'] = train_preds
result_obj['val_predictions'] = val_preds
result_obj['test_predictions'] = test_preds

#TODO: Replace run_id + TIMESTAMP with id ? 
res_file = os.path.join(args.result_dir, args.run_id + TIMESTAMP + ".json")

with open(res_file, mode='w') as f:
    f.write(json.dumps(result_obj, indent=2))

