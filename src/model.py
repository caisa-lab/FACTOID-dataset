import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from constants import *


class GatClassification(nn.Module):
    # nfeat -> number of features in each node
    # nhid -> number of output features of the GAT
    # nhead-> Number of attention heads
    # window -> window window length.
    def __init__(self, nfeat, nhid_graph, nhid, nclass, dropout, nheads, gnn_name='gat'):
        super(GatClassification, self).__init__()
        self.gnn_name = gnn_name
        self.nhid_graph = nhid_graph
        self.gnn = GATConv(nfeat, nhid_graph, heads=nheads, negative_slope=0.2, concat=False, dropout=dropout)
       
        self.linear_pass = nn.Linear(nhid_graph, nhid)
        
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(nhid, nclass)


    def forward(self, graph, fts, time_steps, adj=None):
        # graph-> list of graphs, 1 per timestep. Each graph in edge list format.
        # fts-> (window,Number of nodes, number of features in each node)
        # Please ensure all nodes are present in each graph. We can feed a zero vector for nodes that do not have any features on a particular time-step.
        y_full = []
        
        for i in range(time_steps):
            x = fts[i]
            G = graph[i]
            y = F.leaky_relu(self.gnn(x, G), 0.2)
            y_full.append(y.reshape(1, x.shape[0], self.nhid_graph))
        y = torch.cat(y_full)
    

        output = self.linear_pass(y)
        output = self.dropout(F.leaky_relu(output.reshape((fts.shape[1], -1)), 0.2))
     
        output = self.linear(output)
        return F.log_softmax(output, dim=1)
