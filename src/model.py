import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from constants import *

class Attention(nn.Module):

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        mix = torch.bmm(attention_weights, context)

        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights


class gru(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(gru, self).__init__()
        self.gru1 = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=False)

    def forward(self, inputs):
        full, last = self.gru1(inputs)
        return full, last

class lstm(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(lstm, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=False)

    def forward(self, inputs):
        context, (hidden, query) = self.lstm(inputs)
        return context, (hidden, query)

        
class ASTGPOLS(nn.Module):
    # nfeat -> number of features in each node
    # nhid -> number of output features of the GAT
    # nhead-> Number of attention heads
    # window -> window window length.

    def __init__(self, nfeat, nhid_graph, nhid_gru, nclass, dropout, nheads, graph_layer=True, rnn_layer='gru', att_layer=True, gnn_name='gat'):
        super(ASTGPOLS, self).__init__()
        self.gnn_name = gnn_name
        if graph_layer:
            print("Initializing the model with graph layer")
            self.nhid_graph = nhid_graph
            if gnn_name == 'gat':
                self.gnn = GATConv(nfeat, nhid_graph, heads=nheads, negative_slope=0.2, concat=False, dropout=dropout)
            else:
                self.gnn = GCNConv(nfeat, nhid_graph)
        else:
            print("Initializing the model without the graph layer")
            nhid_graph = nfeat
            self.nhid_graph = nhid_graph

        self.nhid_gru = nhid_gru
        self.att_layer = att_layer
        self.attention = Attention(nhid_gru)
        self.rnn_layer = rnn_layer
        
        if rnn_layer.lower() == 'lstm':
            print("Initializing the model with LSTM layer")
            self.rnn = lstm(nhid_graph, nhid_gru)
        elif rnn_layer.lower() == 'gru':
            print("Initializing the model with GRU layer")
            self.rnn = gru(nhid_graph, nhid_gru)
        elif rnn_layer.lower() == 'linear':
            print("Initializing the model with Linear layer")
            self.rnn = nn.Linear(nhid_graph, nhid_gru)
        else:
            print("The rnn layer has to be gru or lstm not {}".format(rnn_layer))
        
        self.linear_seq = nn.Linear(nhid_graph, nhid_gru)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(nhid_gru, nclass)
        self.graph_layer = graph_layer

    def forward(self, graph, fts, time_steps, adj=None):
        # graph-> list of graphs, 1 per timestep. Each graph in edge list format.
        # fts-> (window,Number of nodes, number of features in each node)
        # Please ensure all nodes are present in each graph. We can feed a zero vector for nodes that do not have any features on a particular time-step.
        
        y_full = []
        
        if self.graph_layer:
            for i in range(time_steps):
                x = fts[i]
                G = graph[i]
                y = F.leaky_relu(self.gnn(x, G), 0.2)
                y_full.append(y.reshape(1, x.shape[0], self.nhid_graph))
            y = torch.cat(y_full)
        else:
            y = fts

        if self.rnn_layer.lower() == 'linear':        
            output = self.rnn(y)
            output = self.dropout(F.leaky_relu(output.reshape((fts.shape[1], -1)), 0.2))
        else:
            context, query = self.rnn(y)

            if type(query) == tuple:
                _, query = query
                
            query = query.permute(1, 0, 2)
            context = context.permute(1, 0, 2)
                    
            if self.att_layer:
                output, _ = self.attention(query, context)
            else:
                output = query
        
            output = F.leaky_relu(output.reshape((fts.shape[1], self.nhid_gru)), 0.2)
            
        output = self.linear(output)
        return F.log_softmax(output, dim=1)
