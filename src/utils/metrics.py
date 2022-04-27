from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from constants import *


def print_metrics(gold, predictions):
    cm = confusion_matrix(gold, predictions)
    print(cm)
    f1Score_1 = f1_score(gold, predictions, average='macro')
    print("Total f1 score macro {:3f}: ".format(f1Score_1))
    f1Score_2 = f1_score(gold, predictions, average='micro')
    print("Total f1 score micro {:3f}:".format(f1Score_2))
    f1Score_3 = f1_score(gold, predictions, average='binary')
    print("Total f1 score binary {:3f}:".format(f1Score_3))
    f1Score_4 = f1_score(gold, predictions, average='weighted')
    print("Total f1 score weighted {:3f}:".format(f1Score_4))
    accuracy = accuracy_score(gold, predictions)
    print("Accuracy {:3f}:".format(accuracy))
    
    return (f1Score_1, f1Score_2, f1Score_3, f1Score_4, accuracy)
    
    

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def conf_matrix(output, labels):
    preds = output.max(1)[1].type_as(labels)
    matrix = confusion_matrix(labels.detach().cpu().numpy(), preds.detach().cpu().numpy())
    return matrix


def calc_confusion_matrices(model, train_samples, val_samples, test_samples):
    train_mats = []
    val_mats = []
    test_mats = []
    
    with torch.no_grad():
        for sample in train_samples: 
            features = sample.features.to(DEVICE)
            label = sample.labels.to(DEVICE)
            all_graphs = [graph.to(DEVICE) for graph in sample.graph_data]
            time_steps = sample.window
            adj = sample.adj
            
            output = model(all_graphs, features, time_steps, adj)
            train_mats.append(conf_matrix(output, label))

        for sample in val_samples:
            features = sample.features.to(DEVICE)
            label = sample.labels.to(DEVICE)
            all_graphs = [graph.to(DEVICE) for graph in sample.graph_data]
            time_steps = sample.window
            adj = sample.adj
            
            output = model(all_graphs, features, time_steps, adj)
            val_mats.append(conf_matrix(output, label))

        for sample in test_samples:
            features = sample.features.to(DEVICE)
            label = sample.labels.to(DEVICE)
            all_graphs = [graph.to(DEVICE) for graph in sample.graph_data]
            time_steps = sample.window
            adj = sample.adj
            output = model(all_graphs, features, time_steps, adj)
            test_mats.append(conf_matrix(output, label))

    return [(sum(train_mats)), (sum(val_mats)), (sum(test_mats))]


def calc_prediction_map(model, train_samples, val_samples, test_samples):
    train_map = {}
    val_map = {}
    test_map = {}
    model.eval()
    
    with torch.no_grad():
        for sample in train_samples:
            train_features = sample.features.to(DEVICE)
            label = sample.labels.to(DEVICE)
            all_graphs = [graph.to(DEVICE) for graph in sample.graph_data]
            time_steps = sample.window
            adj = sample.adj
            
            output = model(all_graphs, train_features, time_steps, adj)
            
            reversed_users = {v: k for k, v in sample.user_index.items()}
            preds = output.max(1)[1].type_as(sample.labels)
            for index, pred in enumerate(preds):
                if reversed_users[index] in train_map.keys():
                    train_map[reversed_users[index]][pred] += 1
                else:
                    new_entry = [0, 0]
                    new_entry[pred] += 1
                    train_map[reversed_users[index]] = new_entry


        for sample in val_samples:
            val_features = sample.features.to(DEVICE)
            label = sample.labels.to(DEVICE)
            all_graphs = [graph.to(DEVICE) for graph in sample.graph_data]
            time_steps = sample.window
            adj = sample.adj
            
            output = model(all_graphs, val_features, time_steps, adj)
            
            reversed_users = {v: k for k, v in sample.user_index.items()}
            preds = output.max(1)[1].type_as(sample.labels)
            for index, pred in enumerate(preds):
                if reversed_users[index] in val_map.keys():
                    val_map[reversed_users[index]][pred] += 1
                else:
                    new_entry = [0, 0]
                    new_entry[pred] += 1
                    val_map[reversed_users[index]] = new_entry


        for sample in test_samples:
            test_features = sample.features.to(DEVICE)
            label = sample.labels.to(DEVICE)
            all_graphs = [graph.to(DEVICE) for graph in sample.graph_data]
            time_steps = sample.window
            adj = sample.adj
            
            output = model(all_graphs, test_features, time_steps, adj)

            reversed_users = {v: k for k, v in sample.user_index.items()}
            preds = output.max(1)[1].type_as(sample.labels)
            for index, pred in enumerate(preds):
                if reversed_users[index] in test_map.keys():
                    test_map[reversed_users[index]][pred] += 1
                else:
                    new_entry = [0, 0]
                    new_entry[pred] += 1
                    test_map[reversed_users[index]] = new_entry


    return train_map, val_map, test_map