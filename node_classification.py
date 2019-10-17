#copied from GCN github to load the dataset

import numpy as np
import pickle as pkl
import copy 
import networkx as nx
import igraph as ig
from sklearn.metrics import roc_auc_score
import scipy.sparse as sp
import random
from scipy.sparse.linalg.eigen.arpack import eigsh
from collections import Counter
from networkx.algorithms.community import LFR_benchmark_graph
from networkx.algorithms import node_classification
import matplotlib
matplotlib.use('agg')
import pylab as plt
import sys


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
    All objects above must be saved using python pickle module.
    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("networks/real-node-label/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("networks/real-node-label/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    #return adj, labels
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask



'''
Load the synthetic LFR dataset
default parameter from assignment 3
'''
def load_synthetic(mu, n=1000, tau1=3, tau2=1.5, edge_drop_percent=0.2):
    G = LFR_benchmark_graph(n, tau1, tau2, mu, average_degree=5, min_community=30, seed=10)
    for n in G.nodes:
    	G.nodes[n]['value'] = list(G.nodes[n]['community'])[0]
    true_coms = list(nx.get_node_attributes(G,'value').values())
    com_keys = list(Counter(true_coms).keys())
    for i in range(0, len(true_coms)):
    	G.nodes[i]['value'] = com_keys.index(true_coms[i])
    #remove self edges 
    selfE = list(G.selfloop_edges())
    for (i,j) in selfE:
        G.remove_edge(i,j)

    #convert all graph to undirected
    G = nx.Graph(G)
    ListOfNodes = list(G.nodes())
    sample = int(len(ListOfNodes)*node_drop_percent)
    RandomSample = random.sample(ListOfNodes, sample)
    for n in G.nodes():
        if (n not in RandomSample):
            G.nodes[n][label_name] = G.nodes[n]['value']
    return (G, RandomSample)



'''
Load the real-classic dataset

dataset_str = 'strike', 'karate', 'polblogs', 'polbooks' or 'football'

'''
def load_real_classic(dataset_str, node_drop_percent, label_name):
    data_path = 'networks/real-classic/'
    if (dataset_str is 'karate'):
        G = nx.karate_club_graph()
        # use the ground truth in https://piratepeel.github.io/slides/groundtruth_presentation.pdf
        # label attribute is the ground truth in this case
        nx.set_node_attributes(G, 0, 'value')
        for n in G.nodes:
        	G.nodes[n]['value'] = G.nodes[n]['club']
        #turn categorical community labels into integer ones
        true_coms = list(nx.get_node_attributes(G,'value').values())
        com_keys = list(Counter(true_coms).keys())
        for i in range(0, len(true_coms)):
        	G.nodes[i]['value'] = com_keys.index(true_coms[i])

    else:
        G = nx.read_gml(data_path+dataset_str+'.gml', label='id')
        if (dataset_str is 'polbooks'):
            #turn categorical community labels into integer ones
            true_coms = list(nx.get_node_attributes(G,'value').values())
            com_keys = list(Counter(true_coms).keys())
            for i in range(0, len(true_coms)):
                G.nodes[i]['value'] = com_keys.index(true_coms[i])

    #remove self edges 
    selfE = list(G.selfloop_edges())
    for (i,j) in selfE:
        G.remove_edge(i,j)

    #convert all graph to undirected
    G = nx.Graph(G)

    #drop nodes after the graph is now undirected
    ListOfNodes = list(G.nodes())
    sample = int(len(ListOfNodes)*node_drop_percent)
    RandomSample = random.sample(ListOfNodes, sample)
    for n in G.nodes():
        if (n not in RandomSample):
            G.nodes[n][label_name] = G.nodes[n]['value']
    return (G, RandomSample)


def load_real_node(dataset_str, label_name):

    (A, features, y_train, y_val, y_test, train_mask, val_mask, test_mask) = load_data(dataset_str) 
    y_train = [ np.where(l == 1)[0][0] if len(np.where(l == 1)[0]) > 0 else -1  for l in y_train]
    y_val = [ np.where(l == 1)[0][0] if len(np.where(l == 1)[0]) > 0 else -1  for l in y_val]
    y_test = [ np.where(l == 1)[0][0] if len(np.where(l == 1)[0]) > 0 else -1  for l in y_test]
    G = nx.Graph(A)
    for n in G.nodes():
        if (y_train[n] >= 0):
            G.nodes[n]['value'] = y_train[n]
        if (y_val[n] >= 0):
            G.nodes[n]['value'] = y_val[n]
        if (y_test[n] >= 0):
            G.nodes[n]['value'] = y_test[n]

    for n in G.nodes():
        if (y_train[n] >= 0):
            G.nodes[n][label_name] = y_train[n]
        if (y_val[n] >= 0):
            G.nodes[n][label_name] = y_val[n]


    #remove self edges 
    selfE = list(G.selfloop_edges())
    for (i,j) in selfE:
        G.remove_edge(i,j)

    #convert all graph to undirected
    G = nx.Graph(G)
    RandomSample = []
    for i in range(0,len(y_test)):
        if (y_test[i] >= 0):
            RandomSample.append(i)
    return (G, RandomSample)

def harmonic_func(G, label_name):
    predicted = list(node_classification.harmonic_function(G, label_name=label_name))
    idx = 0
    for n in G.nodes():
        G.nodes[n][label_name] = predicted[idx]
        idx = idx + 1
    return G

def l_g_consistency(G, label_name):
    predicted = list(node_classification.local_and_global_consistency(G, label_name=label_name))
    idx = 0
    for n in G.nodes():
        G.nodes[n][label_name] = predicted[idx]
        idx = idx + 1
    return G

def evaluation(G, label_name, RandomSample):
    Accuracy = 0
    Total = len(RandomSample)
    for n in G.nodes():
        if (n in RandomSample):
            if (G.nodes[n][label_name] == G.nodes[n]['value']):
                Accuracy = Accuracy + 1
    Accuracy = Accuracy / Total
    return Accuracy


def plot_result(x_list, y, name, dataset_strs, LFR=False):
    print ("the algorithm is " + name)
    idx = 0
    num_x = len(x_list)
    for dataset_str in dataset_strs:
        print ("for dataset : "+ dataset_str)
        plt.rcParams.update({'figure.autolayout': True})
        plt.rc('xtick', labelsize='x-small')
        plt.rc('ytick', labelsize='x-small')
        fig = plt.figure(figsize=(4, 2))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x_list, y[idx:idx+num_x], marker='o', color='#74a9cf', ls='solid', linewidth=0.5, markersize=1, label=name)
        if(LFR):
            ax.set_xlabel('Percentage of dropped labels', fontsize=8)
        else:
            ax.set_xlabel('mu value / diffilculty level', fontsize=8)
        ax.set_ylabel('Accuracy', fontsize=8)
        plt.legend(fontsize = 'x-small')
        plt.savefig(dataset_str+name+'.pdf',bbox_inches='tight', pad_inches=0)
        plt.clf()

    print ("----------------------------------------------------")
    print ("----------------------------------------------------")

def print_result(accu_list, name, dataset_strs):
    print ("the algorithm is " + name)
    idx = 0
    for dataset_str in dataset_strs:
        print ("for dataset : "+ dataset_str)
        print ("Accuracy is " + str(accu_list[idx]))
        idx = idx + 1
    print ("----------------------------------------------------")
    print ("----------------------------------------------------")



def run_real_classic():
	#real_classic: 'strike', 'karate', 'polblog', 'polbooks' or 'footbal'
    real_classic = ['strike', 'karate', 'polblogs', 'polbooks', 'football']
    #label is the field where predicted label will be stored
    node_drop_percent = 0.05
    label_name = 'target'
    total_harmonic_accu = []
    total_l_g_accu = []
    x_list = list(range(5,100,5))
    x_list = [(x/100) for x in x_list]
    num_x = len(x_list)

    for datastr in real_classic:
        node_drop_percent = 0.05
        for k in range(0,num_x):
            harmonic_accu = []
            l_g_accu = []
            for i in range(0,10):
                print ("looking at " + datastr + " dataset")
                print ("dropping " + str(node_drop_percent) + " percentage of node label")
                (nG, RandomSample) = load_real_classic(datastr, node_drop_percent, label_name)
                G = nx.Graph(nG)

                G = harmonic_func(G, label_name)
                h_accu = evaluation(G, label_name, RandomSample)
                harmonic_accu.append(h_accu)

                G = nx.Graph(nG)
                G = l_g_consistency(G, label_name)
                l_accu = evaluation(G, label_name, RandomSample)
                l_g_accu.append(l_accu)
            node_drop_percent = node_drop_percent + 0.05
            total_harmonic_accu.append(sum(harmonic_accu) / len(harmonic_accu))
            total_l_g_accu.append(sum(l_g_accu) / len(l_g_accu))

    plot_result(x_list,  total_harmonic_accu, "harmonic function", real_classic, LFR=False)
    plot_result(x_list, total_l_g_accu, "local and global consistency", real_classic, LFR=False)

def run_synthetic():
    base_str = 'LFR'
    print ("looking at " + base_str + " dataset")
    node_drop_percent = 0.20
    label_name = 'target'
    mu_list = list(range(1, 10, 1))
    mu_list = [(x/10) for x in mu_list]
    LFR_difficulty = [(base_str+str(x)) for x in mu_list]
    total_harmonic_accu = []
    total_l_g_accu = []
    idx = 0

    for mu in mu_list:
        datastr = LFR_difficulty[idx]
        harmonic_accu = []
        l_g_accu = []
        for k in range(0,10):
            print ("looking at " + datastr + " dataset")
            print ("dropping " + str(node_drop_percent) + " percentage of node label")
            (nG, RandomSample) = load_synthetic(node_drop_percent, label_name, mu)
            G = nx.Graph(nG)

            G = harmonic_func(G, label_name)
            h_accu = evaluation(G, label_name, RandomSample)
            harmonic_accu.append(h_accu)

            G = nx.Graph(nG)
            G = l_g_consistency(G, label_name)
            l_accu = evaluation(G, label_name, RandomSample)
            l_g_accu.append(l_accu)
        total_harmonic_accu.append(sum(harmonic_accu) / len(harmonic_accu))
        total_l_g_accu.append(sum(l_g_accu) / len(l_g_accu))
        idx = idx + 1
    plot_result(mu_list, total_harmonic_accu, "harmonic function", [base_str], LFR=True)
    plot_result(mu_list, total_l_g_accu, "local and global consistency", [base_str], LFR=True)





def run_real_node():
    real_node = ['cora','citeseer', 'pubmed']
    label_name = 'target'
    total_harmonic_accu = []
    total_l_g_accu = []


    for datastr in real_node:
        harmonic_accu = []
        l_g_accu = []
        for k in range(0,10):
            print ("looking at " + datastr + " dataset")
            (nG, RandomSample) = load_real_node(datastr, label_name)
            G = nx.Graph(nG)

            G = harmonic_func(G, label_name)
            h_accu = evaluation(G, label_name, RandomSample)
            harmonic_accu.append(h_accu)

            G = nx.Graph(nG)
            G = l_g_consistency(G, label_name)
            l_accu = evaluation(G, label_name, RandomSample)
            l_g_accu.append(l_accu)
        total_harmonic_accu.append(sum(harmonic_accu) / len(harmonic_accu))
        total_l_g_accu.append(sum(l_g_accu) / len(l_g_accu))
    print_result(total_harmonic_accu, 'harmonic function', real_node)
    print_result(total_l_g_accu, 'local and global consistency', real_node)




def main():
    run_real_node()
    run_real_classic()
    run_synthetic()








    

if __name__ == "__main__":
    main()



