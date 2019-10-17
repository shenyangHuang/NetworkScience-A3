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
    ListOfEdges = [e for e in G.edges]
    sample = int(len(ListOfEdges)*edge_drop_percent)
    RandomSample = random.sample(ListOfEdges, sample)
    G.remove_edges_from(RandomSample)

    return (G, RandomSample)




'''
Load the real-classic dataset

dataset_str = 'strike', 'karate', 'polblogs', 'polbooks' or 'football'

'''
def load_real_classic(dataset_str, edge_drop_percent=0.2):
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

    #drop edges after the graph is now undirected
    ListOfEdges = [e for e in G.edges]
    sample = int(len(ListOfEdges)*edge_drop_percent)
    RandomSample = random.sample(ListOfEdges, sample)
    G.remove_edges_from(RandomSample)

    return (G, RandomSample)





def load_real_node(dataset_str, edge_drop_percent=0.2):

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

    #remove self edges 
    selfE = list(G.selfloop_edges())
    for (i,j) in selfE:
        G.remove_edge(i,j)

    #convert all graph to undirected
    G = nx.Graph(G)
    #drop edges after the graph is now undirected
    ListOfEdges = [e for e in G.edges]
    sample = int(len(ListOfEdges)*edge_drop_percent)
    RandomSample = random.sample(ListOfEdges, sample)
    G.remove_edges_from(RandomSample)

    return (G, RandomSample)


'''
helper function for sorting purpose only
return the last element in each tuple in list of tuples
'''
def getKey(item):
    return item[-1]



'''
Make edge prediction based on preferential_attachment
https://networkx.github.io/documentation/networkx-1.10/reference/algorithms.link_prediction.html
https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.algorithms.link_prediction.preferential_attachment.html#networkx.algorithms.link_prediction.preferential_attachment
return (u,v,p) for all potential edges
'''
def preferential_algorithm(G):
    predictions = list(nx.preferential_attachment(G))
    return predictions

'''
make edge prediction based on jaccard coefficient
https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.algorithms.link_prediction.jaccard_coefficient.html#networkx.algorithms.link_prediction.jaccard_coefficient
return top--num_edge candidates
'''
def jaccard_algorithm(G):
    predictions = list(nx.jaccard_coefficient(G))
    return predictions



'''
make edge prediction based on resource_allocation_index
https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.algorithms.link_prediction.resource_allocation_index.html#networkx.algorithms.link_prediction.resource_allocation_index
return top--num_edge candidates
'''
def resource_algorithm(G):
    predictions = list(nx.resource_allocation_index(G))
    return predictions

'''
make edge prediction based on adamic_adar_index
https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.algorithms.link_prediction.adamic_adar_index.html#networkx.algorithms.link_prediction.adamic_adar_index
return top--num_edge candidates
'''
def adamic_algorithm(G):
    predictions = list(nx.adamic_adar_index(G))
    return predictions



'''
calculate the area under the curve score based on the probability predicted by the algorithms
sort the edges in the same order
'''
def auc_evaluation(G, RandomSample, predictions):
    p_preds = [p for (u,v,p) in predictions]
    p_true = [0] * len(p_preds)
    all_edges = [(u,v) for (u,v,p) in predictions]
    for i in range(0, len(all_edges)):
        if (all_edges[i] in RandomSample):
            p_true[i] = 1
    auc = roc_auc_score(p_true, p_preds)
    return auc


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
            ax.set_xlabel('mu value / diffilculty level', fontsize=8)
        else:
            ax.set_xlabel('Percentage of dropped labels', fontsize=8)
        ax.set_ylabel('AUC score', fontsize=8)
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
        print ("auc average is " + str(accu_list[idx]))
        idx = idx + 1
    print ("----------------------------------------------------")
    print ("----------------------------------------------------")


def run_real_classic():
	#real_classic: 'strike', 'karate', 'polblog', 'polbooks' or 'footbal'
    real_classic = ['strike', 'karate', 'polblogs', 'polbooks', 'football']
    total_preferential_auc = []
    total_jaccard_auc = []
    total_resource_auc = []
    total_adamic_auc = []

    for datastr in real_classic:
        preferential_auc = []
        jaccard_auc = []
        resource_auc = []
        adamic_auc = []
        for i in range(0,10):
            print ("looking at " + datastr + " dataset")
            (nG, RandomSample) = load_real_classic(datastr)
            G = nx.Graph(nG)

            predictions = preferential_algorithm(G)
            preferential_auc.append(auc_evaluation(G, RandomSample, predictions))

            predictions = jaccard_algorithm(G)
            jaccard_auc.append(auc_evaluation(G, RandomSample, predictions))

            predictions = resource_algorithm(G)
            resource_auc.append(auc_evaluation(G, RandomSample, predictions))

            predictions = adamic_algorithm(G)
            adamic_auc.append(auc_evaluation(G, RandomSample, predictions))

        total_preferential_auc.append(sum(preferential_auc) / len(preferential_auc))
        total_jaccard_auc.append(sum(jaccard_auc) / len(jaccard_auc))
        total_resource_auc.append(sum(resource_auc) / len(resource_auc))
        total_adamic_auc.append(sum(adamic_auc) / len(adamic_auc))

    print_result(total_preferential_auc, 'preferential attachment', real_classic)
    print_result(total_jaccard_auc, 'jaccard coefficient', real_classic)
    print_result(total_resource_auc, 'resource allocation index', real_classic)
    print_result(total_adamic_auc, 'adamic adar index', real_classic)

def run_synthetic():
    base_str = 'LFR'
    print ("looking at " + base_str + " dataset")
    mu_list = list(range(1, 10, 1))
    mu_list = [(x/10) for x in mu_list]
    LFR_difficulty = [(base_str+str(x)) for x in mu_list]
    total_preferential_auc = []
    total_jaccard_auc = []
    total_resource_auc = []
    total_adamic_auc = []
    idx = 0

    for mu in mu_list:
        datastr = LFR_difficulty[idx]
        preferential_auc = []
        jaccard_auc = []
        resource_auc = []
        adamic_auc = []
        for i in range(0,10):
            print ("looking at " + datastr + " dataset")
            print ("difficulty is " + str(mu))
            (nG, RandomSample) = load_synthetic(mu)
            
            G = nx.Graph(nG)

            predictions = preferential_algorithm(G)
            preferential_auc.append(auc_evaluation(G, RandomSample, predictions))
            print (auc_evaluation(G, RandomSample, predictions))


            predictions = jaccard_algorithm(G)
            jaccard_auc.append(auc_evaluation(G, RandomSample, predictions))
            print (auc_evaluation(G, RandomSample, predictions))

            predictions = resource_algorithm(G)
            resource_auc.append(auc_evaluation(G, RandomSample, predictions))
            print (auc_evaluation(G, RandomSample, predictions))

            predictions = adamic_algorithm(G)
            adamic_auc.append(auc_evaluation(G, RandomSample, predictions))
            print (auc_evaluation(G, RandomSample, predictions))

        total_preferential_auc.append(sum(preferential_auc) / len(preferential_auc))
        total_jaccard_auc.append(sum(jaccard_auc) / len(jaccard_auc))
        total_resource_auc.append(sum(resource_auc) / len(resource_auc))
        total_adamic_auc.append(sum(adamic_auc) / len(adamic_auc))
        idx = idx + 1
    plot_result(mu_list, total_preferential_auc, 'preferential attachment', [base_str], LFR=True)
    plot_result(mu_list, total_jaccard_auc, "jaccard coefficient", [base_str], LFR=True)
    plot_result(mu_list, total_resource_auc, 'resource allocation index', [base_str], LFR=True)
    plot_result(mu_list, total_adamic_auc, 'adamic adar index', [base_str], LFR=True)


def run_real_node():
    real_node = ['cora','citeseer', 'pubmed']
    print ("running real node")
    total_preferential_auc = []
    total_jaccard_auc = []
    total_resource_auc = []
    total_adamic_auc = []

    for datastr in real_node:
        preferential_auc = []
        jaccard_auc = []
        resource_auc = []
        adamic_auc = []
        for i in range(0,10):
            print ("looking at " + datastr + " dataset")
            (nG, RandomSample) = load_real_node(datastr)
            G = nx.Graph(nG)

            predictions = preferential_algorithm(G)
            preferential_auc.append(auc_evaluation(G, RandomSample, predictions))
            print (auc_evaluation(G, RandomSample, predictions))

            predictions = jaccard_algorithm(G)
            jaccard_auc.append(auc_evaluation(G, RandomSample, predictions))
            print (auc_evaluation(G, RandomSample, predictions))

            predictions = resource_algorithm(G)
            resource_auc.append(auc_evaluation(G, RandomSample, predictions))
            print (auc_evaluation(G, RandomSample, predictions))

            predictions = adamic_algorithm(G)
            adamic_auc.append(auc_evaluation(G, RandomSample, predictions))
            print (auc_evaluation(G, RandomSample, predictions))

        total_preferential_auc.append(sum(preferential_auc) / len(preferential_auc))
        total_jaccard_auc.append(sum(jaccard_auc) / len(jaccard_auc))
        total_resource_auc.append(sum(resource_auc) / len(resource_auc))
        total_adamic_auc.append(sum(adamic_auc) / len(adamic_auc))

    print_result(total_preferential_auc, 'preferential attachment', real_node)
    print_result(total_jaccard_auc, 'jaccard coefficient', real_node)
    print_result(total_resource_auc, 'resource allocation index', real_node)
    print_result(total_adamic_auc, 'adamic adar index', real_node)
    

def main():
	run_synthetic()
	run_real_node()
	run_real_classic()








    

if __name__ == "__main__":
    main()



