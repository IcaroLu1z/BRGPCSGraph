import pandas as pd
import numpy as np
import networkx as nx
from node2vec import Node2Vec
import time
from gensim.models.callbacks import CallbackAny2Vec
import multiprocessing as mp
from functools import partial
from itertools import product

# -----------------------------------------------------------CALLBACK-----------------------------------------------------------
class StopTrainingException(Exception):
    pass

class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0
        self.loss_to_be_subed = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        print('Loss after epoch {}: {:.3f}'.format(self.epoch, loss_now))
        
        if loss_now <= 0:
            self.model = model
            raise StopTrainingException('Loss is zero or negative. Stopping training.')
        
        self.epoch += 1

# -----------------------------------------------------------TRAINING-----------------------------------------------------------
def train_node2vec(colab_net, epochs, dim, w_l, n_w, window, p, q, workers):
    n2v = Node2Vec(colab_net, p=p, q=q, dimensions=dim, walk_length=w_l, num_walks=n_w, workers=workers, weight_key='weight')
    
    try:
        callback_instance = callback()
        model = n2v.fit(window=window, min_count=1, epochs=epochs, workers=workers, 
                        alpha=1e-3, min_alpha=1e-4, compute_loss=True, callbacks=[callback_instance])
    except StopTrainingException as e:
        model = callback_instance.model
        print(e)
    
    return model

# -----------------------------------------------------------GET EMBEDDINGS-----------------------------------------------------------
def extract_embeddings(dim_wl, colab_net, alpha, epochs, p, q, workers, ppgcc_subgraph):
    dim, w_l = dim_wl
    print(f'Extracting embeddings for dim={dim}, w_l={w_l}') # Paper original -> dim = 128 -> estabiliza no 100
    '''n_w = round(0.3 * w_l) # Paper original -> w_l = 80, n_w = 10 ratio = 12,5 -> maior w_l, maior n_w = melhor resultado
    window = round(0.05 * w_l) if round(0.05 * w_l) > 3 else 3''' # Paper original -> window = 10 -> variar tem pouca influência
    n_w = round(0.25 * w_l)
    window = 10
    model = train_node2vec(colab_net, epochs, dim, w_l, n_w, window, p, q, workers) # Paper original -> p = 1, q = 1 -> menor p, menor q = melhor resultado
    
    try:
        author_embeddings = {node: model.wv[node] for node in ppgcc_subgraph if node in model.wv}
    except:
        print('Error: some nodes were not found in the model.')
    #emb_path = f'/media/work/icarovasconcelos/mono/data/backbone_embeddings/n2v_authors_embeddings_{alpha}_{window}win_{epochs}ep_{dim}dim_{w_l}wl_{n_w}nw_{p}p_{q}q.npy'
    emb_path = f'/media/work/icarovasconcelos/mono/data/test_embeddings/n2v_authors_embeddings_{alpha}_{window}win_{epochs}ep_{dim}dim_{w_l}wl_{n_w}nw_{p}p_{q}q.npy'
    np.save(emb_path, author_embeddings)
    
    return emb_path

# -----------------------------------------------------------LOAD DATA-----------------------------------------------------------
begin = time.time()
authors_data = pd.read_csv('/media/work/icarovasconcelos/mono/data/authors/authors-processed.csv')

ppgcc_subgraph = []
for author_id in authors_data['author_id']:
    ppgcc_subgraph.append(author_id)

print(f'Number of nodes in the subgraph: {len(ppgcc_subgraph)}')
colab_net_path = '/media/work/icarovasconcelos/mono/data/backbones/collabNetx_backbone_a0.24.graphml'
colab_net = nx.read_graphml(colab_net_path)

# --------------------------------------------------------------------------------------------------------------------------------

alpha = colab_net_path.split('_')[-1].replace('.graphml', '')

'''dimensions = list(range(8, 33, 8))
w_l = list(range(50, 201, 25))'''
dimensions = list(range(4, 33, 4))
dimensions.append(2)
w_l = list(range(75, 201, 25))

p = 1
q = 1
epochs = 50
dim_wl = list(product(dimensions, w_l))

extract_embeddings_partial = partial(extract_embeddings, colab_net=colab_net, alpha=alpha, 
                                     epochs=epochs, p=p, q=q, workers=10, ppgcc_subgraph=ppgcc_subgraph)

with mp.Pool(10) as pool:
    embeddings_paths = pool.map(extract_embeddings_partial, dim_wl)
    

print('Finished')