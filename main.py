import itertools
import os
import sys
import warnings

import numpy as np
import torch as T

from src.pipeline import DivergencePipeline, GANPipeline, MLEPipeline
from src.scm.model_classes import XORModel, RoundModel
from src.scm.ctm import CTM
from src.scm.ncm import FF_NCM, GAN_NCM, MLE_NCM
from src.run import NCMRunner, NCMMinMaxRunner
from src.ds.causal_graph import CausalGraph
from src.metric.queries import get_query, get_experimental_variables, is_q_id_in_G

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Hardcoded arguments
NAME = "dat"  # Replace <NAME> with your desired name
PIPELINE = "gan"
LR = 2e-5
DATA_BS = 256
NCM_BS = 256
H_SIZE = 64
U_SIZE = 2
LAYER_NORM = True
GAN_MODE = "wgangp"
D_ITERS = 1
ID_QUERY = "ate"  # Replace <QUERY> with your desired query
ID_RERUNS = 4
MAX_LAMBDA = 1e-4
MIN_LAMBDA = 1e-5
MAX_QUERY_ITERS = 5#1000
SINGLE_DISC = True
GEN_SIGMOID = True
MC_SAMPLE_SIZE = 256
GRAPH = "expl_set"
N_TRIALS = 20
N_SAMPLES = 50 #10000
DIM = 16  # Replace <DIM> with your desired dimension
GPU = 0

valid_pipelines = {
    "divergence": DivergencePipeline,
    "gan": GANPipeline,
    "mle": MLEPipeline,
}
valid_generators = {
    "ctm": CTM,
    "xor": XORModel,
    "round": RoundModel
}
architectures = {
    "divergence": FF_NCM,
    "gan": GAN_NCM,
    "mle": MLE_NCM,
}

graph_sets = {
    "expl_set": {"expl", "expl_dox", "expl_xm", "expl_xm_dox", "expl_xy", "expl_xy_dox", "expl_my", "expl_my_dox"}
}

valid_queries = {"ate", "ett", "nde", "ctfde"}

pipeline = valid_pipelines[PIPELINE]
dat_model = valid_generators["ctm"]  # Using CTM as the default generator
ncm_model = architectures[PIPELINE]

gpu_used = [GPU]

hyperparams = {
    'lr': LR,
    'data-bs': DATA_BS,
    'ncm-bs': NCM_BS,
    'h-layers': 2,  # Default value
    'h-size': H_SIZE,
    'u-size': U_SIZE,
    'neural-pu': False,  # Default value
    'layer-norm': LAYER_NORM,
    'regions': 20,  # Default value
    'c2-scale': 1.0,  # Default value
    'gen-bs': 10000,  # Default value
    'gan-mode': GAN_MODE,
    'd-iters': D_ITERS,
    'grad-clamp': 0.01,  # Default value
    'gp-weight': 10.0,  # Default value
    'query-track': ID_QUERY,
    'id-reruns': ID_RERUNS,
    'max-query-iters': MAX_QUERY_ITERS,
    'min-lambda': MIN_LAMBDA,
    'max-lambda': MAX_LAMBDA,
    'mc-sample-size': MC_SAMPLE_SIZE,
    'single-disc': SINGLE_DISC,
    'gen-sigmoid': GEN_SIGMOID,
    'perturb-sd': 0.1,  # Default value
    'full-batch': False,  # Default value
    'positivity': True  # Default value
}

hyperparams['data-bs'] = hyperparams['data-bs'] * hyperparams['d-iters']

graph_set = graph_sets[GRAPH]

n_list = [N_SAMPLES]
d_list = [int(DIM)]

for graph in graph_set:
    do_var_list = get_experimental_variables(graph)
    eval_query, opt_query = get_query(graph, ID_QUERY)

    hyperparams['do-var-list'] = do_var_list
    hyperparams['eval-query'] = eval_query
    hyperparams['max-query-1'] = opt_query[0]
    hyperparams['max-query-2'] = opt_query[1]

    for (n, d) in itertools.product(n_list, d_list):
        n = int(n)
        hyperparams["data-bs"] = min(DATA_BS, n)
        hyperparams["ncm-bs"] = min(NCM_BS, n)

        for i in range(N_TRIALS):
            while True:
                cg_file = "dat/cg/{}.cg".format(graph)
                try:
                    runner = NCMMinMaxRunner(pipeline, dat_model, ncm_model)
                    if not runner.run("{}/{}".format(NAME, graph), cg_file, n, d, i,
                                      hyperparams=hyperparams, gpu=gpu_used, verbose=True):
                        break
                except Exception as e:
                    print(e)
                    print('[failed]', i, NAME)
                    raise

if __name__ == "__main__":
    # The script will run automatically when executed
    pass