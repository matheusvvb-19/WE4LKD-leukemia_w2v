##################################################
## Trains Word2Vec or FastText models from cmbinations of hyperparameters.
##################################################
## Author: Matheus Vargas Volpon Berto
## Copyright: Copyright 2022, Discovering Latent Knowledge in medical paper on Acute Myeloid Leukemia
## Email: matheusvvb@hotmail.com
##################################################

import os, re, sys, shutil, itertools, gensim
import pandas as pd
import numpy as np
from gensim.utils import RULE_KEEP, RULE_DEFAULT
from gensim.models import Word2Vec, FastText
from pathlib import Path
from os import listdir

def list_from_txt(file_path):
    """Creates a list of itens based on a .txt file, each line becomes an item.
    
    Args: 
      file_path: the path where the .txt file was created. 
    """
    
    strings_list = []
    with open (file_path, 'rt', encoding='utf-8') as file:
        for line in file:
            strings_list.append(line.rstrip('\n'))
    return strings_list

def clear_folder(dirpath):
    """ Clears all files from a folder, without deleting the folder.

    Args:
        dirpath: the path of the folder.    
    """

    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)

def get_target_compounds():
    return sorted(['cytarabine', 'daunorubicin', 'azacitidine', 'midostaurin', 'gemtuzumab-ozogamicin', 'vyxeos', 'ivosidenib', 'venetoclax', 'enasidenib', 'gilteritinib', 'glasdegib', 'arsenictrioxide', 'cyclophosphamide', 'dexamethasone', 'idarubicin', 'mitoxantrone', 'pemigatinib', 'prednisone', 'rituximab', 'thioguanine', 'vincristine'])

def keep_target_compounds(word, countm, min_count):
    if word in get_target_compounds():
        return gensim.utils.RULE_KEEP

    else:
        return gensim.utils.RULE_DEFAULT

if __name__ == '__main__':
    print('Starting script')

    # CONSTANT(S):
    TRAINING_FASTTETX_MODELS = True
    models_already_trained = []
    if TRAINING_FASTTETX_MODELS:
        os.makedirs('../fasttext/', exist_ok=True)
        os.makedirs('../fasttext/models_nolemma/', exist_ok=True)
        models_already_trained = [x for x in os.listdir('../fasttext/models_nolemma/') if x.endswith('.model')]
    
    else:
        os.makedirs('./models_nolemma/', exist_ok=True)
        models_already_trained = [x for x in os.listdir('./models_nolemma/') if x.endswith('.model')]

    print('Reading DataFrame of papers')
    df = pd.read_csv('/data/ac4mvvb/WE4LKD-leukemia_w2v/pubchem/results_pandas.csv', escapechar='\\')
    years = sorted(df.filename.unique().tolist())
    ranges = [years[:i+1] for i in range(len(years))]
    ranges = [x for x in ranges if 1963 in x]

    general_index = -1
    print('\nSelected index: {}'.format(general_index))
    print('range: from {} to {}'.format(ranges[general_index][0], ranges[general_index][-1]))

    abstracts = df[df.filename.isin(ranges[general_index])]['summary'].to_list()
    print('Number of abstracts: {}\n'.format(len(abstracts)))
    abstracts = [x.split() for x in abstracts]

    parm_dict = {'size': (100, 200, 300), 'alpha': (0.0025, 0.025), 'negative': (5, 10, 15)}
    size, alpha, negative = [tup for k, tup in parm_dict.items()]
    parm_combo = list(itertools.product(size, alpha, negative))
    tam = len(parm_combo)

    print('\nGridsearch')
    for index, parms in enumerate(parm_combo):
        s, a, n = parms
        if 'model_{}_{}_s{}_a{}_n{}.model'.format(ranges[general_index][0], ranges[general_index][-1], s, a, n) in models_already_trained:
            continue

        else:
            print('training model {}/{}: size={}, alpha={}, negative={}'.format(index+1, tam, s, a, n))

            if TRAINING_FASTTETX_MODELS:
                model = FastText(
                            # constant parameters:
                            sentences=abstracts,
                            sorted_vocab=True,
                            min_count=5,
                            sg=1,
                            hs=0,
                            iter=15,
                            trim_rule=keep_target_compounds,
                            # variable parameters:
                            size=s,
                            alpha=a,
                            negative=n
                        )

                model.save('../fasttext/models_nolemma/model_{}_{}_s{}_a{}_n{}.model'.format(ranges[general_index][0], ranges[general_index][-1], s, a, n))

            else:
                model = Word2Vec(
                            # constant parameters:
                            sentences=abstracts,
                            sorted_vocab=True,
                            min_count=5,
                            sg=1,
                            hs=0,
                            iter=15,
                            trim_rule=keep_target_compounds,
                            # variable parameters:
                            size=s,
                            alpha=a,
                            negative=n
                        )

                model.save('./models_nolemma/model_{}_{}_s{}_a{}_n{}.model'.format(ranges[general_index][0], ranges[general_index][-1], s, a, n))           

    print('END!')
