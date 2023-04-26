##################################################
## Trains incremental Word2Vec or FastText language models from the selected hyperparameter combination.
##################################################
## Author: {name}
## Copyright: Copyright 2022, Discovering Latent Knowledge in medical paper on Acute Myeloid Leukemia
## Email: {contact_email}
##################################################

# IMPORTS:
import os, re, sys, shutil, itertools, gensim
from gensim.utils import RULE_KEEP, RULE_DEFAULT
from gensim.models import Word2Vec, FastText
from pathlib import Path
from os import listdir
import pandas as pd
import numpy as np

from train import list_from_txt, keep_target_compounds

# MAIN PROGRAM:
if __name__ == '__main__':
    print('Starting script')

    # CONSTANTS:
    MODEL_TYPE = 'ft' # 'w2v' for Word2Vec or 'ft' for FastText
    if MODEL_TYPE == 'w2v':
        os.makedirs('./models_yoy_combination15/', exist_ok=True)
        os.makedirs('./models_yoy_combination2/', exist_ok=True)

        parameters_combination = [[100, 0.0025, 10], [200, 0.025, 15]]
    
    else:
        os.makedirs('../fasttext/models_yoy_combination16/', exist_ok=True)
        parameters_combination = [[300, 0.0025, 5]]

    # leitura do arquivo .csv em um DataFrame:
    print('Reading DataFrame of papers')
    df = pd.read_csv('/data/ac4mvvb/WE4LKD-leukemia_w2v/pubchem/results_pandas.csv', escapechar='\\')
    
    years = sorted(df.filename.unique().tolist())
    first_year = years[0]

    # computing timespan :
    # assuming that the first collected article was published in 1921, and that more articles were published after this one in the following years, we have ranges equal to:
    # [[1921], [1921, 1922], [1921, 1922, 1923], [1921, 1922, 1923, 1924], [1921, 1922, 1923, 1924, 1925], .......]
    ranges = [years[:i+1] for i in range(len(years))]

    for r in ranges:
        print('training model from {} to {}'.format(r[0], r[-1]))
        abstracts = df[df.filename.isin(r)]['summary'].to_list()
        print('number of abstracts: {}\n'.format(len(abstracts)))
        abstracts = [x.split() for x in abstracts]

        # train model
        if MODEL_TYPE == 'w2v':            
            model_comb15 = Word2Vec(
                            # constant parameters:
                            sentences=abstracts,
                            sorted_vocab=True,
                            min_count=5,
                            sg=1,
                            hs=0,
                            iter=15,
                            trim_rule=keep_target_compounds,
                            # variable parameters:
                            size=parameters_combination[1][0],
                            alpha=parameters_combination[1][1],
                            negative=parameters_combination[1][2]
                        )
            model_comb15.save('./models_yoy_combination15/model_{}_{}.model'.format(first_year, r[-1]))
        
        else:
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
                    size=parameters_combination[0][0],
                    alpha=parameters_combination[0][1],
                    negative=parameters_combination[0][2])
            model.save('/fastdata/ac4mvvb/fasttext/models_yoy_combination16/model_{}_{}.model'.format(first_year, r[-1]))        

    print('END!')
