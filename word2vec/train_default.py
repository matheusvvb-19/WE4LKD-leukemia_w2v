##################################################
## Trains the default Word2Vec model with the entire corpus, used during the hyperparameter optimization.
##################################################
## Author: {name}
## Copyright: Copyright 2022, Discovering Latent Knowledge in medical paper on Acute Myeloid Leukemia
## Email: {contact_email}
##################################################

# IMPORTS:
import os, re, sys, shutil, itertools, gensim
from gensim.models import Word2Vec
from pathlib import Path
from os import listdir
import pandas as pd
import numpy as np
from train import list_from_txt, clear_folder, keep_target_compounds

# MAIN PROGRAM:
if __name__ == '__main__':
    print('Starting script')

    os.makedirs('./models_nolemma/', exist_ok=True)

    print('Reading DataFrame of papers')
    df = pd.read_csv('/data/ac4mvvb/WE4LKD-leukemia_w2v/pubchem/results_pandas.csv', escapechar='\\')
    years = sorted(df.filename.unique().tolist())
    ranges = [years[:i+1] for i in range(len(years))]
    ranges = [x for x in ranges if 1963 in x]

    general_index = -1
    print('\nselected index: {}'.format(general_index))
    print('range: from {} to {}'.format(ranges[general_index][0], ranges[general_index][-1]))

    abstracts = df[df.filename.isin(ranges[general_index])]['summary'].to_list()
    print('number of abstracts: {}\n'.format(len(abstracts)))
    abstracts = [x.split() for x in abstracts]

    model = Word2Vec(
                # constant parameters:
                sentences=abstracts,
                trim_rule=keep_target_compounds
            )

    model.save('./models_nolemma/default_{}_{}_s100_a0.025_n5.model'.format(ranges[general_index][0], ranges[general_index][-1]))           

    print('END!')
