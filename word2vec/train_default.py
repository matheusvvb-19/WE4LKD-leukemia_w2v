#######################################################
"""
    Esse script realiza a geração dos modelos Word2Vec a partir dos prefácios dos artigos
    já pré-processados/limpos/normalizados, presentes na pasta definida pela constante 
    CLEANED_DOCUMENTS_PATH.

    Os modelos sao treinados de forma incremental, por exemplo:
        Modelo 1: contempla artigos publicados entre 1900 e 1901;
        Modelo 2: contempla artigos publicados entre 1900 e 1902;
        Modelo 3: contempla artigos publicados entre 1900 e 1903;
        .
        .
        .
"""
#######################################################

import os, re, sys, shutil, itertools

import gensim
from gensim.utils import RULE_KEEP, RULE_DEFAULT
from gensim.models import Word2Vec
from pathlib import Path
from os import listdir
import pandas as pd
import numpy as np

from gensim.models import Word2Vec

def list_from_txt(file_path):
    '''Creates a list of itens based on a .txt file, each line becomes an item.
    
    Args: 
      file_path: the path where the .txt file was created. 
    '''
    
    strings_list = []
    with open (file_path, 'rt', encoding='utf-8') as file:
        for line in file:
            strings_list.append(line.rstrip('\n'))
    return strings_list

def clear_folder(dirpath):
    """ Clears all files from a folder, without deleting the folder.

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