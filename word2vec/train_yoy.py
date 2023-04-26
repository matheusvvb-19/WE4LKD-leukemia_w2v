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
from gensim.models import Word2Vec, FastText
from pathlib import Path
from os import listdir
import pandas as pd
import numpy as np

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
    if word in get_target_compounds() + ['aml']:
        return gensim.utils.RULE_KEEP

    else:
        return gensim.utils.RULE_DEFAULT

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

    # todos os anos de publicação (sem repetição) presentes no arquivo .csv:
    years = sorted(df.filename.unique().tolist())
    first_year = years[0]

    # intervalos ou "janelas" de tempo, a partir do primeiro artigo publicado, até o último. Exemplo:
    # supondo que o primeiro artigo coleto foi publicado em 1921, e que, depois deste, mais artigos foram publicados nos anos seguintes, temos:
    # [[1921], [1921, 1922], [1921, 1922, 1923], [1921, 1922, 1923, 1924], [1921, 1922, 1923, 1924, 1925], .......]
    ranges = [years[:i+1] for i in range(len(years))]
    ranges = [x for x in ranges if 1980 in x]
    ranges = [x for x in ranges if x[-1] == 1980]

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