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

from gensim.models import Word2Vec
from pathlib import Path
from os import listdir
import pandas as pd

if __name__ == '__main__':
    # pasta que contém o .csv de artigos pré-processados e nome desse arquivo:
    CLEANED_DOCUMENTS_PATH = '/home/matheus/WE4LKD-leukemia_w2v/pubchem/results/'
    FILENAME = listdir(CLEANED_DOCUMENTS_PATH)
    FILENAME = [f for f in FILENAME if f.endswith('.csv')]
    assert len(FILENAME) == 1, FILENAME

    # leitura do arquivo .csv em um DataFrame:
    print('reading papers...')
    df = pd.read_csv(CLEANED_DOCUMENTS_PATH + FILENAME[0], escapechar='\\')

    # todos os anos de publicação (sem repetição) presentes no arquivo .csv:
    years = sorted(df.filename.unique().tolist())
    first_year = years[0]       # ano de publicação do primeiro artigo presente no .csv:

    # intervalos ou "janelas" de tempo, a partir do primeiro artigo publicado, até o último. Exemplo:
    # supondo que o primeiro artigo coleto foi publicado em 1921, e que, depois deste, mais artigos foram publicados nos anos seguintes, temos:
    # [[1921], [1921, 1922], [1921, 1922, 1923], [1921, 1922, 1923, 1924], [1921, 1922, 1923, 1924, 1925], .......]
    ranges = [years[:i+1] for i in range(len(years))]

    for r in ranges:
        print('training model from {} to {}'.format(r[0], r[-1]))
        abstracts = df[df.filename.isin(r)]['summary'].to_list()
        print('number of abstracts: {}\n'.format(len(abstracts)))
        abstracts = [x.split() for x in abstracts]

        # train model
        model = Word2Vec(abstracts, min_count=2, size=30, sg=1, alpha=0.01, iter=30, window=8)

        # save model
        model.save('./model_{}_{}.model'.format(first_year, r[-1]))
