from Bio import Entrez
import json, string, re
from Bio import Entrez
import os, fnmatch
from pathlib import Path
from datetime import datetime

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

def search(query):
    final_query = '{} AND English[Language]'.format(query) # especificando que os artigos devem ser apenas em inglês
    Entrez.email = 'matheus.berto@estudante.ufscar.br' #change here
    handle = Entrez.esearch(db='pubmed', 
                            sort='relevance', 
                            retmax='999999',
                            retmode='xml', 
                            term=final_query)
    results = Entrez.read(handle)
    return results

def fetch_details(id_list):
    ids = ','.join(id_list)
    Entrez.email = 'priscila.portela.c@gmail.com'
    handle = Entrez.efetch(db='pubmed',
                           retmode='xml',
                           id=ids)
    results = Entrez.read(handle)
    return results

search_strings = list_from_txt('search_strings.txt')
ids = []

txt_filenames = []
for root, dirnames, filenames in os.walk('results'):
    for filename in fnmatch.filter(filenames, '*.txt'):
        txt_filenames.append(filename)


for s in search_strings:
    print('searching for {}'.format(s))

    results = search('"' + s + '"')
    id_list = results['IdList']
    len_id_list = len(id_list)
    ids.extend(x for x in id_list if x not in ids)
    papers = fetch_details(id_list)
    s = s.lower().replace(' ', '_')
    Path('./results/{}'.format(s)).mkdir(parents=True, exist_ok=True)
    print('{} papers found for {}'.format(len_id_list, s))
    counter = 0

    for i, paper in enumerate(papers['PubmedArticle']):
        try:
            article_title = paper['MedlineCitation']['Article']['ArticleTitle']
            article_abstract = ' '.join(paper['MedlineCitation']['Article']['Abstract']['AbstractText'])

            article_title = article_title.translate(str.maketrans('', '', string.punctuation))
            article_title = article_title.lower().replace(' ', '_')

            article_year = paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']['Year']
        except KeyError as e:
            if 'ArticleTitle' in e.args or 'Abstract' in e.args:    # caso o artigo não tenha título ou resumo, pula ele e segue o loop para o próximo artigo
                counter += 1
                continue
            elif 'Year' in e.args:          # caso o artigo não tenha a chave "Year" na data de publicação, pega os 4 primeiros caracteres da "MedlineDate"
                article_year = paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']['MedlineDate'][0:4]
        

        filename = '{}_{}'.format(article_year, article_title)
        if len(filename) > 150:
            filename = filename[0:146]

        path_name = './results/{}/{}.txt'.format(s, filename)
        path_name = path_name.encode('ascii', 'ignore').decode('ascii')

        if path_name.split('/')[3] not in txt_filenames:
            # depois de pegar o título, resumo e data (e pulando o loop, quando não é possível), escrever o arquivo
            with open(path_name, "a", encoding='utf-8') as myfile:
                myfile.write(article_abstract)
            txt_filenames.append(path_name.split('/')[3])
    
    if counter > 0:
        print('{} papers without title or abstract'.format(counter))
        print('{} papers successfully read'.format(len_id_list - counter))
    else:
        print('all papers successfully read')
