from Bio import Entrez
import json, string, re
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

if __name__ == '__main__':
    search_strings = list_from_txt('search_strings.txt')
    ids = []

    txt_filenames = []
    for root, dirnames, filenames in os.walk('./results'):
        for filename in fnmatch.filter(filenames, '*.txt'):
            txt_filenames.append(filename)
    
    txt_filenames = set(txt_filenames)

    for s in search_strings:
        print('searching for {}'.format(s))

        results = search('"' + s + '"')
        id_list = results['IdList']
        
        old_papers = len(ids)                                               # quantidade acumulativa de artigos salvos
        ids.extend(id_list)
        ids = list(dict.fromkeys(ids))                                      # extendendo lista de artigos salvos, com remoção de duplicatas
        
        # se nenhum artigo for retornado ou se todos os artigos encontrados pelo termo de busca atual já tiverem sido salvos, pula para o próximo termo de busca:
        if len(ids) - old_papers == 0:
            continue
        
        papers = fetch_details(id_list)
        print('{} papers found for {}'.format(len(id_list), s))
        print('{} papers are new ones'.format(len(ids) - old_papers))
        
        s = s.lower().replace(' ', '_')
        Path('./results/{}'.format(s)).mkdir(parents=True, exist_ok=True)
        
        without_abstract_counter = 0

        for i, paper in enumerate(papers['PubmedArticle']):
            try:
                article_title = paper['MedlineCitation']['Article']['ArticleTitle']
                
                article_title_filename = article_title.translate(str.maketrans('', '', string.punctuation))
                article_title_filename = article_title_filename.lower().replace(' ', '_')
                
            except KeyError as e:
                if 'ArticleTitle' in e.args:
                    continue
                    
            try:                
                article_abstract = ' '.join(paper['MedlineCitation']['Article']['Abstract']['AbstractText'])
                article_year = paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']['Year']
                
            except KeyError as e:
                if 'Abstract' in e.args:        # caso o artigo não tenha prefácio, continua o processamento, pois o título já foi extraído
                    without_abstract_counter += 1
                    pass
                
                elif 'Year' in e.args:
                    article_year = paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']['MedlineDate'][0:4]


            filename = '{}_{}'.format(article_year, article_title_filename)
            if len(filename) > 150:
                filename = filename[0:146]

            path_name = './results/{}/{}.txt'.format(s, filename)
            path_name = path_name.encode('ascii', 'ignore').decode('ascii')

            if path_name.split('/')[3] not in txt_filenames:
                # depois de pegar o título, resumo e data (e pulando o loop, quando não é possível), escrever o arquivo:
                with open(path_name, "a", encoding='utf-8') as myfile:
                    myfile.write(article_title + ' ' + article_abstract)
                txt_filenames.append(path_name.split('/')[3])

        if without_abstract_counter > 0:
            print('{} papers without abstract'.format(without_abstract_counter))
            
        else:
            print('all papers successfully read')
