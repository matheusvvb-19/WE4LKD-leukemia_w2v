import json, string, re
from Bio import Entrez
import os
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
    Entrez.email = 'your@email.com' #change here
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

if __name__ == '__main__':
    for s in search_strings:
        print('searching for {}'.format(s))

        results = search('"' + s + '"')
        id_list = results['IdList']
        len_id_list = len(id_list)
        print('tamanho do idlist: ', len_id_list)
        ids.extend(x for x in id_list if x not in ids) # antes, a lista "ids" era composta por várias listas dentro dela ([[e], a, b, [a, c, d]]), agora é apenas uma lista e com elementos não repetidos ([a, b, c, d, e])
        papers = fetch_details(id_list)
        s = s.lower().replace(' ', '_')
        Path('./results/{}'.format(s)).mkdir(parents=True, exist_ok=True)
        print('{} papers for {} fetched'.format(len_id_list, s))
        counter = 0     # contador

        for i, paper in enumerate(papers['PubmedArticle']):
            try:
                article_title = paper['MedlineCitation']['Article']['ArticleTitle']
                article_abstract = paper['MedlineCitation']['Article']['Abstract']['AbstractText'][0]

                article_title = article_title.translate(str.maketrans('', '', string.punctuation))  # pré-processamento do título
                article_title = article_title.lower().replace(' ', '_')         # pré-processamento do título

                article_year = paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']['Year']    # pegando o ano
            except KeyError as e:
                if 'ArticleTitle' in e.args or 'Abstract' in e.args:    # caso o artigo não tenha título ou resumo, pula ele e segue o loop para o próximo artigo
                    counter += 1
                    continue
                elif 'Year' in e.args:          # caso o artigo não tenha a chave "Year" na data de publicação, pega os 4 primeiros caracteres da "MedlineDate"
                    article_year = paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']['MedlineDate'][0:4]
            
            path_name = './results/{}/{}.txt'.format(s, article_year+'_'+article_title)
            path_name = path_name.encode('ascii', 'ignore').decode('ascii')

            if len(path_name) > 256:
                path_name = path_name[:256]

            # depois de pegar o título, resumo e data (e pulando o loop, quando não é possível), escrever o arquivo
            with open(path_name, "a", encoding='utf-8') as myfile:
                myfile.write(article_abstract)


