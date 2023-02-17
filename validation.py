######################################################################
"""Script de validação do projeto:
    1) buscar nos artigos (texto) prefácios que contenham os compostos utilizados para tratamento de AML
    e que NÃO contenham a palara "AML";

    2) listar os compostos obtidos a partir da busca anterior e calcular o dot product de suas embeddings com a embedding de "AML". 
    Ordenar a lista de compostos a partir do resultado do dot product de forma decrescente, salvo-os em um arquivo csv.
"""
######################################################################


import os, nltk, torch, sys, shutil
from nltk.tokenize import sent_tokenize

from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

from gensim import models
from gensim.models import Word2Vec

from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence

import numpy as np

from scipy.special import softmax

import pandas as pd

from transformers import AutoTokenizer, AutoModel

#sys.path.append('./pubchem/')
#from clean_summaries import ss, get_csv_in_folder, read_csv_table_files, to_csv, get_target_compounds

def ss():
    """Creates a PySpark Session and returns it"""

    sc = SparkContext.getOrCreate()
    return SparkSession(sc)

def get_csv_in_folder(file_path):
    """Search for a .csv file in a given path. It must find just one .csv file - this constraint is tested with assert command.
    This is an auxiliar function used during reading .csv PySpark DataFrames.

    Args:
        file_path: path to the folder containg the .csv file.
    """

    files = os.listdir(file_path)
    files = filter(lambda x: x[-3:] == 'csv', files)
    files = list(files)

    assert len(files) == 1, files

    return os.path.join(file_path, files[0])

def read_csv_table_files(file_path, sep=','):
    full_path = file_path

    if file_path[-3:] != 'csv':
        file_path = get_csv_in_folder(file_path)

    return ss()\
        .read\
        .option('header', 'true')\
        .option('sep', sep)\
        .csv(full_path)

def flat_list(composed_list):
    if any(isinstance(x, list) for x in composed_list):
        composed_list = [item for sublist in composed_list for item in sublist]

    return composed_list

def split_paragraph_to_sentence(text):
    return sent_tokenize(text)

def to_csv(df, target_folder, num_files=1, sep=','):
    """Saves a PySpark Dataframe into .csv file.

    Args:
        df: object of the DataFrame;
        target_folder: path where the .csv is going to be saved;
        num_files: number of .csv files to be created, default is 1.
    """

    return df\
        .coalesce(num_files)\
        .write\
        .mode('overwrite')\
        .option('header', 'true')\
        .option('sep', sep)\
        .format('csv')\
        .save(target_folder)

def get_target_compounds():
    return sorted(['cytarabine', 'daunorubicin', 'azacitidine', 'midostaurin', 'gemtuzumab-ozogamicin', 'vyxeos', 'ivosidenib', 'venetoclax', 'enasidenib', 'gilteritinib', 'glasdegib', 'arsenictrioxide', 'cyclophosphamide', 'dexamethasone', 'idarubicin', 'mitoxantrone', 'pemigatinib', 'prednisone', 'rituximab', 'thioguanine', 'vincristine'])

def write_candidates_compounds_dataframe(candidate_compounds_dict, AML_we, year, w2v_method=1):
    schema = T.StructType([
                T.StructField('compound', T.StringType(), False),
                T.StructField('dot_product_result', T.DoubleType(), True),
                T.StructField('dot_product_result_absolute', T.DoubleType(), True),
                T.StructField('softmax_normalization', T.DoubleType(), False),
                T.StructField('softmax_standartization', T.DoubleType(), False),
            ])

    data = []
    dot_products_for_softmax = []
    for c, cwe in zip(candidate_compounds_dict['compound'], candidate_compounds_dict['word_embedding']):
        dot_product = 0
        dot_product = np.dot(cwe, AML_we).type(torch.DoubleTensor).item()
        dot_products_for_softmax.append(dot_product)

        data.append({
                'compound': c,
                'dot_product_result': dot_product,
                'dot_product_result_abosulte': None,
                'softmax_normalization': None,
                'softmax_standartization': None,
        })

    dot_products_for_softmax = [x * -1 if x < 0 else x for x in dot_products_for_softmax]

    # Normalization:
    maximum = max(dot_products_for_softmax)
    normalized_dot_products = [x/maximum for x in dot_products_for_softmax]
    normalized_dot_products = softmax(normalized_dot_products)

    for index, dp in enumerate(normalized_dot_products):
        data[index]['softmax_normalization'] = dp.item()

    # Standartization:
    standartized_dot_products = preprocessing.scale(dot_products_for_softmax)
    standartized_dot_products = softmax(standartized_dot_products)

    for index, dp in enumerate(standartized_dot_products):
        data[index]['softmax_standartization'] = dp.item()

    # creating PySpark DataFrame:
    candidate_compounds_df = ss().createDataFrame(data=data, schema=schema)
    candidate_compounds_df = candidate_compounds_df\
                            .withColumn('dot_product_result_absolute', F.when(
                                F.col('dot_product_result') > 0, F.col('dot_product_result')).otherwise(-1 * F.col('dot_product_result'))
                            )

    if w2v_method == 1:
        to_csv(candidate_compounds_df, target_folder='./validation/w2v_da/word_embeddings_{}/'.format(year))

    else:
        to_csv(candidate_compounds_df, target_folder='./validation/w2v_avg/word_embeddings_{}/'.format(year))

def clear_hugging_face_cache_folder(dirpath='../../../home/ac4mvvb/.cache/huggingface/hub/'):
    """ Clears the Hugging Face cache folder, to prevent memory error.

        dirpath: the path of the folder.    
    """

    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)

def tensors_to_df(filename):
    """ Convert the tensor files containing the decontextualized word embeddings from BERT-based models to csv files
    
    """

    schema = T.StructType([
                T.StructField('compound', T.StringType(), False),
                T.StructField('dot_product_result', T.DoubleType(), True),
                T.StructField('dot_product_result_absolute', T.DoubleType(), True),
                T.StructField('normalized_dot_product_absolute', T.DoubleType(), True),
                T.StructField('standartized_dot_product_absolute', T.DoubleType(), True),
                T.StructField('softmax_normalization', T.DoubleType(), False),
                T.StructField('softmax_standartization', T.DoubleType(), False),
            ])
    
    tensor_dict = torch.load(filename)

    index_of_AML = list(tensor_dict['compound']).index('AML')
    AML_we = tensor_dict['word_embedding'][index_of_AML]
    
    data = []
    dot_products_for_softmax = []
    for index, c in enumerate(tensor_dict['compound'][0:-1]):
        dot_product = 0
        dot_product = torch.dot(tensor_dict['word_embedding'][index], AML_we).type(torch.DoubleTensor).item()
        dot_products_for_softmax.append(dot_product)

        data.append({
                'compound': c,
                'dot_product_result': dot_product,
                'dot_product_result_abosulte': None,
                'normalized_dot_product_absolute': None,
                'standartized_dot_product_absolute': None,
                'softmax_normalization': None,
                'softmax_standartization': None,
        })

    dot_products_for_softmax = [x * -1 if x < 0 else x for x in dot_products_for_softmax]
    
    # Normalization:
    maximum = np.max(dot_products_for_softmax)
    normalized_dot_products = [x/maximum for x in dot_products_for_softmax]
    for index, dp in enumerate(normalized_dot_products):
        data[index]['normalized_dot_product_absolute'] = dp.item()

    normalized_dot_products = softmax(normalized_dot_products)
    for index, dp in enumerate(normalized_dot_products):
        data[index]['softmax_normalization'] = dp.item()

    # Standartization:
    mean = np.mean(dot_products_for_softmax)
    standart_deviation = np.std(dot_products_for_softmax)
    standartized_dot_products = [(x - mean)/standart_deviation for x in dot_products_for_softmax]
    for index, dp in enumerate(standartized_dot_products):
        data[index]['standartized_dot_product_absolute'] = dp.item()

    standartized_dot_products = softmax(standartized_dot_products)
    for index, dp in enumerate(standartized_dot_products):
        data[index]['softmax_standartization'] = dp.item()
    
    # creating PySpark DataFrame:
    df = ss().createDataFrame(data=data, schema=schema)
    df = df\
        .withColumn('dot_product_result_absolute', F.when(
            F.col('dot_product_result') > 0, F.col('dot_product_result')).otherwise(-1 * F.col('dot_product_result'))
        )

    to_csv(df, target_folder='./validation/bert/word_embeddings_test_{}/'.format(filename[-7:-3]))

if __name__ == '__main__':
    # User Defined Function to split the prefaces ('summary' column) into sentences, without loosing track of the 'ID' and 'filename' (year of publication) columns:
    convertUDF = F.udf(lambda z: split_paragraph_to_sentence(z), T.ArrayType(T.StringType(), False))

    # NLTK module:
    nltk.download('punkt', quiet=True)

    # CONSTANTS:
    VALIDATION_TYPE = 'bert'            # possible values must be: 'bert' or 'w2v'
    WRITE_CANDIDATES_PAPERS = False
    BERT_MODELS_PATH = './bert/distilbert/'
    W2V_MODELS_PATH = './word2vec/models/'

    # if the validation is for BERT-based models, some constants must pointer to specific BERT files and the cache folder must be enough memory:
    if VALIDATION_TYPE == 'bert':
        CLENED_PAPERS_PATH = './bert/results/'
        VALIDATION_FOLDER = './validation/bert/'
        tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
        clear_hugging_face_cache_folder()
        os.makedirs(VALIDATION_FOLDER, exist_ok=True)
        """
        # adding target compounds to tokenizer vocabulary:
        old_vocab = [k for k,v in tokenizer.get_vocab().items()]
        new_vocab = get_target_compounds() + ['aml']
        idx_old_vocab_list = list()
        same_tokens_list = list()
        different_tokens_list = list()

        for idx_new,w in enumerate(new_vocab): 
            try:
                idx_old = old_vocab.index(w)
            except:
                idx_old = -1
            if idx_old>=0:
                idx_old_vocab_list.append(idx_old)
                same_tokens_list.append((w,idx_new))
            else:
                different_tokens_list.append((w,idx_new))

        new_tokens = [k for k,v in different_tokens_list]

        print("[ BEFORE ] tokenizer vocab size: ", len(tokenizer)) 
        added_tokens = tokenizer.add_tokens(new_tokens)

        print("[ AFTER ] tokenizer vocab size: ", len(tokenizer)) 
        print('added_tokens: ',added_tokens)
        """
    
    # if the validation is for Word2Vec models, somen constants must pointer to specific Word2Vec files:
    elif VALIDATION_TYPE == 'w2v':
        MODELS = sorted([f.path for f in os.scandir(W2V_MODELS_PATH) if f.is_file() and f.name.endswith('.model')])
        CLENED_PAPERS_PATH = './pubchem/results/'
        os.makedirs('./validation/w2v_da/', exist_ok=True)
        os.makedirs('./validation/w2v_avg/', exist_ok=True)    

    # creating Spark session:
    spark = ss()

    # reading spark dataframe of papers (columns: id, filename (year) and summary):
    if VALIDATION_TYPE == 'bert':
        df = read_csv_table_files(CLENED_PAPERS_PATH, sep='|')
        
    else:
        df = read_csv_table_files(CLENED_PAPERS_PATH)

    # adding a new column, named 'sentences', that contains each of the sentences present in the prefaces:
    #df = df\
        #.withColumn('filename', df.filename.cast(T.IntegerType()))\
        #.withColumn('sentences', convertUDF(F.col('summary')))\
        #.withColumn('sentences', F.explode(F.col('sentences')))

    # list of years which models were trained (the length of this list is the number of models trained):
    years = sorted(list(df.select('filename').distinct().toPandas()['filename']))
    years = [int(x) for x in years]

    compounds_to_search = get_target_compounds()

    ########################
    # DEBUGGING:
    print('VALIDATION_TYPE: ', VALIDATION_TYPE)
    print('CLENED_PAPERS_PATH: ', CLENED_PAPERS_PATH)    
    print('Years: \n', years)
    print('\nMAIN LOOP:')
    ########################

    """ MAIN LOOP:
        For each year (y) from 1963 (first occurrence of 'AML' in the corpus), do:
        (1) Select the papers used for training the language model. In other words, select the papers published in 1963 and before.
            From that set of papers, select the ones that contains the word 'AML' and the ones that contains any of the target compounds.

        (2) Search for the compounds (get_target_compounds()) present in the selected papers. For each of them, compute it word embedding:
            (2.1)
        (2) Caso seja encontrada a palavra "AML", é necessário agora verificar se há compostos candidatos a conhecimento latente.
            Para ser considerado candidato a conhecimento latente, o composto NÃO deve aparecer em um mesmo artigo que cite "AML".
            (2.1) Se o DataFrame df_training_candidates não conter nenhuma tupla, significa que não foi encontrado nenhum candidato a conhecimento latente
                  nos artigos publicados até y-1 (ano anterior) - então, busca-se pelos compostos em todos os artigos publicados até o ano y.
                  Se já existirem compostos candidatos a conhecimento latente encontrados nos anos anteriores, basta unir esse DataFrame com
                  o DataFrame retornado pela busca de compostos nos artigos publicados exatamente no ano y.

        (3) Caso haja artigos que apresentem compostos candidatos a conhecimento latente, é necessário agora identificar exatamente quais são esses compostos presentes nos artigos
            Após a identificação dos compostos, acessar a word embedding de cada um deles:
                - se os modelos forem BERT, é necessário calcular a média aritmética para cada ocorrência do composto
                - se os modelos forem Word2Vec, basta extrair a word embedding diretamente
            Armazenar os compostos e suas respectivas word embeddings em uma lista, dicionário ou outro tipo de variável
            Realizar os dois passos anteriores novamente, mas agora para a palavra "AML"

        (4) Calcular o dot product entre as word embeddings dos compostos e a word embedding de "AML".
            Ordenar os compostos de forma descrescente, segundo o resultado do dot product e escrever em um arquivo csv.
    """
    
    for index_y, y in enumerate(years):
        print('\nCurrent year of analysis: {}'.format(y))
        
        if y >= 1963:
            print("There are at least one paper published until {} that cotains the word 'AML'".format(y))

            # step (1): selecting the papers published until y
            df_training = df.filter(F.col('filename') <= y)

            df_training_AML = df_training.filter(F.col('sentences').rlike('AML'))

            df_training_compounds = None
            regex = '(?i)({})'.format('|'.join(compounds_to_search))
            df_training_compounds = df_training.filter(F.col('sentences').rlike(regex))

            # step (3):
            if len(df_training_compounds.take(1)) > 0:
                print('There are at least one paper published until {} that contains any of the target compounds, shown below:'.format(y))
                df_training_compounds.show()

                candidate_compounds_dict = {
                    'compound': [],
                    'word_embedding': [],
                }

                print('Identifying exactly which compounds are present in the selected papers')
                candidates_compounds = []
                for c in compounds_to_search:
                    regex = '(?i)({})'.format(c)
                    df_aux = df_training_compounds.filter(F.col("sentences").rlike(c))

                    if len(df_aux.take(1)) > 0:
                        candidates_compounds.append(c)  

                    else:
                        continue                

                candidate_compounds_dict['compound'] = candidates_compounds
                print('The compounds present in the selected papers are:')
                print(candidates_compounds)

                # validation of BERT-based models:
                if VALIDATION_TYPE == 'bert':
                    print('\nLoading BERT-based model')

                    # cleaning again the cache folder, because it its occupped each time a BERT-based model is loaded from Hugging Face:
                    clear_hugging_face_cache_folder()

                    model = AutoModel.from_pretrained('matheusvolpon/WE4LKD_AML_distilbert_1921_{}'.format(y))
                    """
                    model.resize_token_embeddings(len(tokenizer)) 

                    print('Accessing the token embeddings for compounds')
                    for c in candidates_compounds:
                        index_of_compound = tokenizer.vocab[c]
                        input_id = torch.tensor([index_of_compound])
                        compound_we = model.get_input_embeddings()(input_id).mean(0)

                        candidate_compounds_dict['word_embedding'].append(compound_we)

                    print('Accessing the token embedding for AML')
                    index_of_AML = tokenizer.vocab['aml']
                    input_id = torch.tensor([index_of_AML])
                    AML_we = model.get_input_embeddings()(input_id).mean(0)
                    """

                    candidate_compounds_dict['compound'].append('AML')
                    candidate_compounds_dict['word_embedding'].append(AML_we)

                    torch.save(candidate_compounds_dict, VALIDATION_FOLDER + 'word_embeddings_test2_{}.pt'.format(y))
                    #tensors_to_df(filename=VALIDATION_FOLDER + 'word_embeddings_test_{}.pt'.format(y))
                    
                    """
                    # FlairNLP framework to extract contextualized word embeddings:
                    embedding = TransformerWordEmbeddings(model='matheusvolpon/WE4LKD_AML_distilbert_1921_{}'.format(y), subtoken_pooling='mean', layers='-1,-2,-3,-4')

                    print('Computing the word embedding for the compounds present in the papers')
                    # for each candidate compound, select the sentences that contains it (save into a list) and compute the average of the word embedding for that compound in all contexts which it appears:
                    for c in candidates_compounds:
                        contextualized_word_embeddings = []

                        regex = '(?i)({})'.format(c)
                        sentences = list(
                                        df_training_compounds\
                                        .select(F.col('sentences'))\
                                        .filter(F.col("sentences").rlike(regex))\
                                        .toPandas()['sentences']
                                    )

                        if len(sentences) > 0:
                            for sent in sentences:
                                embeded_sentence = Sentence(sent)

                                try:
                                    embedding.embed(embeded_sentence)
                                
                                except:
                                    continue

                                # get the indexes of the target compound in the embeded sentence:
                                indexes = []
                                for index, token in enumerate(embeded_sentence):
                                    if c in token.text.lower():
                                        indexes.append(index)
                                
                                for idx in indexes:
                                    contextualized_word_embeddings.append(embeded_sentence[idx].embedding)

                            decontextualized_word_embedding = torch.mean(torch.stack(contextualized_word_embeddings), dim=0)
                            candidate_compounds_dict['word_embedding'].append(decontextualized_word_embedding)

                    print("Computing the word embedding for the word 'AML' in the papers")
                    # computing the word embedding for the word "AML":                                        
                    sentences = list(
                                    df_training_AML\
                                    .select(F.col('sentences'))\
                                    .toPandas()['sentences']
                                )

                    if len(sentences) > 0:
                        contextualized_word_embeddings = []
                        
                        for sent in sentences:
                            embeded_sentence = Sentence(sent)

                            try:
                                embedding.embed(embeded_sentence)
                            
                            except:
                                continue

                            # get the indexes of the target compound in the embeded sentence:
                            indexes = []
                            for index, token in enumerate(embeded_sentence):
                                if 'AML' in token.text:
                                    indexes.append(index)
                            
                            for idx in indexes:
                                contextualized_word_embeddings.append(embeded_sentence[idx].embedding)

                        decontextualized_word_embedding = torch.mean(torch.stack(contextualized_word_embeddings), dim=0)

                        candidate_compounds_dict['compound'].append('AML')
                        candidate_compounds_dict['word_embedding'].append(decontextualized_word_embedding)

                        torch.save(candidate_compounds_dict, VALIDATION_FOLDER + 'word_embeddings_test_{}.pt'.format(y))

                        tensors_to_df(filename=VALIDATION_FOLDER + 'word_embeddings_test_{}.pt'.format(y))
                    """
                # validation of Word2Vec models:
                elif VALIDATION_TYPE == 'w2v':
                    # loading Word2Vec model from file:
                    print('Loading Word2Vec model {}'.format([x for x in MODELS if str(y) in x][0]))
                    model = Word2Vec.load([x for x in MODELS if str(y) in x][0])

                    AML_we = model.wv['aml']

                    # MÉTODO 1: acessando diretamente a word embedding do composto isolado, se houver:
                    for c in candidates_compounds:
                        try:
                            index_of_word_in_vocab = list(model.wv.vocab).index(c)
                        
                        except ValueError: # não existe o token do composto (c) no vocabulário
                            words = [x for x in list(model.wv.vocab) if c in x]
                            if len(words) > 0:
                                index_of_word_in_vocab = list(model.wv.vocab).index(words[0])

                            else:
                                candidate_compounds_dict['compound'].remove(c)
                                continue
                        
                        candidate_compounds_dict['word_embedding'].append(model.wv.vectors[index_of_word_in_vocab])                    

                    write_candidates_compounds_dataframe(candidate_compounds_dict, AML_we, y, w2v_method=1)

                    # MÉTODO 2: calculando a média das embeddings que contém o composto:
                    candidate_compounds_dict.update({'word_embedding': []})

                    for c in candidates_compounds:
                        words_with_the_compound = [x for x in model.wv.vocab if c in x]

                        indexes_of_words_with_the_compound = []
                        for w in words_with_the_compound:
                            indexes_of_words_with_the_compound.append(list(model.wv.vocab).index(w))

                        word_embeddings = []
                        for idx in indexes_of_words_with_the_compound:
                            word_embeddings.append(model.wv.vectors[idx])
                        
                        avg = 0
                        for we in word_embeddings:
                            avg += we
                        
                        avg = avg / len(word_embeddings)
                        candidate_compounds_dict['word_embedding'].append(avg)  

                    write_candidates_compounds_dataframe(candidate_compounds_dict, AML_we, y, w2v_method=2)

            else:
                print('There are no papers published until {} that contains any of the target compounds'.format(y))
        
        else:
            print("There are no papers published until {} that contains the word 'AML', mandatory for the validation process".format(y))
            continue
    

    os.makedirs('./validation/per_compound/', exist_ok=True)
    os.makedirs('./validation/per_compound/w2v_da/', exist_ok=True)
    os.makedirs('./validation/per_compound/w2v_avg/', exist_ok=True)
    os.makedirs('./validation/per_compound/bert/', exist_ok=True)

    if VALIDATION_TYPE == 'w2v':
        for c in get_target_compounds():
            print('Compound: {}'.format(c))

            compound_dict_method1 = {
                'year': [],
                'dot_product_result': [],
                'dot_product_result_absolute': [],
                'normalized_dot_product_abolsute': [],
                'standartized_dot_product_absolute': [],
                'softmax_normalization': [],
                'softmax_standartization': [],
            }

            compound_dict_method2 = {
                'year': [],
                'dot_product_result': [],
                'dot_product_result_absolute': [],
                'normalized_dot_product_abolsute': [],
                'standartized_dot_product_absolute': [],
                'softmax_normalization': [],
                'softmax_standartization': [],
            }

            for index_y, y in enumerate(years):
                print('\nYear of analysis: {}'.format(y))

                if int(y) >= 1963:
                    # loading model from file:
                    print('Loading Word2Vec model {}'.format([x for x in MODELS if str(y) in x][0]))
                    model = Word2Vec.load([x for x in MODELS if str(y) in x][0])

                    AML_we = model.wv['aml']

                    # METHOD 1:
                    try:
                        index_of_word_in_vocab = list(model.wv.vocab).index(c)
                    
                    except ValueError: # não existe o token do composto (c) no vocabulário
                        words = [x for x in list(model.wv.vocab) if c in x]
                        if len(words) > 0:
                            index_of_word_in_vocab = list(model.wv.vocab).index(words[0])

                        else:
                            continue

                    compound_we = model.wv.vectors[index_of_word_in_vocab]
                    
                    dot_product = np.dot(compound_we, AML_we).type(torch.DoubleTensor).item()

                    compound_dict_method1['year'].append(y)
                    compound_dict_method1['dot_product_result'].append(dot_product)
                    if dot_product < 0:
                        compound_dict_method1['dot_product_result_absolute'].append(dot_product * -1)
                    
                    else:
                        compound_dict_method1['dot_product_result_absolute'].append(dot_product)
                    
                    # METHOD 2:
                    words_with_the_compound = [x for x in model.wv.vocab if c in x]

                    indexes_of_words_with_the_compound = []
                    for w in words_with_the_compound:
                        indexes_of_words_with_the_compound.append(list(model.wv.vocab).index(w))

                    word_embeddings = []
                    for idx in indexes_of_words_with_the_compound:
                        word_embeddings.append(model.wv.vectors[idx])
                    
                    avg = 0
                    for we in word_embeddings:
                        avg += we
                    
                    compound_we = avg / len(word_embeddings)

                    dot_product = np.dot(compound_we, AML_we).type(torch.DoubleTensor).item()

                    compound_dict_method2['year'].append(y)
                    compound_dict_method2['dot_product_result'].append(dot_product)
                    if dot_product < 0:
                        compound_dict_method2['dot_product_result_absolute'].append(dot_product * -1)
                    
                    else:
                        compound_dict_method2['dot_product_result_absolute'].append(dot_product)
                    
                    print('Dot products computed')
                
                else:
                    print("não há artigos publicados até {} que contenham a palavra 'AML'".format(y))
                    continue

            print('method1, dot_product_result: ', len(compound_dict_method1['dot_product_result']))
            print('method1, dot_product_result_absolute: ', len(compound_dict_method1['dot_product_result_absolute']))
            print('method2, dot_product_result: ', len(compound_dict_method2['dot_product_result']))
            print('method2, dot_product_result_absolute: ', len(compound_dict_method2['dot_product_result_absolute']))

            # Normalization:
            maximum = np.max(compound_dict_method1['dot_product_result_absolute'])
            compound_dict_method1['normalized_dot_product_absolute'] = [x/maximum for x in compound_dict_method1['dot_product_result_absolute']]
            compound_dict_method1['softmax_normalization'] = softmax(compound_dict_method1['normalized_dot_product_absolute'])

            maximum = np.max(compound_dict_method2['dot_product_result_absolute'])
            compound_dict_method2['normalized_dot_product_absolute'] = [x/maximum for x in compound_dict_method2['dot_product_result_absolute']]
            compound_dict_method2['softmax_normalization'] = softmax(compound_dict_method2['normalized_dot_product_absolute'])

            print('method1, normalized_dot_product_absolute: ', len(compound_dict_method1['normalized_dot_product_absolute']))
            print('method1, softmax_normalization: ', len(compound_dict_method1['softmax_normalization']))
            print('method2, normalized_dot_product_absolute: ', len(compound_dict_method2['normalized_dot_product_absolute']))
            print('method2, softmax_normalization: ', len(compound_dict_method2['softmax_normalization']))

            # Standartization:
            mean = np.mean(compound_dict_method1['dot_product_result_absolute'])
            standart_deviation = np.std(compound_dict_method1['dot_product_result_absolute'])
            compound_dict_method1['standartized_dot_product_absolute'] = [(x - mean)/standart_deviation for x in compound_dict_method1['dot_product_result_absolute']]
            compound_dict_method1['softmax_standartization'] = softmax(compound_dict_method1['standartized_dot_product_absolute'])

            mean = np.mean(compound_dict_method2['dot_product_result_absolute'])
            standart_deviation = np.std(compound_dict_method2['dot_product_result_absolute'])
            compound_dict_method2['standartized_dot_product_absolute'] = [(x - mean)/standart_deviation for x in compound_dict_method2['dot_product_result_absolute']]
            compound_dict_method2['softmax_standartization'] = softmax(compound_dict_method2['standartized_dot_product_absolute'])

            print('method1, standartized_dot_product_absolute: ', len(compound_dict_method1['standartized_dot_product_absolute']))
            print('method1, softmax_standartization: ', len(compound_dict_method1['softmax_standartization']))
            print('method2, standartized_dot_product_absolute: ', len(compound_dict_method2['standartized_dot_product_absolute']))
            print('method2, softmax_standartization: ', len(compound_dict_method2['softmax_standartization']))

            pd.DataFrame.from_dict(data=compound_dict_method1).to_csv('./validation/per_compound/w2v_da/{}.csv'.format(c), columns=['year', 'dot_product_result', 'dot_product_result_absolute', 'normalized_dot_product_absolute', 'standartized_dot_product_absolute', 'softmax_normalization', 'softmax_standartization'], index=False)
            pd.DataFrame.from_dict(data=compound_dict_method2).to_csv('./validation/per_compound/w2v_avg/{}.csv'.format(c), columns=['year', 'dot_product_result', 'dot_product_result_absolute', 'normalized_dot_product_absolute', 'standartized_dot_product_absolute', 'softmax_normalization', 'softmax_standartization'], index=False)
        
    elif VALIDATION_TYPE == 'bert':
        BERT_EMBEDDINGS = sorted([f.path for f in os.scandir(VALIDATION_FOLDER) if f.is_file() and f.name.endswith('.pt')])
        BERT_EMBEDDINGS = [x for x in BERT_EMBEDDINGS if 'test' in x]

        for c in get_target_compounds():
            print('\Compound: {}'.format(c))

            compound_dict = {
                'year': [],
                'dot_product_result': [],
                'dot_product_result_absolute': [],
                'normalized_dot_product_absolute': [],
                'standartized_dot_product_absolute': [],
                'softmax_normalization': [],
                'softmax_standartization': [],
            }

            for be in BERT_EMBEDDINGS:
                print(be)

                tensor_dict = torch.load(be)

                try:
                    index_compound = tensor_dict['compound'].index(c)
                    index_AML = tensor_dict['compound'].index('AML')

                    AML_we = tensor_dict['word_embedding'][index_AML]
                    compound_we = tensor_dict['word_embedding'][index_compound]
                
                except:
                    continue

                            
                dot_product = torch.dot(compound_we, AML_we).type(torch.DoubleTensor).item()
                if dot_product > 0:
                    dot_product_absolute = dot_product
                
                else:
                    dot_product_absolute =  -1 * dot_product

                compound_dict['year'].append(int(be.split('.pt')[0][-4:]))
                compound_dict['dot_product_result'].append(dot_product)
                compound_dict['dot_product_result_absolute'].append(dot_product_absolute)
                print('Dot products computed')

            # Normalization:
            maximum = np.max(compound_dict['dot_product_result_absolute'])
            compound_dict['normalized_dot_product_absolute'] = [x/maximum for x in compound_dict['dot_product_result_absolute']]
            compound_dict['softmax_normalization'] = softmax(compound_dict['normalized_dot_product_absolute'])

            # Standartization:
            #compound_dict['softmax_standartization'] = preprocessing.scale(compound_dict['dot_product_result_absolute'])
            mean = np.mean(compound_dict['dot_product_result_absolute'])
            standart_deviation = np.std(compound_dict['dot_product_result_absolute'])
            compound_dict['standartized_dot_product_absolute'] = [(x - mean)/standart_deviation for x in compound_dict['dot_product_result_absolute']]
            compound_dict['softmax_standartization'] = softmax(compound_dict['standartized_dot_product_absolute'])

            print('writing compound csv file: {}'.format(c))
            pd.DataFrame.from_dict(data=compound_dict).to_csv('./validation/per_compound/bert/{}_test2.csv'.format(c), columns=['year', 'dot_product_result', 'dot_product_result_absolute', 'normalized_dot_product_absolute', 'standartized_dot_product_absolute', 'softmax_normalization', 'softmax_standartization'], index=False)
    
    print('END!')
