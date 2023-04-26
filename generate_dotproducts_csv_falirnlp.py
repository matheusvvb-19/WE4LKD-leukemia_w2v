##################################################
## Generates the csv files containing the dt prodcuts results between compounds and 'AML', using the FalirNLP framework (BERT-based models).
##################################################
## Author: {name}
## Copyright: Copyright 2022, Discovering Latent Knowledge in medical paper on Acute Myeloid Leukemia
## Email: {contact_email}
##################################################

# IMPORTS:
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
sys.path.append('./pubchem/')
from clean_summaries import ss, get_csv_in_folder, read_csv_table_files, to_csv, get_target_compounds

# FUNCTIONS:
def flat_list(composed_list):
    if any(isinstance(x, list) for x in composed_list):
        composed_list = [item for sublist in composed_list for item in sublist]

    return composed_list

def split_paragraph_to_sentence(text):
    return sent_tokenize(text)

def clear_hugging_face_cache_folder(dirpath='/home/doubleblind/.cache/huggingface/hub/'):
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
    """Convert the tensor files containing the decontextualized word embeddings from BERT-based models to csv files"""

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
    print(tensor_dict['compound'])
    print(len(tensor_dict['word_embedding']))

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

    to_csv(df, target_folder='./validation/bert/word_embeddings_{}/'.format(filename[-7:-3]))

# MAIN PROGRAM:
if __name__ == '__main__':
    # User Defined Function to split the prefaces ('summary' column) into sentences, without loosing track of the 'ID' and 'filename' (year of publication) columns:
    convertUDF = F.udf(lambda z: split_paragraph_to_sentence(z), T.ArrayType(T.StringType(), False))

    # NLTK module:
    nltk.download('punkt', quiet=True)

    # CONSTANTS:
    VALIDATION_TYPE = 'bert'            # possible values must be: 'bert'
    BERT_MODELS_PATH = './bert/distilbert/'

    # if the validation is for BERT-based models, some constants must pointer to specific BERT files and the cache folder must be enough memory:
    if VALIDATION_TYPE == 'bert':
        CLENED_PAPERS_PATH = './bert/results/'
        VALIDATION_FOLDER = './validation/bert/'
        tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
        clear_hugging_face_cache_folder()
        os.makedirs(VALIDATION_FOLDER, exist_ok=True) 

    # creating Spark session:
    spark = ss()

    # reading spark dataframe of papers (columns: id, filename (year) and summary):
    if VALIDATION_TYPE == 'bert':
        df = read_csv_table_files(CLENED_PAPERS_PATH, sep='|')

    # adding a new column, named 'sentences', that contains each of the sentences present in the prefaces:
    df = df\
        .withColumn('filename', df.filename.cast(T.IntegerType()))\
        .withColumn('sentences', convertUDF(F.col('summary')))\
        .withColumn('sentences', F.explode(F.col('sentences')))

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
                    
                    # FlairNLP framework to extract contextualized word embeddings:
                    embedding = TransformerWordEmbeddings(model='doubleblind{}'.format(y), subtoken_pooling='mean', layers='-1,-2,-3,-4')

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
                            print('sentences compound > 0')
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
                        print('sentences AML > 0')
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

                        torch.save(candidate_compounds_dict, VALIDATION_FOLDER + 'word_embeddings_{}.pt'.format(y))
                        tensors_to_df(filename=VALIDATION_FOLDER + 'word_embeddings_{}.pt'.format(y))

            else:
                print('There are no papers published until {} that contains any of the target compounds'.format(y))
        
        else:
            print("There are no papers published until {} that contains the word 'AML', mandatory for the validation process".format(y))
            continue
    
    os.makedirs('./validation/per_compound/bert_flairnlp/', exist_ok=True)

    if VALIDATION_TYPE == 'bert':
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
            mean = np.mean(compound_dict['dot_product_result_absolute'])
            standart_deviation = np.std(compound_dict['dot_product_result_absolute'])
            compound_dict['standartized_dot_product_absolute'] = [(x - mean)/standart_deviation for x in compound_dict['dot_product_result_absolute']]
            compound_dict['softmax_standartization'] = softmax(compound_dict['standartized_dot_product_absolute'])

            print('writing compound csv file: {}'.format(c))
            pd.DataFrame.from_dict(data=compound_dict).to_csv('./validation/per_compound/bert_flairnlp/{}.csv'.format(c), columns=['year', 'dot_product_result', 'dot_product_result_absolute', 'normalized_dot_product_absolute', 'standartized_dot_product_absolute', 'softmax_normalization', 'softmax_standartization'], index=False)
    
    print('END!')
