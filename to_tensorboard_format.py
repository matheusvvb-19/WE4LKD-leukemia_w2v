##################################################
## Transforms the Word2Vec or FastText embeddings into tsv files, for the TensorFlow Embedding Projector.
##################################################
## Author: {name}
## Copyright: Copyright 2022, Discovering Latent Knowledge in medical paper on Acute Myeloid Leukemia
## Email: {contact_email}
##################################################

# IMPORTS:
from gensim.models import Word2Vec
import numpy as np
import sys
import os
from get_n_common_words_english import get_most_common

# MAIN PROGRAM:
if __name__ == '__main__':
    #create folder if not exists
    os.makedirs('./tensorboard_inputs/', exist_ok=True) 

    #parser
    embedding = sys.argv[1]

    #n first common words
    try:
        n = int(sys.argv[2])
    except Exception:
        n = 10000

    #get most common words to remove from projector file
    list_common_words = get_most_common(n)

    if embedding == 'word2vec':
        model = Word2Vec.load('./word2vec/models_yoy_combination15/model_1921_2022.model')

        metadata = []
        word_vectors = []
        for idx, key in enumerate(model.wv.vocab): 
            metadata.append(key)
            word_vectors.append(model.wv[key].tolist())

        index_to_remove_list = []
        for l in list_common_words:
            try:
                index_to_remove_list.append(metadata.index(l))
            except:
                pass

        metadata = [i for j, i in enumerate(metadata) if j not in index_to_remove_list]
        word_vectors = [i for j, i in enumerate(word_vectors) if j not in index_to_remove_list]

        with open("./tensorboard_inputs/metadata_w2v.tsv", 'w', encoding='utf-8') as output:
            for m in metadata:
                output.write(str(m) + '\n')

        with open("./tensorboard_inputs/vectors_w2v.tsv", 'w', encoding='utf-8') as output:
            for vw in word_vectors:
                vw = map(str, vw)
                output.write('\t'.join(vw) + '\n')
                
    else if embeddings == 'fasttext':
        model = Word2Vec.load('./fasttext/models_yoy_combination16/model_1921_2022.model')

        metadata = []
        word_vectors = []
        for idx, key in enumerate(model.wv.vocab): 
            metadata.append(key)
            word_vectors.append(model.wv[key].tolist())

        index_to_remove_list = []
        for l in list_common_words:
            try:
                index_to_remove_list.append(metadata.index(l))
            except:
                pass

        metadata = [i for j, i in enumerate(metadata) if j not in index_to_remove_list]
        word_vectors = [i for j, i in enumerate(word_vectors) if j not in index_to_remove_list]

        with open("./tensorboard_inputs/metadata_ft.tsv", 'w', encoding='utf-8') as output:
            for m in metadata:
                output.write(str(m) + '\n')

        with open("./tensorboard_inputs/vectors_ft.tsv", 'w', encoding='utf-8') as output:
            for vw in word_vectors:
                vw = map(str, vw)
                output.write('\t'.join(vw) + '\n')
