# based on https://towardsdatascience.com/visualizing-word-embedding-with-pca-and-t-sne-961a692509f5

# IMPORTS:
import plotly, pickle, csv, re, string, zipfile, gdown, os
import plotly.graph_objs as go
import streamlit as st
import pandas as pd
import numpy as np
from get_n_common_words_english import get_most_common
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from random import random, seed
from collections import OrderedDict
from gensim.models import Word2Vec

# FUNCTIONS:
@st.cache()
def download_w2v_models():
    dictionary_of_files = {
        '19: 1921 - 2022': ['https://drive.google.com/file/d/1izHlD2j5V7tIQ8TEmc3qyp-_9cLGQNNU/view?usp=sharing', 'https://drive.google.com/file/d/1Kv-7_EYUc-RAsEXYZUFP_iuTkeIcTEht/view?usp=sharing', 'https://drive.google.com/file/d/1M_9dZa1fIYiOo296OSJexAM74HnqZaMb/view?usp=sharing'],
        '18: 1921 - 2018': 'https://drive.google.com/file/d/1dWasQHwzNZelz4Kjxno_Xbb6eHYO58VB/view?usp=sharing',
        '17: 1921 - 2014': 'https://drive.google.com/file/d/16JCH139NZsFH4Rd5cmnyHCqz0dbNlAXz/view?usp=sharing',
        '16: 1921 - 2013': 'https://drive.google.com/file/d/1kSjA5LbWKqzcdM0nZpmJhpFSEc_v0OYo/view?usp=sharing',
        '15: 1921 - 2011': 'https://drive.google.com/file/d/1HUYb_FiCgOncLn75h-8kciY2B79dd6Ko/view?usp=sharing',
        '14: 1921 - 2009': 'https://drive.google.com/file/d/1Pr39aSMOsJUeJunFVqa_nAv1FBoCWkAH/view?usp=sharing',
        '13: 1921 - 2001': 'https://drive.google.com/file/d/1wxy6lslVtxhOhY1Lr5BR3CZpo7007ekA/view?usp=sharing',
        '12: 1921 - 1999': 'https://drive.google.com/file/d/1hZjlZroZLLnmAZAJSCTfiirs5Vx890zc/view?usp=sharing',
        '11: 1921 - 1998': 'https://drive.google.com/file/d/1G2nS6Bt__j3xcPtCu_y8G4O1IDma7Bdk/view?usp=sharing',
        '10: 1921 - 1995': 'https://drive.google.com/file/d/1wC8CSHVrvXAcSjfzwBfs4VqS81Mp3d5s/view?usp=sharing',
        '09: 1921 - 1983': 'https://drive.google.com/file/d/1VffGYoLMTgjmM8Ye_TanlN8zqikQ1P15/view?usp=sharing',
        '08: 1921 - 1982': 'https://drive.google.com/file/d/13SkRCZmhzAxarNA9zD-fMPqfX-tvVK3d/view?usp=sharing',
        '07: 1921 - 1977': 'https://drive.google.com/file/d/1ElQz0JlrCZslaLiEvSjeAl5nEU4-ErC7/view?usp=sharing',
        '06: 1921 - 1976': 'https://drive.google.com/file/d/1u5lHlU3VHSNmfeyt0BNNt2gOZ5icfmik/view?usp=sharing',
        '05: 1921 - 1974': 'https://drive.google.com/file/d/1r4lQKhr4hkFRP0uioB9Sn9dyI5-sE4ZT/view?usp=sharing',
        '04: 1921 - 1971': 'https://drive.google.com/file/d/1aIv-qGoVaNNSb6H9A5ukfKKdY6RwNofh/view?usp=sharing',
        '03: 1921 - 1969': 'https://drive.google.com/file/d/1HIgfbJwcX22oaZm1zJNBxZTpW_YdujdH/view?usp=sharing',
        '02: 1921 - 1967': 'https://drive.google.com/file/d/1XEzPlJmMFHsxLvfm0r9mU85E31jTBKY_/view?usp=sharing',
        '01: 1921 - 1963': 'https://drive.google.com/file/d/1BNNRdCMl98pOFxIvg7szHj1SlClUy1no/view?usp=sharing',
    }

    os.makedirs('./models_streamlit_app/', exist_ok=True)
    for k, v in dictionary_of_files.items():
        if isinstance(v, str):
            gdown.download(v, './models_streamlit_app/model_1921_{}.model'.format(k[-4:]), quiet=False, fuzzy=True)
        
        else:
            for index, u in enumerate(v):
                if index == 0:
                    gdown.download(u, './models_streamlit_app/model_1921_{}.model'.format(k[-4:]), quiet=False, fuzzy=True)
                
                elif index == 1:
                    gdown.download(u, './models_streamlit_app/model_1921_{}.model.trainables.syn1neg.npy'.format(k[-4:]), quiet=False, fuzzy=True)
                    
                else:
                    gdown.download(u, './models_streamlit_app/model_1921_{}.model.wv.vectors.npy'.format(k[-4:]), quiet=False, fuzzy=True)
    
@st.cache()
def get_target_compounds():
    return ['cytarabine', 'daunorubicin', 'azacitidine', 'midostaurin', 'gemtuzumab-ozogamicin', 'vyxeos', 'ivosidenib', 'venetoclax', 'enasidenib', 'gilteritinib', 'glasdegib', 'arsenictrioxide', 'cyclophosphamide', 'dexamethasone', 'idarubicin', 'mitoxantrone', 'pemigatinib', 'prednisone', 'rituximab', 'thioguanine', 'vincristine']

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

@st.cache()
def read_matched_synonyms():
    df = pd.read_csv('./matched_synonyms.csv', sep=',', escapechar='\\')
    df = df[df['synonym'].map(len) >= 3]

    return df

@st.cache(suppress_st_warning=True, max_entries=10, ttl=2400)
def create_entities_lists():
    '''Creates the lists of possible entity filters by reading the words in the .txt files. Only executed once.'''
    
    list_drugs_chemicals = list_from_txt('./ner/list_drugs_chemicals.txt')
    list_dna_rna = list_from_txt('./ner/list_dna_rna.txt')
    list_proteins = list_from_txt('./ner/list_proteins.txt')
    list_cellular = list_from_txt('./ner/list_cellular.txt')
    
    return list_drugs_chemicals, list_dna_rna, list_proteins, list_cellular

@st.cache(suppress_st_warning=True, max_entries=10, ttl=2400)
def read_fda_drugs_file():
    words_list = []
    with open('fda_drugs.txt', newline = '') as file_txt:                                                                                          
        file_line = csv.reader(file_txt, delimiter='\t')
        for e in file_line:
            if len(e) == 8:
                s = e[5]
                s = re.sub('<[^>]+>', '', s)
                s = re.sub('\\s+', ' ', s)
                s = re.sub('([--:\w?@%&+~#=]*\.[a-z]{2,4}\/{0,2})((?:[?&](?:\w+)=(?:\w+))+|[--:\w?@%&+~#=]+)?', '', s)
                s = s.lower()
                s = s.translate(str.maketrans('', '', string.punctuation.replace('-', '')))
                words_list.append(s)

    words_list.pop(0)
    words_list = list(dict.fromkeys(words_list))
    
    return words_list
    
@st.cache(suppress_st_warning=True, max_entries=10, ttl=2400)
def read_domain_table():
    domains_table = pd.read_csv('https://docs.google.com/spreadsheets/d/' + 
                   '1SgYG4gZuL3grEHHAZt49dAUw_jFAc4LADajFeGAf2-w' +
                   '/export?gid=0&format=csv',
                  )
    
    return domains_table
    
def similarities_table_streamlit(words_list, model):  
    '''Creates and prints the similarity table between the base compounds and the terms searched by the user.
    
    Args:
      words_list: the words selected by the user.
      model: the Word2Vec model, needed for the Gensim functions. 
    '''

    table = [['Word']]
    for w in base_compounds:
        if w in model.wv.vocab:
            table[0].append(w)

    for w in words_list:
        if w in model.wv.vocab:
            row = [w]
            for y in table[0][1:]:
                if w == y:
                    row.append('---')
                else:
                    similarity = round(float(model.wv.similarity(y, w)), 2)
                    rank = model.wv.rank(y, w)
                    row.append('{}, {}°'.format(similarity, rank))
            table.append(row)
            
    df = pd.DataFrame(table)
    st.table(df)
    
def restrict_w2v(w2v, restricted_word_set, domain=False):
    '''Restrict the vocabulary of certain model, removing words according to an especific domain.
    
    Args:
      w2v: Word2Vec model.
      restricted_word_set: list of words of the domain.
      domain: boolean that informs how to remove the words (remove words that belong or not to the domain).
    '''

    new_vectors = []
    new_vocab = {}
    new_index2entity = []
    new_vectors_norm = []

    if domain == False:
      for i in range(len(w2v.vocab)):
          word = w2v.index2entity[i]
          vec = w2v.vectors[i]
          vocab = w2v.vocab[word]
          vec_norm = w2v.vectors_norm[i]
          if word not in restricted_word_set:
              vocab.index = len(new_index2entity)
              new_index2entity.append(word)
              new_vocab[word] = vocab
              new_vectors.append(vec)
              new_vectors_norm.append(vec_norm)
    else:
      for i in range(len(w2v.vocab)):
          word = w2v.index2entity[i]
          vec = w2v.vectors[i]
          vocab = w2v.vocab[word]
          vec_norm = w2v.vectors_norm[i]
          if word in restricted_word_set:
              vocab.index = len(new_index2entity)
              new_index2entity.append(word)
              new_vocab[word] = vocab
              new_vectors.append(vec)
              new_vectors_norm.append(vec_norm)

    w2v.vocab = new_vocab
    w2v.vectors = np.array(new_vectors)
    w2v.index2entity = np.array(new_index2entity)
    w2v.index2word = np.array(new_index2entity)
    w2v.vectors_norm = np.array(new_vectors_norm)
    
def wv_restrict_w2v(w2v, restricted_word_set, domain=False):
    '''The same of above function but usin the ".wv" before the acceses to the model's properties.'''

    new_vectors = []
    new_vocab = {}
    new_index2entity = []
    new_vectors_norm = []

    if domain == False:
      for i in range(len(w2v.wv.vocab)):
          word = w2v.wv.index2entity[i]
          vec = w2v.wv.vectors[i]
          vocab = w2v.wv.vocab[word]
          vec_norm = w2v.wv.vectors_norm[i]
          if word not in restricted_word_set:
              vocab.index = len(new_index2entity)
              new_index2entity.append(word)
              new_vocab[word] = vocab
              new_vectors.append(vec)
              new_vectors_norm.append(vec_norm)
    else:
      for i in range(len(w2v.wv.vocab)):
          word = w2v.wv.index2entity[i]
          vec = w2v.wv.vectors[i]
          vocab = w2v.wv.vocab[word]
          vec_norm = w2v.wv.vectors_norm[i]
          if word in restricted_word_set:
              vocab.index = len(new_index2entity)
              new_index2entity.append(word)
              new_vocab[word] = vocab
              new_vectors.append(vec)
              new_vectors_norm.append(vec_norm)

    w2v.wv.vocab = new_vocab
    w2v.wv.vectors = np.array(new_vectors)
    w2v.wv.index2entity = np.array(new_index2entity)
    w2v.wv.index2word = np.array(new_index2entity)
    w2v.wv.vectors_norm = np.array(new_vectors_norm)

def append_list(sim_words, words):
    list_of_words = []
    for i in range(len(sim_words)):
        sim_words_list = list(sim_words[i])
        sim_words_list.append(words)
        sim_words_tuple = tuple(sim_words_list)
        list_of_words.append(sim_words_tuple)
        
    return list_of_words

def display_scatterplot_3D(model, user_input=None, words=None, label=None, color_map=None, annotation='On',  dim_red = 'TSNE', perplexity = 5, learning_rate = 0, iteration = 0, topn=0, sample=10):
    if words == None:
        if sample > 0:
            words = np.random.choice(list(model.wv.vocab.keys()), sample)
        else:
            words = [word for word in model.wv.vocab]
    
    word_vectors = np.array([model.wv[w] for w in words])
    
    if len(word_vectors) > 0:
        if dim_red == 'PCA':
            three_dim = PCA(random_state=0).fit_transform(word_vectors)[:,:3]
        else:
            three_dim = TSNE(n_components = 3, random_state=0, perplexity = perplexity, learning_rate = learning_rate, n_iter = iteration).fit_transform(word_vectors)[:,:3]

        data = []
        count = 0
        for i in range (len(user_input)):

                    trace = go.Scatter3d(
                        x = three_dim[count:count+topn,0], 
                        y = three_dim[count:count+topn,1],  
                        z = three_dim[count:count+topn,2],
                        text = words[count:count+topn] if annotation == 'On' else '',
                        customdata = words,
                        hovertemplate =
                          "<b>Word</b>: %{customdata}",
                        name = user_input[i],
                        textposition = "top center",
                        textfont_size = 30,
                        mode = 'markers+text',
                        marker = {
                            'size': 10,
                            'opacity': 0.8,
                            'color': 2
                        }
                    )

                    data.append(trace)
                    count = count+topn

        trace_input = go.Scatter3d(
                        x = three_dim[count:,0], 
                        y = three_dim[count:,1],  
                        z = three_dim[count:,2],
                        text = words[count:],
                        name = 'input words',
                        textposition = "top center",
                        textfont_size = 30,
                        mode = 'markers+text',
                        marker = {
                            'size': 10,
                            'opacity': 1,
                            'color': 'black'
                        }
                        )

        data.append(trace_input)

    # Configure the layout.
        layout = go.Layout(
            margin = {'l': 0, 'r': 0, 'b': 0, 't': 0},
            showlegend=True,
            legend=dict(
            orientation='h',
            yanchor='bottom',
            xanchor='right',
            x=1,
            y=1.02,
            font=dict(
                family="Courier New",
                size=25,
                color="black"
            )),
            font = dict(
                family = " Courier New ",
                size = 15),
            autosize = False,
            width = 1000,
            height = 1000
            )


        plot_figure = go.Figure(data = data, layout = layout)
        plot_figure.update_layout(scene=dict(xaxis_showspikes=False, yaxis_showspikes=False, zaxis_showspikes=False))
        st.plotly_chart(plot_figure)

def horizontal_bar(word, similarity, input_word=''):
    '''Build and print the horizontal bar plot for each word searched by the user.
    
    Args:
      word: vector of similar words calculated.
      similarity: vector of similarities, according to the vector of words.
      input_word: word subject of the plot.
    '''
    
    similarity = [round(elem, 2) for elem in similarity]
    
    reduced_words = []
    for w in word:
        if len(w) > 15:
            reduced_words.append(w[0:14]+'...')
        else:
            reduced_words.append(w)
    
    data = go.Bar(
            x= similarity,
            y= reduced_words,
            customdata = word,
            hovertemplate =
              "<b>Word</b>: %{customdata}<br>"+
              "Similarity: %{x:.2f}<extra></extra>",
            orientation='h',
            text = similarity,
            marker_color= 4,
            textposition='auto')

    layout = go.Layout(
            font = dict(size=20),
            xaxis = dict(showticklabels=False, automargin=True),
            yaxis = dict(showticklabels=True, automargin=True,autorange="reversed"),
            #margin = dict(t=20, b= 20, r=10),
            title = 'Words similar to {}'.format(input_word)
            )

    plot_figure = go.Figure(data = data, layout = layout)
    st.plotly_chart(plot_figure)

def display_scatterplot_2D(model, user_input=None, words=None, label=None, color_map=None, annotation='On', dim_red = 'TSNE', perplexity = 5, learning_rate = 0, iteration = 0, topn=0, sample=10):
    if words == None:
        if sample > 0:
            words = np.random.choice(list(model.wv.vocab.keys()), sample)
        else:
            words = [word for word in model.wv.vocab]
    
    word_vectors = np.array([model.wv[w] for w in words])
    
    if dim_red == 'PCA':
        two_dim = PCA(random_state=0).fit_transform(word_vectors)[:,:2]
    else:
        two_dim = TSNE(random_state=0, perplexity = perplexity, learning_rate = learning_rate, n_iter = iteration).fit_transform(word_vectors)[:,:2]


    data = []
    count = 0
    for i in range (len(user_input)):

                trace = go.Scatter(
                    x = two_dim[count:count+topn,0], 
                    y = two_dim[count:count+topn,1],  
                    text = words[count:count+topn] if annotation == 'On' else '',
                    customdata = words,
                    hovertemplate =
                      "<b>Word</b>: %{customdata}",
                    name = user_input[i],
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 15,
                        'opacity': 0.8,
                        'color': 2
                    }
                )

                data.append(trace)
                count = count+topn

    trace_input = go.Scatter(
                    x = two_dim[count:,0], 
                    y = two_dim[count:,1],  
                    text = words[count:],
                    name = 'input words',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 25,
                        'opacity': 1,
                        'color': 'black'
                    }
                    )

    data.append(trace_input)

# Configure the layout.
    layout = go.Layout(
        margin = {'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        hoverlabel=dict(
            bgcolor="white", 
            font_size=20, 
            font_family="Courier New"),
        legend=dict(
        orientation='h',
        yanchor='bottom',
        xanchor='right',
        x=1,
        y=1.02,
        font=dict(
            family="Courier New",
            size=25,
            color="black"
        )),
        font = dict(
            family = " Courier New ",
            size = 15),
        autosize = False,
        width = 1000,
        height = 1000
        )


    plot_figure = go.Figure(data = data, layout = layout)
    st.plotly_chart(plot_figure)

def split_list(items_list, n):
    '''Divide a list into sublists of size n.
    
    Args:
      items_list: original list.
      n: number of sublists to be created.
    '''

    k, m = divmod(len(items_list), n)
    return (items_list[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
    
def set_page_layout():
    '''Define some configs of the Streamlit App page, only front-end settings.'''

    st.set_page_config(
        page_title="WE4LKD | Word2Vec",
        page_icon="🖥️",
        layout="wide",
        initial_sidebar_state="expanded",
     )
    
    hide_streamlit_style = """
            <style>           
            footer {
                visibility: hidden;
            }
            
            footer:after {
                content:'Developed by Matheus Vargas Volpon Berto.'; 
                visibility: visible;
                display: block;
                position: relative;
                padding: 5px;
                top: 2px;
                color: black;
            }
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def plot_data_config(user_input, model):
    '''Calculates the variables used for the scatter plot (2D or 3D) functions.
    
    Args:
      w2v: Word2Vec model.
      restricted_word_set: list of words of the domain.
      domain: boolean that informs how to remove the words (remove words that belong or not to the domain).
    '''

    result_word = []
    sim_words = []
    
    for word in user_input:
        sim_words = model.wv.most_similar(word, topn = top_n)
        sim_words = append_list(sim_words, word)
        result_word.extend(sim_words)
    
    similar_word = [word[0] for word in result_word]
    similarity = [word[1] for word in result_word]
    try:
        similar_word.extend(user_input)
    except TypeError:
        pass
    labels = [word[2] for word in result_word]
    label_dict = dict([(y,x+1) for x,y in enumerate(set(labels))])
    color_map = [label_dict[x] for x in labels]
    
    return result_word, sim_words, similar_word, similarity, labels, label_dict, color_map

def deep_search(words_session_state, new_word):
    '''Itertive serach for terms, adds the new term to the session_state variable and ikncrements the execution counter.
    
    Args:
      words_session_state: words saved in the session_state variable 'user_input'.
      new_word: new term to be added to the session_state variable.
    '''

    st.session_state['execution_counter'] += 1
    aux = words_session_state
    aux.append(new_word)
    aux = list(dict.fromkeys(aux))
    st.session_state['user_input'] = aux

def clear_session_state():
    '''Delete all variables saved in the session_state. This function is used when the user wants to start a new search.'''

    for key in st.session_state.keys():
        del st.session_state[key]
        
# GLOBAL VARIABLES:
specific_domain = []
base_compounds = get_target_compounds()
    
# MAIN PROGRAM:
if __name__ == '__main__':
    download_w2v_models()
    vocabulary_restricted = False
    
    hide_streamlit_style = """
            <style>           
            footer {
                visibility: hidden;
            }
            
            footer:after {
                content:'Developed by Matheus Vargas Volpon Berto.'; 
                visibility: visible;
                display: block;
                position: relative;
                padding: 5px;
                top: 2px;
                color: black;
            }
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    if 'widget' not in st.session_state:
        st.session_state['widget'] = 0

    if 'execution_counter' not in st.session_state:
        st.session_state['execution_counter'] = 0
        
    if 'user_input' not in st.session_state:
        st.session_state['user_input'] = []

    # sidebar widgets form:
    with st.sidebar.form('sidebar_form'):
        st.header('Models exploration settings')

        loaded_model = st.selectbox(
         'Choose one of the preloaded models:',
         ('19: 1921 - 2022',
          '18: 1921 - 2018',
          '17: 1921 - 2014',
          '16: 1921 - 2013',
          '15: 1921 - 2011',
          '14: 1921 - 2009',
          '13: 1921 - 2001',
          '12: 1921 - 1999',
          '11: 1921 - 1998',
          '10: 1921 - 1995',
          '09: 1921 - 1983',
          '08: 1921 - 1982',
          '07: 1921 - 1977',
          '06: 1921 - 1976',
          '05: 1921 - 1974',
          '04: 1921 - 1971',
          '03: 1921 - 1969',
          '02: 1921 - 1967',
          '01: 1921 - 1963'))

        restrict_domain = st.selectbox("Restrict vocabulary domain:",
        ('general', 'NCI cancer drugs', 'FDA drugs'))

        if restrict_domain == 'general':
            st.markdown('Filter vocabulary by entities:')
            cellular = st.checkbox('Cellular')
            dna_rna = st.checkbox('DNA/RNA')
            drugs_chemicals = st.checkbox('Drugs/Chemicals')
            proteins = st.checkbox('Proteins')

            if (drugs_chemicals == False and dna_rna == False and proteins == False and cellular == False):
                common_words_number = st.selectbox('Select the number of the most common words to remove from the view',
                ('None', '5000', '10000', '15000', '20000'))       

        dim_red = st.selectbox(
         'Select the dimensionality reduction method',
         ('TSNE','PCA'))

        dimension = st.selectbox(
             "Select the display dimension",
             ('2D', '3D'))

        user_input = st.text_input("Enter the words to be searched. For more than one word, separate them with a comma (,)", value='', key='words_search')

        top_n = st.slider('Select the neighborhood size',
            5, 20, (5), 5)

        annotation = st.radio(
             "Dot plot labels",
             ('On', 'Off'))  
            
        submitted = st.form_submit_button('Apply settings')
        if submitted or st.session_state['execution_counter'] != 0:
            if loaded_model == '01: 1921 - 1963':
                model = pickle.load(open('./models_streamlit_app/model_1921_1963.model', 'rb'))
            elif loaded_model == '02: 1921 - 1967':
                model = pickle.load(open('./models_streamlit_app/model_1921_1967.model', 'rb'))
            elif loaded_model == '03: 1921 - 1969':
                model = pickle.load(open('./models_streamlit_app/model_1921_1969.model', 'rb'))
            elif loaded_model == '04: 1921 - 1971':
                model = pickle.load(open('./models_streamlit_app/model_1921_1971.model', 'rb'))
            elif loaded_model == '05: 1921 - 1974':
                model = pickle.load(open('./models_streamlit_app/model_1921_1974.model', 'rb'))
            elif loaded_model == '06: 1921 - 1976':
                model = pickle.load(open('./models_streamlit_app/model_1921_1976.model', 'rb'))
            elif loaded_model == '07: 1921 - 1977':
                model = pickle.load(open('./models_streamlit_app/model_1921_1977.model', 'rb'))
            elif loaded_model == '08: 1921 - 1982':
                model = pickle.load(open('./models_streamlit_app/model_1921_1982.model', 'rb'))
            elif loaded_model == '09: 1921 - 1983':
                model = pickle.load(open('./models_streamlit_app/model_1921_1983.model', 'rb'))
            elif loaded_model == '10: 1921 - 1995':
                model = pickle.load(open('./models_streamlit_app/model_1921_1995.model', 'rb'))
            elif loaded_model == '11: 1921 - 1998':
                model = pickle.load(open('./models_streamlit_app/model_1921_1998.model', 'rb'))
            elif loaded_model == '12: 1921 - 1999':
                model = pickle.load(open('./models_streamlit_app/model_1921_1999.model', 'rb'))
            elif loaded_model == '13: 1921 - 2001':
                model = pickle.load(open('./models_streamlit_app/model_1921_2001.model', 'rb'))
            elif loaded_model == '14: 1921 - 2009':
                model = pickle.load(open('./models_streamlit_app/model_1921_2009.model', 'rb'))
            elif loaded_model == '15: 1921 - 2011':
                model = pickle.load(open('./models_streamlit_app/model_1921_2011.model', 'rb'))
            elif loaded_model == '16: 1921 - 2013':
                model = pickle.load(open('./models_streamlit_app/model_1921_2013.model', 'rb'))
            elif loaded_model == '17: 1921 - 2014':
                model = pickle.load(open('./models_streamlit_app/model_1921_2014.model', 'rb'))
            elif loaded_model == '18: 1921 - 2018':
                model = pickle.load(open('./models_streamlit_app/model_1921_2018.model', 'rb'))
            elif loaded_model == '19: 1921 - 2022':
                model = pickle.load(open('./models_streamlit_app/model_1921_2022.model', 'rb'))

            #model.init_sims()

            if restrict_domain != 'general':
                if restrict_domain == 'NCI cancer drugs':
                    domains_table = read_domain_table()
                    specific_domain = domains_table['name'].tolist()

                elif restrict_domain == 'FDA drugs':
                    specific_domain = read_fda_drugs_file()

                wv_restrict_w2v(model, set(specific_domain), True)
                vocabulary_restricted = True

            else:
                if (drugs_chemicals or dna_rna or proteins or cellular):
                    list_drugs_chemicals, list_dna_rna, list_proteins, list_cellular = create_entities_lists()
                    entities_list = [list_drugs_chemicals, list_dna_rna, list_proteins, list_cellular]
                    selected_entities = [drugs_chemicals, dna_rna, proteins, cellular]

                    specific_domain = []
                    for list_name, selected in zip(entities_list, selected_entities):
                        if (selected == True):
                            specific_domain.extend(list_name)

                    wv_restrict_w2v(model, set(specific_domain), True)

                else:
                    if common_words_number != 'None':
                        common_words = get_most_common(int(common_words_number))
                        wv_restrict_w2v(model, set(common_words))

                vocabulary_restricted = True   

            if dim_red == 'TSNE':
                perplexity = 5
                learning_rate = 0.001
                iteration = 250

            else:
                perplexity = 5
                learning_rate = 0.001
                iteration = 0    

        else:
            #model = pickle.load(open('./models_streamlit_app/model_1921_2022.model', 'rb'))
            model = Word2Vec.load('./models_streamlit_app/model_1921_2022.model', mmap='r')
            dim_red = 'TSNE'
            perplexity = 5
            learning_rate = 0.001
            iteration = 250
            top_n = 5
            annotation = 'On'

    reset_search = st.sidebar.button("Reset search", key='clear_session_button', on_click=clear_session_state, help='Delete all previous search record and start a new one')
    if reset_search:
        st.session_state['words_search'] = ''
        user_input = ''

    st.sidebar.header('GitHub Repository')
    st.sidebar.markdown("[![Foo](https://cdn-icons-png.flaticon.com/32/25/25231.png)](https://github.com/matheusvvb-19/WE4LKD-leukemia_w2v)")

    header_container = st.container()
    with header_container:
        st.title('Embedding Viewer')
        st.header('Word Embedding Visualization Based on Cosine Similarity')
        with st.expander('How to use this app'):
            st.markdown('**Sidebar**')
            st.markdown('Define the exploration settings in the sidebar. First, choose one of the preloaded Word2Vec models. Then choose whether you want to restrict the terms in the model to a specific domain. If there is no domain restriction, you can choose how many common English words you want to remove from the visualization; removing these words can improve your investigation since they are often outside the medical context. However, be careful about removing common words or the domain restriction, they can drastically reduce the vocabulary of the model.')    
            st.markdown('Then select the dimensionality reduction method. If you do not know what this means, leave the default value "TSNE". Below this option, set the number of dimensions to be plotted (2D or 3D). You can also search for specific words by typing them into the text field. For more than one word, separate it with commas. Be careful, if you decide to remove too many common words or restrct the vocabulary to a specific domain, the word you are looking for may no longer be present in the model.')
            st.markdown('Finally, you can increase or decrease the neighborhood of the searched terms using the slider and enable or disable the labels of each point on the plot. After defining all the search parameters, click on "Apply settings".')
            st.markdown('If you want to restart your exploration from another input word or change radically change the search parameters, click on the "Reset search" button.')

            st.markdown('**Main window**')
            st.markdown('_Hint: To see this window content better, you can minimize the sidebar._')
            st.markdown('The first dot plot shows the words similar to each input and their distribution in vectorial space. You can move the plot, crop a specific area or hide some points by clicking on the words in the right caption. Then, the table below the dot plot shows the cosine similarity and the rank (ordinal position) from the base compounds of this project - header of the table - and the words you chose to explore. Below the table, the app generates bar plots with similar words for each term you explored. Also, you can search for words returned by your previous search, clicking on the button with the term. This way, you can explore the neighborhood of your original input and find out the context of them.')

    plot_container = st.empty()
    if user_input == '':
        similar_word = None
        labels = None
        color_map = None

        with plot_container:
            if dimension == '2D':
                display_scatterplot_2D(model, user_input, similar_word, labels, color_map, annotation, dim_red, perplexity, learning_rate, iteration, top_n)
            else:
                display_scatterplot_3D(model, user_input, similar_word, labels, color_map, annotation, dim_red, perplexity, learning_rate, iteration, top_n)

    else:
        # se o usuário digitar algo no campo de entrada de texto, cria-se a lista de palavras de busca:
        user_input = [x.strip().lower() for x in user_input.split(',') if len(x) >= 2]

        # se essa execução for a primeira, é necessário buscar pelas palavras digitadas no vocabulário do modelo:
        if st.session_state['execution_counter'] == 0:
            matched_synonyms = read_matched_synonyms()      # pandas DataFrame com os termos que foram efetivamente substituídos por sinônimos durante o pré-processamento

            replaced_words = []

            # para cada palavra de user_input, verificar se ela se encontra no DataFrame recém-lido:
            for w in user_input:
                row = matched_synonyms.loc[matched_synonyms['synonym'] == w]

                # palavra está presente no DataFrame - ou seja, foi substituída por um sinônimo durante a limpeza do texto:
                if len(row.index) != 0:
                    # se o synonym_title estiver presente no vocabulário do modelo, significa que a palavra foi efetivamente substituída e ESTAVA PRESENTE no modelo em questão:
                    if row['synonym_title'].values[0] in set(model.wv.vocab):
                        st.warning("'{}' was replaced by '{}' during text preprocessing.".format(w, row['synonym_title'].values[0]))
                        replaced_words.append((w, row['synonym_title'].values[0]))

            # percorrendo a lista de tuplas <termo_substituido, termo>, substutindo pelos sinônimos pré-processados:
            for pair in replaced_words:
                user_input = list(map(lambda x: x.replace(pair[0], pair[1]), user_input))

            matches = []
            words_to_remove = []

            # para cada uma das palavras digitadas, busca no vocbulário palavras que a contenham como substring:
            for w in user_input:
                found = list(filter(lambda x: w in x, model.wv.vocab))

                # se houver ao menos uma embedding que tenha como substring o termo digitado pelo usuário:
                if len(found) > 0:
                    # se o termo completo não for encontrado, ele terá que posteriormente ser removido de user_input - pois ele não existe no vocabulário
                    if w not in found:
                        matches.extend(found)           # e as opções de palavras semelhantes são salvas na lista matches
                        words_to_remove.append(w)

                # se nenhuma embedding conter como substring a palavra digitada pelo usuário, ela também é adicionada à lista de futuras palavras a serem eliminadas e um aviso ao usuário é feito:
                else:
                    words_to_remove.append(w)
                    st.warning("'{}' is out of the model's vocabulary. Try again using another keyword.".format(w))

            # removendo de user_input as palavras que não foram encontradas (por inteiro) no vocabulário, mas apresentavam variações:
            user_input = [x for x in user_input if x not in words_to_remove]
            st.session_state['user_input'] = user_input                 # atualizando o valor da variável no session_state

        # se essa não for a primeira execução, apenas recupera as palavras previamente buscadas salvas em session_state:
        else:
            user_input = st.session_state['user_input']

        if st.session_state['execution_counter'] == 0 and len(matches) > 0:
            st.markdown('The following word embeddings have the sub-word you typed. Please, select one to explore.')
            for w in matches:
                st.button(w, on_click=deep_search, args=(st.session_state['user_input'], w), key='{}@{}'.format(w, random()))

        else:
            if len(user_input) > 0:
                if vocabulary_restricted:
                    words_to_remove = []
                    for w in user_input:
                        if w not in model.wv.vocab:
                            words_to_remove.append(w)
                            st.warning("'{}' is out of the model's vocabulary. Try again using another keyword.".format(w))

                    user_input = [x for x in user_input if x not in words_to_remove]
                    st.session_state['user_input'] = user_input                 # atualizando o valor da variável no session_state

                if len(user_input) > 0:
                    result_word, sim_words, similar_word, similarity, labels, label_dict, color_map = plot_data_config(user_input, model)   
                    with plot_container:
                        if dimension == '2D':
                            display_scatterplot_2D(model, user_input, similar_word, labels, color_map, annotation, dim_red, perplexity, learning_rate, iteration, top_n)
                        else:
                            display_scatterplot_3D(model, user_input, similar_word, labels, color_map, annotation, dim_red, perplexity, learning_rate, iteration, top_n)

                    seed(st.session_state['widget'])

                    table_section = st.container()
                    with table_section:
                        table_title_div = st.container()
                        with table_title_div:
                            st.header('Similarity between the search terms and the base compounds.')
                            st.markdown("Size of model's vocabulary: {}".format(len(model.wv.vocab)))
                            table_cells_div = st.empty()
                            with table_cells_div:
                                similarities_table_streamlit(user_input, model)

                    subplots_section = st.container()
                    with subplots_section:
                        subplots_title_div = subplots_section.container()
                        with subplots_title_div:
                            st.header('{} most similar words for each input.'.format(top_n))

                        number_terms = len(user_input)
                        previous_number_terms = len(user_input)
                        count=0
                        i=0
                        options_list = list(split_list(similar_word[:-number_terms], number_terms))

                        if number_terms % 2 == 0:
                            number_containers = int(number_terms/2)
                        else:
                            number_containers = int(number_terms/2) + 1

                        previous_number_containers = number_containers        
                        subplots_plots_div = subplots_section.container()
                        with subplots_plots_div:
                            for j in range(number_containers):
                                subplots_plots_div_row = subplots_plots_div.container()
                                col1, col2 = subplots_plots_div_row.columns(2)
                                col1_plot = col1.empty()
                                col2_plot = col2.empty()

                                with col1_plot:
                                    horizontal_bar(similar_word[count:count+top_n], similarity[count:count+top_n], user_input[i])

                                i = i + 1
                                count = count + top_n
                                try:
                                    with col2_plot:
                                        horizontal_bar(similar_word[count:count+top_n], similarity[count:count+top_n], user_input[i])
                                except:
                                    pass

                                count = count + top_n
                                i = i + 1     

                    form_section = st.container()
                    with form_section:
                        form_title_div = st.container()
                        with form_title_div:
                            st.write('You can go deep and search for one of the terms returned by your search. Click on the word that you want to add to the exploration - choose only one:')
                            st.write('The words are in descending order of similarity.')

                        if (st.session_state['execution_counter'] > 0):
                            last_word_search = len(st.session_state['user_input']) - 1
                            form_selection_div = st.container()
                            with form_selection_div:
                                st.markdown('**{}**'.format(user_input[last_word_search]))
                                for w in options_list[last_word_search]:
                                    st.session_state['widget'] += 1
                                    st.button(w, on_click=deep_search, args=(st.session_state['user_input'], w), key='{}@{}'.format(w, random()))

                        else:
                            form_selection_div = st.empty()
                            with form_selection_div:
                                cols = form_selection_div.columns(number_terms)
                                for k, col in enumerate(cols):
                                    col.markdown('**{}**'.format(user_input[k]))
                                    for w in options_list[k]:
                                        st.session_state['widget'] += 1
                                        col.button(w, on_click=deep_search, args=(st.session_state['user_input'], w), key='{}@{}'.format(w, random()))
