# based on https://towardsdatascience.com/visualizing-word-embedding-with-pca-and-t-sne-961a692509f5

# IMPORTS:
import plotly, pickle, csv, re, string
import plotly.graph_objs as go
import streamlit as st
import pandas as pd
import numpy as np
from get_n_common_words_english import get_most_common
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from random import random, seed

# GLOBAL VARIABLES:
specific_domain = []
base_compounds = ['cytarabine', 'daunorubicin', 'azacitidine', 'gemtuzumab-ozogamicin', 'midostaurin', 'vyxeos', 'ivosidenib', 'venetoclax', 'enasidenib', 'gilteritinib', 'glasdegib']

# FUNCTIONS:
def process_entity_list(entity_list):
    for index, s in enumerate(entity_list):
        entity_list[index] = re.sub('<[^>]+>', '', s)
        entity_list[index] = re.sub('\\s+', ' ', s)
        entity_list[index] = re.sub('([--:\w?@%&+~#=]*\.[a-z]{2,4}\/{0,2})((?:[?&](?:\w+)=(?:\w+))+|[--:\w?@%&+~#=]+)?', '', s)
        entity_list[index] = re.sub('\d+\W+\d+', '', s)
        entity_list[index] = s.replace('/', ' ')

    entity_list = [x.split(' ') for x in entity_list]
    entity_list = [val for sublist in entity_list for val in sublist]
    
    return entity_list

def create_entities_lists():
    url = 'https://drive.google.com/file/d/1Q-lA9xtZztUETz5zJrbdN0Fdpg96u2y7/view?usp=sharing'
    path = 'https://drive.google.com/uc?export=download&id=' + url.split('/')[-2]
    entities_table = pd.read_csv(path)
    
    list_diseases = entities_table[entities_table['entities'].str.contains('oncological|disease_syndrome_disorder|symptom|treatment|cancer')]['word'].to_list()
    list_drugs_chemicals = entities_table[entities_table['entities'].str.contains('substance|drug_ingredient|drug_brandname|drugchem|drug')]['word'].to_list()
    list_dna_rna = entities_table[entities_table['entities'].str.contains('dna|gene_or_gene_product|rna')]['word'].to_list()
    list_proteins = entities_table[entities_table['entities'].str.contains('protein|amino_acid')]['word'].to_list()
    list_cellular = entities_table[entities_table['entities'].str.contains('cell_type|cell_line|cell|cellular_component|tissue|multi-tissue_structure')]['word'].to_list()
    
    list_diseases = process_entity_list(list_diseases)
    list_drugs_chemicals = process_entity_list(list_drugs_chemicals)
    list_dna_rna = process_entity_list(list_dna_rna)
    list_proteins = process_entity_list(list_proteins)
    list_cellular = process_entity_list(list_cellular)
    
    return list_diseases, list_drugs_chemicals, list_dna_rna, list_proteins, list_cellular

@st.cache
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
                s = re.sub('\d+\W+\d+', '', s)
                s = s.lower()
                s = s.translate(str.maketrans('', '', string.punctuation.replace('-', '')))
                words_list.append(s)

    words_list.pop(0)
    words_list = list(dict.fromkeys(words_list))
    
    return words_list
    
@st.cache
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

def display_scatterplot_3D(model, user_input=None, words=None, label=None, color_map=None, annotation='On',  dim_red = 'TSNE', perplexity = 0, learning_rate = 0, iteration = 0, topn=0, sample=10):
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
            x=1,
            y=0.5,
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

def display_scatterplot_2D(model, user_input=None, words=None, label=None, color_map=None, annotation='On', dim_red = 'TSNE', perplexity = 0, learning_rate = 0, iteration = 0, topn=0, sample=10):
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
        x=1,
        y=0.5,
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
    '''Define some configs of the Streamlit App page, most of them only front-end settings.'''

    st.set_page_config(
        page_title="Embedding Viewer",
        page_icon="🖥️",
        layout="wide",
        initial_sidebar_state="expanded",
     )
    
    hide_streamlit_style = """
            <style>
            .css-zbg2rx, .css-sygy1k, .css-18e3th9 {
                padding-top: 2rem !important;
            }
            
            footer {
                visibility: hidden;
            }
            
            footer:after {
                content:'Developed by Matheus Volpon, WE4LKD Team.'; 
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
    '''Calculates the variables used for the scatter plot (2D or 3D) funcitons.
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
    '''Delete all variables saved in the session_state. This function is used when the user wnats to strat a new search from zero.'''

    for key in st.session_state.keys():
        del st.session_state[key]

@st.cache(allow_output_mutation=True)
def load_model(model_name, loaded=False):
    if loaded:
        model = pickle.load(model_name)
    else:
        if model_name == '1: 1900 - 1967':
            model = pickle.load(open('./models_streamlit_app/model_1900_1967.model', 'rb'))
        elif model_name == '2: 1900 - 1977':
            model = pickle.load(open('./models_streamlit_app/model_1900_1977.model', 'rb'))
        elif model_name == '3: 1900 - 1999':
            model = pickle.load(open('./models_streamlit_app/model_1900_1999.model', 'rb'))
        elif model_name == '4: 1900 - 2001':
            model = pickle.load(open('./models_streamlit_app/model_1900_2001.model', 'rb'))
        elif model_name == '5: 1900 - 2009':
            model = pickle.load(open('./models_streamlit_app/model_1900_2009.model', 'rb'))
        elif model_name == '6: 1900 - 2011':
            model = pickle.load(open('./models_streamlit_app/model_1900_2011.model', 'rb'))
        elif model_name == '7: 1900 - 2013':
            model = pickle.load(open('./models_streamlit_app/model_1900_2013.model', 'rb'))
        elif model_name == '8: 1900 - 2014':
            model = pickle.load(open('./models_streamlit_app/model_1900_2014.model', 'rb'))
        elif model_name == '9: 1900 - 2016':
            model = pickle.load(open('./models_streamlit_app/model_1900_2016.model', 'rb'))
        elif model_name == '10: 1900 - 2021':
            model = pickle.load(open('./models_streamlit_app/model_1900_2021.model', 'rb'))
        
    model.init_sims()
    return model
    
# MAIN PROGRAM:
if __name__ == '__main__':
    set_page_layout()

    # sidebar widgets:
    st.sidebar.header('Models exploration settings')
    uploaded_file = st.sidebar.file_uploader("Upload a new model:")
    if uploaded_file is not None:
        model = pickle.load(uploaded_file)
        model.init_sims()

    loaded_model = st.sidebar.selectbox(
     'Or choose one of the preloaded models:',
     ('9: 1900 - 2016', '8: 1900 - 2014', '7: 1900 - 2013', '6: 1900 - 2011', '5: 1900 - 2009', '4: 1900 - 2001', '3: 1900 - 1999', '2: 1900 - 1977', '1: 1900 - 1967'))

    if uploaded_file is None:
        if loaded_model == '1: 1900 - 1967':
          model = pickle.load(open('./models_streamlit_app/model_1900_1967.model', 'rb'))
        elif loaded_model == '2: 1900 - 1977':
          model = pickle.load(open('./models_streamlit_app/model_1900_1977.model', 'rb'))
        elif loaded_model == '3: 1900 - 1999':
          model = pickle.load(open('./models_streamlit_app/model_1900_1999.model', 'rb'))
        elif loaded_model == '4: 1900 - 2001':
          model = pickle.load(open('./models_streamlit_app/model_1900_2001.model', 'rb'))
        elif loaded_model == '5: 1900 - 2009':
          model = pickle.load(open('./models_streamlit_app/model_1900_2009.model', 'rb'))
        elif loaded_model == '6: 1900 - 2011':
          model = pickle.load(open('./models_streamlit_app/model_1900_2011.model', 'rb'))
        elif loaded_model == '7: 1900 - 2013':
          model = pickle.load(open('./models_streamlit_app/model_1900_2013.model', 'rb'))
        elif loaded_model == '8: 1900 - 2014':
          model = pickle.load(open('./models_streamlit_app/model_1900_2014.model', 'rb'))
        elif loaded_model == '9: 1900 - 2016':
          model = pickle.load(open('./models_streamlit_app/model_1900_2016.model', 'rb'))
        elif loaded_model == '10: 1900 - 2021':
          model = pickle.load(open('./models_streamlit_app/model_1900_2021.model', 'rb'))
        model.init_sims()
        
    st.markdown('tamanho do vocabulário: {}'.format(len(model.wv.vocab)))

    restrict_domain = st.sidebar.selectbox("Restrict vocabulary domain:",
    ('general', 'NCI cancer drugs', 'FDA drugs'))
    if restrict_domain != 'general':
        if restrict_domain == 'NCI cancer drugs':
            domains_table = read_domain_table()
            specific_domain = domains_table['name'].tolist()
            wv_restrict_w2v(model, set(specific_domain), True)
            
        elif restrict_domain == 'FDA drugs':
            specific_domain = read_fda_drugs_file()
            wv_restrict_w2v(model, set(specific_domain), True)
    else:
        st.sidebar.markdown('Filter vocabulary by entities:')
        diseases = st.sidebar.checkbox('Diseases')
        drugs_chemicals = st.sidebar.checkbox('Drugs/Chemicals')
        dna_rna = st.sidebar.checkbox('DNA/RNA')
        proteins = st.sidebar.checkbox('Proteins')
        cellular = st.sidebar.checkbox('Cellular')
        
        if (diseases or drugs_chemicals or dna_rna or proteins or cellular):
            list_diseases, list_drugs_chemicals, list_dna_rna, list_proteins, list_cellular = create_entities_lists()
            selected_entities = [diseases, drugs_chemicals, dna_rna, proteins, cellular]
            
            st.markdown('diseases: {}'.format(len(list_diseases)))
            st.markdown('drugs/chemicals: {}'.format(len(list_drugs_chemicals)))
            st.markdown('dna/rna: {}'.format(len(list_dna_rna)))
            st.markdown('proteins: {}'.format(len(list_proteins)))
            st.markdown('cellular: {}'.format(len(list_cellular)))
            
            specific_domain = []
            for list_name, selected in zip(list_diseases, selected_entities):
                if (selected == True):
                    specific_domain.extend(list_name)
                    
            st.markdown(len(specific_domain))
            wv_restrict_w2v(model, specific_domain, True)
            
        else:
            common_words_number = st.sidebar.selectbox('Select the number of the most common words to remove from the view',
            ('None', '5000', '10000', '15000', '20000'))
            if common_words_number != 'None':
                common_words = get_most_common(int(common_words_number))
                wv_restrict_w2v(model, set(common_words))

    dim_red = st.sidebar.selectbox(
     'Select the dimensionality reduction method',
     ('TSNE','PCA'))

    dimension = st.sidebar.selectbox(
         "Select the display dimension",
         ('2D', '3D'))

    user_input = st.sidebar.text_input("Enter the words to be searched. For more than one word, separate them with a comma (,)", value='', key='words_search')

    top_n = st.sidebar.slider('Select the neighborhood size',
        5, 20, (5), 5)

    annotation = st.sidebar.radio(
         "Enable or disable dot plot labels",
         ('On', 'Off'))  

    if dim_red == 'TSNE':
        perplexity = 0
        learning_rate = 0
        iteration = 250

    else:
        perplexity = 0
        learning_rate = 0
        iteration = 0    

    if user_input == '':
        similar_word = None
        labels = None
        color_map = None

    else:
        if 'user_input' not in st.session_state:
            user_input = [x.strip().lower() for x in user_input.split(',') if len(x) > 1]
            st.session_state['user_input'] = user_input

        else:
            user_input = st.session_state['user_input']

        for w in user_input:
            if w not in model.wv.vocab:
                user_input.remove(w)
                st.error("The word {} is not present in model's vocabulary and will be ignored.".format(w))

        result_word, sim_words, similar_word, similarity, labels, label_dict, color_map = plot_data_config(user_input, model)

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
            st.markdown('First, upload your word embedding model file with ".model" extension or choose one of the preloaded Word2Vec models. Then choose whether you want to restrict the terms in the model to a specific domain. If there is no domain restriction, you can choose how many common English words you want to remove from the visualization; removing these words can improve your investigation since they are often outside the medical context. However, be careful about removing common words or the domain restriction, they can drastically reduce the vocabulary of the model.')    
            st.markdown('Then select the dimensionality reduction method. If you do not know what this means, leave the default value "TSNE". Below this option, set the number of dimensions to be plotted (2D or 3D). You can also search for specific words by typing them into the text field. For more than one word, separate it with commas. Be careful, if you decide to remove too many common words, the word you are looking for may no longer be present in the model.')
            st.markdown('Finally, you can increase or decrease the neighborhood of the searched terms using the slider and enable or disable the labels of each point on the plot. If you want to restart your exploration, click on the "Reset search" button and type the new word(s) in the text field.')

            st.markdown('**Main window**')
            st.markdown('_Hint: To see this window content better, you can minimize the sidebar._')
            st.markdown('The first dot plot shows the words similar to each input and their distribution in vectorial space. You can move the plot, crop a specific area or hide some points by clicking on the words in the right caption. Then, the table below the dot plot shows the cosine similarity and the rank (ordinal position) from the base compounds of this project - header of the table - and the words you chose to explore. Below the table, the app generates bar plots with similar words for each term you explored. Also, you can search for words returned by your previous search, clicking on the button with the term. This way, you can explore the neighborhood of your original input and find out the context of them.')

    plot_container = st.empty()
    with plot_container:
        if dimension == '2D':
            display_scatterplot_2D(model, user_input, similar_word, labels, color_map, annotation, dim_red, perplexity, learning_rate, iteration, top_n)
        else:
            display_scatterplot_3D(model, user_input, similar_word, labels, color_map, annotation, dim_red, perplexity, learning_rate, iteration, top_n)

    if user_input != '':
        if 'widget' not in st.session_state:
            st.session_state['widget'] = 0

        if 'execution_counter' not in st.session_state:
            st.session_state['execution_counter'] = 0

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
