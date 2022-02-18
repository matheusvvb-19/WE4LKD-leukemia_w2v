#based on https://towardsdatascience.com/visualizing-word-embedding-with-pca-and-t-sne-961a692509f5

import plotly, pickle, csv, re, string
import plotly.graph_objs as go
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from get_n_common_words_english import get_most_common
from gensim.models import Word2Vec, KeyedVectors
from clean_text import replace_synonyms
import plotly.graph_objects as go
import plotly.figure_factory as ff

specific_domain = []
base_compounds = ['cytarabine', 'daunorubicin', 'gemtuzumab ozogamicin', 'midostaurin', 'cpx-351', 'ivosidenib', 'venetoclax', 'enasidenib', 'gilteritinib', 'glasdegib']

# domains table:
domains_table = pd.read_csv('https://docs.google.com/spreadsheets/d/' + 
                   '1SgYG4gZuL3grEHHAZt49dAUw_jFAc4LADajFeGAf2-w' +
                   '/export?gid=0&format=csv',
                  )

def similarities_table_streamlit(words_list, model):  
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
    
    word_vectors = np.array([model[w] for w in words])
    
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
            hovertemplate =
              '<b>Word:</b>: $%{word}'+
              '<b>Similarity:</b>: $%{x:.2f}',
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
    
    word_vectors = np.array([model[w] for w in words])
    
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
    k, m = divmod(len(items_list), n)
    return (items_list[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
    
def set_page_layout():
    st.set_page_config(
        page_title="Embedding Viewer",
        page_icon="🖥️",
        layout="wide",
        initial_sidebar_state="expanded"
     )
    
    hide_streamlit_style = """
            <style>
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
            }
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def plot_data_config(user_input, model):
    result_word = []
    sim_words = []
    
    for words in user_input:
        try:
            sim_words = model.wv.most_similar(words, topn = top_n)
            sim_words = append_list(sim_words, words)
            result_word.extend(sim_words)
        except KeyError:
            st.error("The word {} is not present in model's vocabulary.".format(words))
        except TypeError:
            pass      
    
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
    
set_page_layout()

uploaded_file = st.sidebar.file_uploader("Upload a new model:")
if uploaded_file is not None:
    model = pickle.load(uploaded_file)
    model.init_sims()

loaded_model = st.sidebar.selectbox(
 'Or choose one of the preloaded models:',
 ('10: 1900 - 2021', '9: 1900 - 2016', '8: 1900 - 2014', '7: 1900 - 2013', '6: 1900 - 2011', '5: 1900 - 2009', '4: 1900 - 2001', '3: 1900 - 1999', '2: 1900 - 1977', '1: 1900 - 1967'))

if uploaded_file is None:
    if loaded_model == '1: 1900 - 1967':
      model = pickle.load(open('./models_streamlit_app/model_results_file_1900_1967_clean.model', 'rb'))
    elif loaded_model == '2: 1900 - 1977':
      model = pickle.load(open('./models_streamlit_app/model_results_file_1900_1977_clean.model', 'rb'))
    elif loaded_model == '3: 1900 - 1999':
      model = pickle.load(open('./models_streamlit_app/model_results_file_1900_1999_clean.model', 'rb'))
    elif loaded_model == '4: 1900 - 2001':
      model = pickle.load(open('./models_streamlit_app/model_results_file_1900_2001_clean.model', 'rb'))
    elif loaded_model == '5: 1900 - 2009':
      model = pickle.load(open('./models_streamlit_app/model_results_file_1900_2009_clean.model', 'rb'))
    elif loaded_model == '6: 1900 - 2011':
      model = pickle.load(open('./models_streamlit_app/model_results_file_1900_2011_clean.model', 'rb'))
    elif loaded_model == '7: 1900 - 2013':
      model = pickle.load(open('./models_streamlit_app/model_results_file_1900_2013_clean.model', 'rb'))
    elif loaded_model == '8: 1900 - 2014':
      model = pickle.load(open('./models_streamlit_app/model_results_file_1900_2014_clean.model', 'rb'))
    elif loaded_model == '9: 1900 - 2016':
      model = pickle.load(open('./models_streamlit_app/model_results_file_1900_2016_clean.model', 'rb'))
    elif loaded_model == '10: 1900 - 2021':
      model = pickle.load(open('./models_streamlit_app/model_results_file_1900_2021_clean.model', 'rb'))
    model.init_sims()
    
restrict_domain = st.sidebar.selectbox("Restrict vocabulary domain:",
('general', 'NCI cancer drugs', 'FDA drugs'))
if restrict_domain != 'general':
    if restrict_domain == 'NCI cancer drugs':
        specific_domain = domains_table['name'].tolist()
        wv_restrict_w2v(model, set(specific_domain), True)
    elif restrict_domain == 'FDA drugs':
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
                    s = replace_synonyms(s)
                    s = s.translate(str.maketrans('', '', string.punctuation.replace('-', '')))
                    specific_domain.append(s)

        specific_domain.pop(0)
        specific_domain = list(dict.fromkeys(specific_domain))
        wv_restrict_w2v(model, set(specific_domain), True)
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
user_input = st.sidebar.text_input("Enter the words to be searched. For more than one word, separate them with a comma (,)",'')
top_n = st.sidebar.slider('Select the neighborhood size',
    5, 30, (5), 5)
annotation = st.sidebar.radio(
     "Habilite ou desabilite os rótulos",
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
    user_input = [x.strip().lower() for x in user_input.split(',')]
    result_word, sim_words, similar_word, similarity, labels, label_dict, color_map = plot_data_config(user_input, model)
    
header_container = st.container()
with header_container:
    st.title('Word Embedding Visualization Based on Cosine Similarity')
    with st.expander('How to use this app'):
        st.markdown('First, upload your word embedding model file with ".model" extension or choose one of the preloaded Word2Vec models. Then choose whether you want to restrict the terms in the model to a specific domain. If there is no domain restriction, you can choose how many common English words you want to remove from the visualization; removing these words can improve your investigation, since they are often words outside the medical context. However, be careful about removing common words or the domain restriction, they can drastically reduce the vocabulary of the model.')    
        st.markdown('Then select the dimensionality reduction method. If you do not know what this means, leave the default value "TSNE". Below this option, set the number of dimensions to be plotted (2D or 3D).')
        st.markdown('You can also search for specific words by typing them into the field. For more than one word, separate them with commas. Be careful, if you decide to remove too many common words, the word you are looking for may no longer be present in the model.')
        st.markdown('Finally, you can increase or decrease the neighborhood of the searched terms using the slider. You can also enable or disable the labels of each point on the plot.')

plot_container = st.empty()
with plot_container:
    if dimension == '2D':
        display_scatterplot_2D(model, user_input, similar_word, labels, color_map, annotation, dim_red, perplexity, learning_rate, iteration, top_n)
    else:
        display_scatterplot_3D(model, user_input, similar_word, labels, color_map, annotation, dim_red, perplexity, learning_rate, iteration, top_n)

if user_input != '':
    original_search = user_input
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
    new_words_to_search = []
    with form_section:
        form_title_div = st.container()
        with form_title_div:
            st.write("You can go deep and search specifically with the terms returned by this search. Choose the words and click on 'Submit' button to search:")
        
        form_selection_div = st.empty()
        with form_selection_div:
            form = form_selection_div.form(key='similar_words_form')
            with form:
                cols = st.columns(number_terms)
                for k, col in enumerate(cols):
                    selected_words = col.multiselect(user_input[k], options_list[k], key=k)
                    new_words_to_search.extend(selected_words)

                new_words_to_search = list(dict.fromkeys(new_words_to_search))
                submitted = st.form_submit_button('Search')
                
        if submitted:
            user_input.extend(new_words_to_search)
            user_input = list(dict.fromkeys(user_input))
            result_word, sim_words, similar_word, similarity, labels, label_dict, color_map = plot_data_config(user_input, model)

            with plot_container:
                if dimension == '2D':
                    display_scatterplot_2D(model, user_input, similar_word, labels, color_map, annotation, dim_red, perplexity, learning_rate, iteration, top_n)
                else:
                    display_scatterplot_3D(model, user_input, similar_word, labels, color_map, annotation, dim_red, perplexity, learning_rate, iteration, top_n)

            with table_cells_div:
                similarities_table_streamlit(user_input, model)

            result_word, sim_words, similar_word, similarity, labels, label_dict, color_map = plot_data_config(new_words_to_search, model)
            with subplots_section:
                number_terms = len(user_input)
                count=0
                i=0
                options_list = list(split_list(similar_word[:-number_terms], number_terms))

                if number_terms % 2 == 0:
                    number_containers = int(number_terms/2)
                else:
                    number_containers = int(number_terms/2) + 1
                
                if (previous_number_terms % 2 != 0 and (previous_number_containers % 2 == 0 or previous_number_containers == 1)):
                    with col2_plot:
                        horizontal_bar(similar_word[count:count+top_n], similarity[count:count+top_n], new_words_to_search[0])
                    i = 1
                    count = count + top_n
                
                subplots_plots_div.empty()
                subplots_plots_div = subplots_section.container()
                with subplots_plots_div:
                    for j in range(number_containers):
                        subplots_plots_div_row = subplots_plots_div.container()
                        col1, col2 = subplots_plots_div_row.columns(2)
                        col1_plot = col1.empty()
                        col2_plot = col2.empty()

                        try:
                            with col1_plot:
                                horizontal_bar(similar_word[count:count+top_n], similarity[count:count+top_n], new_words_to_search[i])
                        except:
                            pass

                        i = i + 1 
                        count = count + top_n
                        try:
                            with col2_plot:
                                horizontal_bar(similar_word[count:count+top_n], similarity[count:count+top_n], new_words_to_search[i])
                        except:
                            pass

                        count = count + top_n
                        i = i + 1
