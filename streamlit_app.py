#based on https://towardsdatascience.com/visualizing-word-embedding-with-pca-and-t-sne-961a692509f5

import plotly, pickle
import plotly.graph_objs as go
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from get_n_common_words_english import get_most_common
from gensim.models import Word2Vec, KeyedVectors

specific_domain = []
filename = 'sample_model.model'
model = pickle.load(open(filename, 'rb'))

# domains table:
domains_table = pd.read_csv('https://docs.google.com/spreadsheets/d/' + 
                   '1SgYG4gZuL3grEHHAZt49dAUw_jFAc4LADajFeGAf2-w' +
                   '/export?gid=0&format=csv',
                  )

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

def horizontal_bar(word, similarity):
    similarity = [round(elem, 2) for elem in similarity]
    
    data = go.Bar(
            x= similarity,
            y= word,
            orientation='h',
            text = similarity,
            marker_color= 4,
            textposition='auto')

    layout = go.Layout(
            font = dict(size=20),
            xaxis = dict(showticklabels=False, automargin=True),
            yaxis = dict(showticklabels=True, automargin=True,autorange="reversed"),
            margin = dict(t=20, b= 20, r=10)
            )

    plot_figure = go.Figure(data = data, layout = layout)
    st.plotly_chart(plot_figure)

def display_scatterplot_2D(model, user_input=None, words=None, label=None, color_map=None, annotation='On', dim_red = 'TSNE', perplexity = 0, learning_rate = 0, iteration = 0, topn=0, sample=10):
    if words == None:
        if sample > 0:
            words = np.random.choice(list(model.wv.vocab.keys()), sample)
        else:
            words = [word for word in model.wv.vocab]
    
    word_vectors = []
    try:
      word_vectors = np.array([model[w] for w in words])
    except:
      pass
    
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

uploaded_file = st.sidebar.file_uploader("Faça upload do modelo:")
if uploaded_file is not None:
    model = pickle.load(uploaded_file)
    #model = KeyedVectors.load_word2vec_format(open(uploaded_file))
    model.init_sims()

    restrict_domain = st.sidebar.selectbox("Restringir domínio do vocabulário:",
    ('geral', 'câncer'))
    if restrict_domain != 'geral':
        if restrict_domain == 'câncer':
            specific_domain = domains_table['name'].tolist()
            wv_restrict_w2v(model, set(specific_domain), True)
    else:
        common_words_number = st.sidebar.selectbox('Selecione a quantidade de palavras mais comuns da língua inglesa que deseja remover da visualização ',
        ('None', '5000', '10000', '15000', '20000'))
        if common_words_number != 'None':
            common_words = get_most_common(int(common_words_number))
            wv_restrict_w2v(model, set(common_words))
    
    
dim_red = st.sidebar.selectbox(
 'Selecione o método de redução de dimensionalidade',
 ('TSNE','PCA'))
dimension = st.sidebar.selectbox(
     "Selecione a dimensão de visualização",
     ('2D', '3D'))
user_input = st.sidebar.text_input("Escreva as palavras que deseja buscar. Para mais de uma palavra, as separe por vírgula (,)",'')
top_n = st.sidebar.slider('Selecione o tamanho da vizinhança a ser visualizada ',
    5, 30, (5))
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
    user_input = [x.strip() for x in user_input.split(',')]
    result_word = []
    sim_words = []
    
    for words in user_input:
        try:
            sim_words = model.wv.most_similar(words, topn = top_n)
            sim_words = append_list(sim_words, words)
            result_word.extend(sim_words)
        except KeyError:
            st.error("A palavra {} não está presente no vocabulário deste modelo.".format(words))
            user_input = None
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
    

st.title('Visualizador Word Embedding por Similaridade Cosseno')

#st.header('This is a web app to visualize the word embedding.')
st.markdown('Primeiramente, faça upload do modelo de representação distribuída com extensão ".model". Depois, escolha a quantidade de palavras comuns da língua inglesa que deseja remover da visualização, a remoção dessas palavras pode melhorar sua investigação, visto que muitas vezes são palavras fora do contexto médico.')
           
st.markdown('Depois, selecione o método de redução de dimensionalidade. Se você não sabe o que significa, deixe o valor padrão "TSNE". Abaixo dessa opção, defina a quantidade de dimensões do gráfico (2D ou 3D).')

st.markdown('É possível também buscar por palavras específicas, digitando elas no campo. Para mais de uma palavra, as separe por vírgulas. Tome cuidado, caso você decida remover muitas palavras comuns, talvez a palavra que você busque não esteja mais disponível.')
st.markdown('Por fim, é possível habiitar e desabilitar os rótulos de cada ponto no gráfico.')

if dimension == '2D':
    st.header('Visualização 2D')
    #st.write('For more detail about each point (just in case it is difficult to read the annotation), you can hover around each points to see the words. You can expand the visualization by clicking expand symbol in the top right corner of the visualization.')
    display_scatterplot_2D(model, user_input, similar_word, labels, color_map, annotation, dim_red, perplexity, learning_rate, iteration, top_n)
else:
    st.header('Visualização 3D')
    #st.write('For more detail about each point (just in case it is difficult to read the annotation), you can hover around each points to see the words. You can expand the visualization by clicking expand symbol in the top right corner of the visualization.')
    display_scatterplot_3D(model, user_input, similar_word, labels, color_map, annotation, dim_red, perplexity, learning_rate, iteration, top_n)

if user_input != '' and user_input:
    #st.header('Palavras mais similares a cada termo buscado')
    count=0
    for i in range (len(user_input)):

        st.write('As palavras mais similares a '+str(user_input[i])+' são:')
        horizontal_bar(similar_word[count:count+top_n], similarity[count:count+top_n])

        count = count+top_n
