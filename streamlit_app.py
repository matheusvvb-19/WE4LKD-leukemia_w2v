import plotly
import plotly.graph_objs as go
import numpy as np
import pickle
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

filename = 'sample_model.bin'
model = pickle.load(open(filename, 'rb'))

def append_list(sim_words, words):
    
    list_of_words = []
    
    for i in range(len(sim_words)):
        
        sim_words_list = list(sim_words[i])
        sim_words_list.append(words)
        sim_words_tuple = tuple(sim_words_list)
        list_of_words.append(sim_words_tuple)
        
    return list_of_words


def display_scatterplot_3D(model, user_input=None, words=None, label=None, color_map=None, annotation='On',  dim_red = 'PCA', perplexity = 0, learning_rate = 0, iteration = 0, topn=0, sample=10):
    
    if words == None:
        if sample > 0:
            words = np.random.choice(list(model.wv.vocab.keys()), sample)
        else:
            words = [ word for word in model.wv.vocab ]
    
    word_vectors = np.array([model[w] for w in words])
    
    if dim_red == 'PCA':
        three_dim = PCA(random_state=0).fit_transform(word_vectors)[:,:3]
    else:
        three_dim = TSNE(n_components = 3, random_state=0, perplexity = perplexity, learning_rate = learning_rate, n_iter = iteration).fit_transform(word_vectors)[:,:3]

    color = 'blue'
    quiver = go.Cone(
        x = [0,0,0], 
        y = [0,0,0],
        z = [0,0,0],
        u = [0.5,0,0],
        v = [0,0.5,0],
        w = [0,0,0.5],
        anchor = "tail",
        colorscale = [[0, color] , [1, color]],
        showscale = False,
        opacity = 0
        )
    
    data = [quiver]

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

    st.plotly_chart(plot_figure)

def horizontal_bar(word, similarity):
    
    similarity = [ round(elem, 2) for elem in similarity ]
    
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

def display_scatterplot_2D(model, user_input=None, words=None, label=None, color_map=None, annotation='On', dim_red = 'PCA', perplexity = 0, learning_rate = 0, iteration = 0, topn=0, sample=10):
    
    if words == None:
        if sample > 0:
            words = np.random.choice(list(model.wv.vocab.keys()), sample)
        else:
            words = [ word for word in model.wv.vocab ]
    
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

dim_red = st.sidebar.selectbox(
 'Select dimension reduction method',
 ('PCA','TSNE'))
dimension = st.sidebar.selectbox(
     "Select the dimension of the visualization",
     ('2D', '3D'))
user_input = st.sidebar.text_input("Escreva a(s) palavra(s) que deseja buscar. Para pesquisar mais de uma, separe as palavras por vírgula.",'')
top_n = st.sidebar.slider('Selecione o tamanho da vizinhança desejada ',
    5, 30, (5))
annotation = st.sidebar.radio(
     "Enable or disable the annotation on the visualization",
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
    
    for words in user_input:
    
        sim_words = model.most_similar(words, topn = top_n)
        sim_words = append_list(sim_words, words)
            
        result_word.extend(sim_words)
    
    similar_word = [word[0] for word in result_word]
    similarity = [word[1] for word in result_word] 
    similar_word.extend(user_input)
    labels = [word[2] for word in result_word]
    label_dict = dict([(y,x+1) for x,y in enumerate(set(labels))])
    color_map = [label_dict[x] for x in labels]
    

st.title('Visualizar Word Embedding Visualization a partir de Similaridade Cosseno')

st.markdown('First, choose which dimension of visualization that you want to see. There are two options: 2D and 3D.')
           
st.markdown('Next, type the word that you want to investigate. You can type more than one word by separating one word with other with comma (,).')

st.markdown('With the slider in the sidebar, you can pick the amount of words associated with the input word you want to visualize. This is done by computing the cosine similarity between vectors of words in embedding space.')
st.markdown('Lastly, you have an option to enable or disable the text annotation in the visualization.')

if dimension == '2D':
    st.header('2D Visualization')
    st.write('For more detail about each point (just in case it is difficult to read the annotation), you can hover around each points to see the words. You can expand the visualization by clicking expand symbol in the top right corner of the visualization.')
    display_scatterplot_2D(model, user_input, similar_word, labels, color_map, annotation, dim_red, perplexity, learning_rate, iteration, top_n)
else:
    st.header('3D Visualization')
    st.write('For more detail about each point (just in case it is difficult to read the annotation), you can hover around each points to see the words. You can expand the visualization by clicking expand symbol in the top right corner of the visualization.')
    display_scatterplot_3D(model, user_input, similar_word, labels, color_map, annotation, dim_red, perplexity, learning_rate, iteration, top_n)

st.header('5 palavras mais similares a cada um do(s) termo(s) de entrada ')
count=0
for i in range (len(user_input)):
    
    st.write('As palavras mais similaes a '+str(user_input[i])+' são:')
    horizontal_bar(similar_word[count:count+5], similarity[count:count+5])
    
    count = count+top_n
