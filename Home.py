# IMPORTS:
import streamlit as st

# FUNCTIONS:
def set_page_layout():
    '''Define some configs of the Streamlit App page, only front-end settings.'''

    st.set_page_config(
        page_title="WE4LKD AML",
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
    
# MAIN PROGRAM:
if __name__ == '__main__':
    set_page_layout()
    
    st.sidebar.header('GitHub Repository')
    st.sidebar.markdown("[![Foo](https://cdn-icons-png.flaticon.com/32/25/25231.png)](https://github.com/matheusvvb-19/WE4LKD-leukemia_w2v)")
    
    st.title('Word Embedding For Latent Knowledge Discovery AML')
    
    st.header('1. Introduction')
    st.markdown('In 2019, Tshitoyan *et al*. [1] employed Skip-Gram to create a model representation froma set of prefaces of scientific papers published in the field of materials science. By computing the vectors in an unsupervised manner, the model captured complex concepts such as the symbols of the periodic table and their corresponding element, and the relationships between material properties and structures. Subsequently, the authors proved that those models were able to relate materials used in thermoelectric power generation years before this use was explicitly mentioned in any paper. Therefore, they conclude that it is possible to use distributed representation models to discover latent knowledge: information that may be implicit in a set of texts and that would hardly be perceptible to humans.')
    st.markdown('In the same study, Tshitoyan *et al*. [1] suggested that future work could investigate textsfrom other fields of knowledge and evaluate modern neural network architectures that consider the context of words. In this way, this research project aims to check if it is possible to discover latent knowledge in medical articles about Acute Myeloid Leukemia (AML), an aggressive typeof cancer without known effective treatment. The methods and approaches used in this projects are explained in the subsequent section.')
    
    st.header('2. Approaches')
    st.subheader('2.1 Word2Vec')
    st.markdown('In our first approach, we followed the strategy used by Tshitoyan *et al*. [1] and used Word2Vec models with the Skip-Gram architecture.')
    
    st.subheader('2.2 BERT-based')
    st.markdown('We also trained BERT-based models from scratch using the same corpus of biomedical articles. To compute unique embeddings for the words, we applied a mean pooling strategy using the [Flair NLP](https://github.com/flairNLP/flair) framework.')
    
    st.header('3. About this web app')
    st.markdown('This web app contains the models trained during the project and making it possible to users to explore the embeddings, more information about how to use each tool are detailed on the respective page - which can be accessed trough the left sidebar.')
    st.markdown("In the Word2Vec page, you can search for words in the models' vocab and visualize the nearest tokens in the vector space.")
    st.markdown('In the BERT page, you can find similar sentences from our dataset according to an input sentence.')
    
    st.header('References')
    st.markdown('[1] Vahe Tshitoyan et al. “Unsupervised word embeddings capture latent knowledge frommaterials science literature”. Nature571 (2019), pp. 95–98')
    
    st.header('Funding')
    st.markdown('This project is fully funded by the Fundação de Amparo à Pesquisa do Estado de São Paulo (FAPESP, Brazil), identification codes 2021/13054-8 and 2022/07236-9.')
    st.markdown('Undergraduate scholarship student: Matheus Vargas Volpon Berto.')
    st.markdown('Professor and supervisor: Dr. Tiago Almeida.')
    st.markdown('Educational institution: Federal University of São Carlos (UFSCar) - Sorocaba campus.')
    st.markdown("In addition, part of the project was developed in partnership with the Computer Science Department of the University of Sheffield (UK), under the supervision of Dr. Carolina Scarton.")
