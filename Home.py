# based on https://towardsdatascience.com/visualizing-word-embedding-with-pca-and-t-sne-961a692509f5

# IMPORTS:
import streamlit as st

# GLOBAL VARIABLES:


# FUNCTIONS:
def set_page_layout():
    '''Define some configs of the Streamlit App page, only front-end settings.'''

    st.set_page_config(
        page_title="WE4LKD | Home",
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
                content:'Developed by Matheus Volpon.'; 
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
    
    st.title('Word Embedding For Latent Knowledge Discovery AML')
    
    st.header('1. Introduction')
    st.markdown('In 2019, Tshitoyan *et al*. [1] employed Skip-Gram to create a model representation froma set of prefaces of scientific papers published in the field of materials science. By computing the vectors in an unsupervised manner, the model captured complex concepts such as the symbols of the periodic table and their corresponding element, and the relationships between material properties and structures. Subsequently, the authors proved that those models were able to relate materials used in thermoelectric power generation years before this use was explicitly mentioned in any paper. Therefore, they conclude that it is possible to use distributed representation models to discover latent knowledge: information that may be implicit in a set of texts and that would hardly be perceptible to humans.')
    st.markdown('In the same study, Tshitoyan *et al*. [1] suggested that future work could investigate textsfrom other fields of knowledge and evaluate modern neural network architectures that consider the context of words. In this way, this research project aims to check if it is possible to discover latent knowledge in medical articles about Acute Myeloid Leukemia (AML), an aggressive typeof cancer without known effective treatment. The methods and approaches used in this projects are explained in the subsequent section.')
    
    st.header('2. Approaches')
    st.subheader('2.1 Word2Vec')
    st.markdown('lalala')
    
    st.subheader('2.2 BERT-based')
    st.markdown('lalala')
    
    st.header('3. About this web app')
    st.markdown('lalala')
    
    st.header('References')
    st.markdown('[1] Vahe Tshitoyan et al. “Unsupervised word embeddings capture latent knowledge frommaterials science literature”. Nature571 (2019), pp. 95–98')
    
    st.markdown('This project is fully funded by the Fundação de Amparo à Pesquisa do Estado de São Paulo (FAPESP, Brazil), under identification code 2021/13054-8.')
    st.text('Undergraduate scholarship student: Matheus Vargas Volpon Berto.\nProfessor and supervisor: Dr. Tiago Almeida.\nEducational institution: Federal University of São Carlos (UFSCar) - Sorocaba campus.')
    st.markdown('In addition, part of the project was developed in partnership with the Computer Science Department of the University of Sheffield (UK), under the supervision of Dr. Carolina Scarton, identification code 2022/07236-9.')
