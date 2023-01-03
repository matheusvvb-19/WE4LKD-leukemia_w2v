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
    st.markdown('lalala')
    
    st.header('2. Approaches')
    st.subheader('2.1 Word2Vec')
    st.markdown('lalala')
    
    st.subheader('2.2 BERT-based')
    st.markdown('lalala')
    
    st.header('3. About this web app')
    st.markdown('lalala')
    
    st.header('References')
    st.markdown('[1] Vahe Tshitoyan et al. “Unsupervised word embeddings capture latent knowledge frommaterials science literature”. Nature571 (2019), pp. 95–98')
