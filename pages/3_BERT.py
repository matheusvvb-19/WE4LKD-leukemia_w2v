# IMPORTS:
import streamlit as st

# GLOBAL VARIABLES:

# MAIN PROGRAM:
if __name__ == '__main__':
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
    
    with st.sidebar.form('sidebar_form'):
        st.header('Models exploration settings')
        
        loaded_model = st.selectbox(
         'Choose one of the preloaded models:',
         ('10: 1900 - 2022', '9: 1900 - 2016', '8: 1900 - 2014', '7: 1900 - 2013', '6: 1900 - 2011', '5: 1900 - 2009', '4: 1900 - 2001', '3: 1900 - 1999', '2: 1900 - 1977', '1: 1900 - 1967'))
        
        top_n = st.slider('Select the neighborhood size',
            5, 20, (5), 5)
        
        submitted = st.form_submit_button('Apply settings')
        
    st.sidebar.header('GitHub Repository')
    st.sidebar.markdown("[![Foo](https://cdn-icons-png.flaticon.com/32/25/25231.png)](https://github.com/matheusvvb-19/WE4LKD-leukemia_w2v)")
