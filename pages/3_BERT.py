# IMPORTS:
import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# GLOBAL VARIABLES:

# FUNCTIONS:
@st.cache()
def get_sentences_dataset():
    sentences = [
        "The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go.",
        "The person box was packed with jelly many dozens of months later.",
        "Standing on one's head at job interviews forms a lasting impression.",
        "It took him a month to finish the meal.",
        "He found a leprechaun in his walnut shell."
    ]
    
    return sentences

def flat_list(composed_list):
    if any(isinstance(x, list) for x in composed_list):
        composed_list = [item for sublist in composed_list for item in sublist]

    return composed_list

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
         ('10: 1921 - 2022', 
          '9: 1921 - 2016',
          '8: 1921 - 2014',
          '7: 1921 - 2013',
          '6: 1921 - 2011',
          '5: 1921 - 2009',
          '4: 1921 - 2001',
          '3: 1921 - 1999',
          '2: 1921 - 1977',
          '1: 1921 - 1967'))
        
        top_n = st.slider('Select the neighborhood size',
            5, 20, (5), 5)
        
        input_sentence = st.text_input(label='Input sentence', max_chars=200, help='Type the sentence that you want to compare to the others.')
        
        submitted = st.form_submit_button('Apply settings')
        if submitted or st.session_state['execution_counter'] != 0:
            tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
            model = AutoModel.from_pretrained('matheusvolpon/WE4LKD_AML_distilbert_1921_{}'.format(loaded_model[-4:]))
            
            # initialize dictionary that will contain tokenized sentences
            tokens = {'input_ids': [], 'attention_mask': []}

            for sentence in sentences:
                # tokenize sentence and append to dictionary lists
                new_tokens = tokenizer.encode_plus(sentence, max_length=128, truncation=True,
                                                   padding='max_length', return_tensors='pt')
                tokens['input_ids'].append(new_tokens['input_ids'][0])
                tokens['attention_mask'].append(new_tokens['attention_mask'][0])

            # reformat list of tensors into single tensor
            tokens['input_ids'] = torch.stack(tokens['input_ids'])
            tokens['attention_mask'] = torch.stack(tokens['attention_mask'])
            
            outputs = model(**tokens)
            
            embeddings = outputs.last_hidden_state
            
            attention_mask = tokens['attention_mask']
            
            mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            
            masked_embeddings = embeddings * mask
            summed = torch.sum(masked_embeddings, 1)
            summed_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled = summed / summed_mask
            
            # convert from PyTorch tensor to numpy array
            mean_pooled = mean_pooled.detach().numpy()

            # calculate
            similarities = flat_list(cosine_similarity([mean_pooled[0]], mean_pooled[1:]).tolist())
            st.markdown(similarities)
        
    st.sidebar.header('GitHub Repository')
    st.sidebar.markdown("[![Foo](https://cdn-icons-png.flaticon.com/32/25/25231.png)](https://github.com/matheusvvb-19/WE4LKD-leukemia_w2v)")
    
    st.title('Sentence Viewer')
    st.header('Sentence Embedding Visualization Based on Cosine Similarity')
    with st.expander('How to use this app'):
        st.markdown('**Sidebar**')
        st.markdown('lalala')

        st.markdown('**Main window**')
        st.markdown('_Hint: To see this window content better, you can minimize the sidebar._')
        st.markdown('lalala')
