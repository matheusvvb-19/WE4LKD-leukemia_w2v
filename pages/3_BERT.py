# IMPORTS:
import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# GLOBAL VARIABLES:

# FUNCTIONS:
def get_sentences_dataset():
    sentences = [
        "There were no deaths directly attributable to cyclophosphamide and no patients developed hemorrhagic cystitis or malignancy..",
        "High-dose cytarabine (HiDAC) and intermediate-dose cytarabine (IDAC) have been introduced as effective and safe consolidation chemotherapy in AML, with relatively low rates of life-threatening infections despite the high total dose of the cytostatic drug.",
        "We demonstrate that high CD14 expression is highly significantly associated with high cellular cytarabine and Dau resistance in univariate as well as multivariate analyses.",
        "Thus, our study suggests that arsenictrioxide not only inhibits the expression of MYC, PCNA, and MCM7 but also leads to cell cycle arrest and apoptosis in KG-1a cells.",
        "While arsenictrioxide (arsenictrioxide) is an infamous carcinogen, it is also an effective chemotherapeutic agent for AML and some solid tumors.",
        "Considering he's played football for only two years, he does it well.",
        "Dexamethasone, but not mifepristone, increased expression of delivered proteins such as GFP that are important for early identification of infected cells.",
    ]
    
    return sentences

def flat_list(composed_list):
    if any(isinstance(x, list) for x in composed_list):
        composed_list = [item for sublist in composed_list for item in sublist]

    return composed_list

# MAIN PROGRAM:
if __name__ == '__main__':
    if 'execution_counter' not in st.session_state:
        st.session_state['execution_counter'] = 0
        
    sentences = get_sentences_dataset()
        
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
        
        top_n = st.slider('Select the neighborhood size',
            5, 20, (5), 5)
        
        input_sentence = st.text_input(label='Input sentence', max_chars=128, help='Type the sentence that you want to compare to the others.')
        
        submitted = st.form_submit_button('Apply settings')
    
    st.sidebar.header('GitHub Repository')
    st.sidebar.markdown("[![Foo](https://cdn-icons-png.flaticon.com/32/25/25231.png)](https://github.com/matheusvvb-19/WE4LKD-leukemia_w2v)")
        
    st.title('Sentence Viewer')
    st.header('Sentence Embedding Visualization Based on Cosine Similarity')
    with st.expander('How to use this app'):
        st.markdown('**Sidebar**')
        st.markdown('Select the BERT-based model that you want to explore. Then, define the number of most similar sentences from the dataset that you want to compare to your input sentence. Finally, type your input sentence in the text box and click on "Apply settings"')

        st.markdown('**Main window**')
        st.markdown('_Hint: To see this window content better, you can minimize the sidebar._')
        st.markdown('lalala')
        
    if submitted or st.session_state['execution_counter'] != 0:
        st.markdown('Input')
        st.markdown(input_sentence)
        
        st.session_state['execution_counter'] += 1

        sentences.insert(0, input_sentence)

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
        
        data = {
            'sentence': [],
            'similarity': [],
        }

        for s, si in zip(sentences[1:], similarities):
            data['sentence'].append(s)
            data['similarity'].append(si)

        df_similar_sentences = pd.DataFrame(data).sort_values(by=['similarity'], ascending=False)
        df_similar_sentences = df_similar_sentences.head(top_n)
        
        st.markdown('Top {} similar sentences to the input'.format(top_n))
        st.table(df_similar_sentences)
