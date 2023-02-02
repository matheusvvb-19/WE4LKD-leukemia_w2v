# IMPORTS:
import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# GLOBAL VARIABLES:

# FUNCTIONS:
@st.cache()
def get_sentences_dataset():
    data = [
        [1970, "Combination chemotherapy using L-asparaginase, daunorubicin, and cytarabine in adults with AML."],
        [1972, "Fourty-four patients with AML were treated with RP 22050, a new, semisynthetic derivative of daunorubicin."],
        [1974, "The use of a combination of daunorubicin and cytarabine in the treatment of AML in a district general hospital is described."],
        [1975, "Thirty-seven adults with AML (AML) were treated with a combination of daunorubicin, cytarabine."],
        [1976, "Maintenance of remission in adult AML using intermittent courses of cytarabine (cytarabine) and thioguanine (thioguanine)."],
        [1978, "Normal infant after treatment of AML in pregnancy with daunorubicin."],
        [1979, "Enhanced cytarabine accumulation after administration of MTX was also observed in human AML cells."],
        [1980, "Cyclophosphamide, cytarabine and methotrexate versus cytarabine and thioguanine for AML in adults."],
        [1981, "Sequential combination of pyrazofurin and azacitidine in patients with AML and carcinoma."],
        [1981, "Three patients with AML received high doses of daunorubicin, first in the free form and later as complex with DNA."],
        [1981, "vincristine, and 6alpha-methylprednisone (ROAP) to 91 patients with AML who were 50 yr of age or older."],
        [1982, "Relapses in nine patients with AML were treated with a combination of aclarubicin (ACR) and cytarabine (cytarabine)."],
        [1982, "Treatment of relapsed AML with a combination of aclarubicin and cytarabine."],
        [1983, "The effects of dexamethasone and tetradecanoyl phorbol acetate on plasminogen activator release by human AML cells."],
        [1983, "Therapeutic responses included a partial remission in a patient with AML (AML) refractory to cytarabine."],
        [1983, "dexamethasone, fluorouracil or methotrexate on human AML cell line KG-1."],
        [1984, "Delayed growth arrest was observed in HL-60 AML cells after exposure to thioguanine (thioguanine)."],
        [1984, "The current status of, and the future prospects for, low-dose cytarabine therapy in patients with AML is discussed."],
        [1984, "The use of intermediate dose cytarabine (ID cytarabine) in the treatment of AML in relapse."],
        [1985, "Low-dose cytarabine regimen induced a complete remission with normal karyotypes in a case with hypoplastic AML with No."],
        [1985, "Treatment of AML with a daunorubicin cytarabine thioguanine regimen without maintenance therapy."],
        [1985, "doxorubicin and cytarabine contribute equally to prediction of response in AML with improved confidence level."],
        [1986, "A comparison of in vitro sensitivity of AML precursors to mitoxantrone, 4'deoxydoxorubicin, idarubicin and daunorubicin."],
        [1986, "A critical appraisal of low-dose cytarabine in patients with AML and myelodysplastic syndromes."],
        [1986, "Patients with AML received cytarabine, either 2 or 6 g/m2/72 h by continuous infusion."],
        [1987, "Etoposide in combination with cytarabine, doxorubicin, and thioguanine for treatment of AML in a protocol adjusted for age."],
        [1987, "Four patients with AML (AML) and three with myelodysplastic syndrome (MDS) were given low dose cytarabine (cytarabine) therapy."],
        [1987, "Low dose arabinosyl cytosine (cytarabine) is effective for treatment of AML (ANLL)."],
        [1987, "Low-dose cytarabine (LD-Ara C) treatment in dysmyelopoietic syndromes (DMPS) and AML (AML)."],
        [1988, "A case of persistent cerebellar dysfunction following high-dose cytarabine (cytarabine) treatment of AML is reported."],
        [1988, "Interleukin 3 enhances the cytotoxic activity of cytarabine (cytarabine) on AML (AML) cells."],
        [1988, "cytarabine is the most important drug in the clinical chemotherapy of AML."],
        [1989, "Eight patients with AML were treated with a combination of daunorubicin (1.5 mg/kg body weight) and cytarabine."],
        [1989, "High-dose cytarabine (HDARA-C) is an effective but toxic treatment for AML (AML)."],
        [1989, "Standard induction and low dose cytarabine treatment in patients over 60 with AML or MDS."],
        [1989, "The urine of a patient who suffered from AML was red coloured after administration of mitoxantrone and etoposid."],
        [1990, "Concordant changes of pyrimidine metabolism in blasts of two cases of AML after repeated treatment with cytarabine in vivo."],
        [1990, "Rapid Remission Induction and Improved Disease Free Survival in AML Using Daunorubicin, cytarabine, and lomustine."],
        [1990, "The original LBN AML is sensitive to the chemotherapeutic agent cyclophosphamide (CY)."],
        [1991, "Growth factors also affect the sensitivity of AML blast cells to cytarabine (cytarabine)."],
        [1991, "Low dose cytarabine (LDARAC) has been commonly used in the treatment of AML (AML) in elderly patients."],
        [1991, "Treatment with HGFs resulted in [3H] cytarabine DNA incorporation that was significantly higher in AML blasts versus NBMMC."],
        [1992, "Both MAbs reacted positively in 1 patient with AML (ANLL) at diagnosis who achieved remission with teniposide and cytarabine."],
        [1992, "Thus, AML blasts respond to fazarabine in culture with a pattern similar to that of 5-aza and opposite to that of cytarabine."],
        [1992, "We conclude that intermediate dose cytarabine did not substantially improve results of induction chemotherapy for AML."],
        [1993, "No difference of intracellular daunorubicin accumulation was observed between CD34+ and CD34- AML cells of 4 P-170- patients."],
        [1993, "This led us to administer fludarabine and cytarabine to 59 patients with AML in relapse or unresponsive to initial therapy."],
        [1993, "cytarabine (cytarabine) and etoposide are often used in combination in the treatment of AML (AML)."],
        [1994, "Cytarabine (cytarabine) is currently used in the treatment of adult AML (AML)."],
        [1994, "Long-term results following treatment of newly-diagnosed AML with continuous-infusion high-dose cytarabine."],
        [1994, "Structural analysis of the deoxycytidine kinase gene in patients with AML and resistance to cytarabine."],
        [1995, "Clonogenic data from fresh AML LDMCs not pretreated with growth factors demonstrated a heterogenous response to cytarabine."],
        [1995, "In summary, high-dose cytarabine/DNR consolidation can improve the long-term outcome of a subgroup of de novo AML patients."],
        [1995, "Inhibition of bcl-2 with antisense oligonucleotides induces apoptosis and increases the sensitivity of AML blasts to cytarabine."],
        [1996, "GM-CSF administered during induction treatment of AML with a DNR/cytarabine combination did not provide any clinical benefit."],
        [1996, "It has been shown recently in China that arsenictrioxide (arsenictrioxide) is a very effective treatment for AML (APL)."],
        [1996, "The clinical development of cytarabine for the treatment of AML (AML) provides a useful paradigm for the study of this process."],
        [1997, "Furthermore, 11 AML and four ALL patients were treated with fractionated daunorubicin at a dose of 50 mg/m2/week."],
        [1997, "Natural resistance of AML cell lines to mitoxantrone is associated with lack of apoptosis."],
        [1997, "The inclusion of HD 6MP and ID cytarabine in the treatment of AML in first remission appears to be feasible."],
        [1998, "Complete remission after treatment of AML with arsenictrioxide."],
        [1998, "Frequency of prolonged remission duration after high-dose cytarabine intensification in AML varies by cytogenetic subtype."],
        [1998, "arsenictrioxide as an inducer of apoptosis and loss of PML/RAR alpha protein in AML cells."],
        [1999, "Anthracyclines such as daunorubicin (daunorubicin) are typically used to treat AML and can induce drug resistance."],
        [1999, "It was recently reported that arsenictrioxide (As(2)O(3)) can induce complete remission in patients with AML (APL)."],
        [1999, "The phosphoinositide 3-kinase/Akt pathway is activated by daunorubicin in human AML cell lines."],
        [2000, "Recently, DNR has been replaced in many centers by idarubicin (IDA) as the first choice anthracycline in AML treatment."],
        [2000, "arsenictrioxide, like all-trans-retinoic acid (RA), induces differentiation of AML (APL) cells in vivo."],
        [2000, "gemtuzumab-ozogamicin is a promising agent in the treatment of patients with AML that expresses CD33."],
        [2001, "Arsenic oxide (arsenictrioxide) has recently been reported to induce remission in a high percentage of patients with AML (APL)."],
        [2001, "Remission induction chemotherapy for AML typically combines cytarabine with an anthracycline or anthracycline derivative."],
        [2001, "These results suggest that dose intensification of cytarabine benefits children with AML and inv(16), as is the case in adults."],
        [2002, "CMV infection occurred rarely during cytarabine and anthracyclin based induction therapy for AML or RAEB."],
        [2002, "Compared with DS ALL, DS AML cells were significantly more sensitive to cytarabine only (21-fold)."],
        [2002, "arsenictrioxide promotes histone H3 phosphoacetylation at the chromatin of CASPASE-10 in AML cells."],
        [2003, "The interaction of daunorubicin and mitoxantrone with the red blood cells of AML patients."],
        [2003, "gemtuzumab-ozogamicin has moderate activity as a single agent in patients with CD33-positive refractory or relapsed AML (AML)."],
        [2003, "vincristine (vincristine) is an effective drug against acute lymphoblastic leukemia (ALL), many solid tumors, but not AML."],
        [2004, "Recent resurgence in the use of arsenictrioxide is related to its high efficacy in AML (APL)."],
        [2004, "The impressive activity of arsenictrioxide in AML (APL) has renewed the interest in this old compound."],
        [2004, "mitoxantrone (MTZ) has been shown to be effective in the treatment of newly diagnosed AML (AML)."],
        [2005, "Combined treatment of AML cells by cytarabine or IDA with anti-CD33 mAb resulted in higher levels of SHP-1 phosphorylation."],
        [2005, "Cytarabine (cytarabine) is the most effective agent for the treatment of AML (AML)."],
        [2005, "Daunorubicin (daunorubicin) is commonly used to treat AML (AML)."],
        [2006, "Recently, patients with AML (APL) have experienced significant clinical gains after treatment with arsenictrioxide."],
        [2006, "Resistance to cytarabine (cytarabine) is a major problem in the treatment of patients with AML (AML)."],
        [2006, "We searched for mechanisms of resistance in 6 patients with AML who had relapses upon midostaurin treatment."],
        [2007, "Combination of all-trans-retinoic acid and gemtuzumab-ozogamicin in an elderly patient with AML and severe cardiac failure."],
        [2007, "Reversible posterior leukoencephalopathy syndrome after repeat intermediate-dose cytarabine chemotherapy in a patient with AML."],
        [2007, "midostaurin has proven activity in the treatment of AML (AML)."],
        [2008, "Previously only acute forms of leukemia particularly AML (APL) have been associated with mitoxantrone treatment in MS."],
        [2008, "arsenictrioxide has remarkable efficacy in AML and is approved by the US Food and Drug Administration for this indication."],
        [2008, "gemtuzumab-ozogamicin (gemtuzumab-ozogamicin) monotherapy is reported to yield a 20-30% response rate in advanced AML (AML)."],
        [2009, "The standard therapeutic approaches for AML (AML) continue to be based on anthracyclines and cytarabine."],
        [2009, "The success of arsenictrioxide in the treatment of AML has renewed interest in the cellular targets of As(III) species."],
        [2009, "sirolimus, the mTOR kinase inhibitor, sensitizes AML cells, HL-60 cells, to the cytotoxic effect of arabinozide cytarabine."],
        [2010, "Promising reports exist regarding the use of arsenictrioxide (arsenictrioxide) as first-line treatment in AML (APL)."],
        [2010, "Today, arsenictrioxide is used as one of the standard therapies for AML (APL)."],
        [2010, "arsenictrioxide enhances the cytotoxic effect of thalidomide in a KG-1a human AML cell line."],
        [2011, "CD34+ AML cells are 10-15-fold more resistant to daunorubicin (daunorubicin) than CD34- AML cells."],
        [2011, "The addition of arsenictrioxide to low-dose cytarabine in older patients with AML does not improve outcome."],
        [2011, "gemtuzumab-ozogamicin combined with chemotherapy is a feasible treatment regimen in AML patients."],
        [2012, "The cytarabine (cytarabine)-based chemotherapy is the major remedial measure for AML (AML)."],
        [2012, "Upfront maintenance therapy with arsenictrioxide in AML provides no benefit for non-t(15;17) subtype."],
        [2012, "We conclude that high BMI should not be a barrier to administer high-dose cytarabine-containing regimens for AML induction."],
        [2013, "We explored the differences by scrutinizing a case of gemtuzumab-ozogamicin (gemtuzumab-ozogamicin) in patients with AML (AML)."],
        [2013, "azacitidine and lenalidomide both have meaningful single-agent clinical activity in HR-MDS and AML with del(5q)."],
        [2013, "cytarabine (cytarabine or cytarabine) has been one of the cornerstones of treatment of AML since its approval in 1969."],
        [2014, "The experience with gemtuzumab-ozogamicin has highlighted both the potential value and limitations of antibodies in AML (AML)."],
        [2014, "Thioredoxin-1 inhibitor PX-12 induces human AML cell apoptosis and enhances the sensitivity of cells to arsenictrioxide."],
        [2014, "gemtuzumab-ozogamicin was the first example of antibody-directed chemotherapy in cancer, and was developed for AML."],
        [2015, "Finally, Msi2 silencing in AML cells also enhanced their chemosensitivity to daunorubicin."],
        [2015, "azacitidine sensitization to arsenictrioxide treatment was re-capitulated also in primary AML samples."],
        [2015, "gemtuzumab-ozogamicin in combination with intensive chemotherapy in relapsed or refractory AML."],
        [2016, "Venetoclax demonstrated activity and acceptable tolerability in patients with AML and adverse features."],
        [2016, "arsenictrioxide (arsenictrioxide) has demonstrated clinical efficacy in AML (APL) and in vitro activity in various solid tumors."],
        [2016, "miR-29c is of prognostic value and influences response to azacitidine treatment in older AML patients."],
        [2017, "Intensive combination chemotherapy including gemtuzumab-ozogamicin emerged as an effective salvage therapy in refractory AML."],
        [2017, "arsenictrioxide (arsenictrioxide) is an old drug that has recently been reintroduced as a therapeutic agent for AML (APL)."],
        [2017, "cytarabine (cytarabine) is one of the key drugs for treating AML (AML)."],
        [2018, "The dose escalation of cytarabine in induction therapy lead to improved remission rates in the elderly AML patients."],
        [2018, "The ectopic expression of hTERT significantly attenuated the apoptotic effect of midostaurin on AML cells."],
        [2018, "arsenictrioxide enhance reactive oxygen species levels and induce apoptosis and suppresses proliferation in AML cells."],
        [2019, "MALAT1 knockdown inhibits proliferation and enhances cytarabine chemosensitivity by upregulating miR-96 in AML cells."],
        [2019, "Venetoclax is approved for older untreated AML (AML) patients."],
        [2019, "We found that venetoclax and voreloxin synergistically induced apoptosis in multiple AML cell lines."],
        [2020, "Venetoclax is a highly selective BCL-2 inhibitor that has been approved by the FDA for treating elderly AML patients."],
        [2020, "Venetoclax-based therapy can induce responses in approximately 70% of older previously untreated patients with AML (AML)."],
        [2020, "arsenictrioxide (arsenictrioxide) is one of the most effective drugs for treatment of AML (APL)."],
        [2021, "CD44 loss of function sensitizes AML cells to the BCL-2 inhibitor venetoclax by decreasing CXCL12-driven survival cues."],
        [2021, "EVs derived from both new cases and relapsed AML patients significantly reduced idarubicin-induced apoptosis in the U937 cells."],
        [2021, "For decades two chemotherapeutic agents, cytarabine and daunorubicin, remained the backbone of AML therapy protocols."],
        [2022, "Acquired genetic mutations can confer resistance to arsenictrioxide (arsenictrioxide) in the treatment of AML (APL)."],
        [2022, "Azacitidine-induced reconstitution of the bone marrow T cell repertoire is associated with superior survival in AML patients."],
        [2022, "BCL-2 inhibition has been shown to be effective in AML (AML) in combination with hypomethylating agents or low-dose cytarabine."],
    ]
    
    #df = pd.read_csv('https://docs.google.com/spreadsheets/d/' + '1A-O505D_vkOprtWKMyQ1hIgNlaLwwBZnIZukiiVTncQ' + '/export?gid=0&format=csv', sep=',', escapechar='\\')
    
    return pd.DataFrame(data, columns=['filename', 'sentences'])

def flat_list(composed_list):
    if any(isinstance(x, list) for x in composed_list):
        composed_list = [item for sublist in composed_list for item in sublist]

    return composed_list

# MAIN PROGRAM:
if __name__ == '__main__':
    if 'execution_counter' not in st.session_state:
        st.session_state['execution_counter'] = 0
        
    sentences_df = get_sentences_dataset()
        
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
        st.markdown('**Input**')
        st.markdown(input_sentence)
        
        st.session_state['execution_counter'] += 1

        aux_df = sentences_df[(sentences_df["filename"].values <= int(loaded_model[-4:]))]
        sentences = aux_df['sentences'].to_list()
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
        
        st.markdown('**Top {} similar sentences to the input**'.format(top_n))
        st.table(df_similar_sentences.head(top_n))
