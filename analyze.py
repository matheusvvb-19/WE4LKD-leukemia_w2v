import os, gensim, sys, pickle, csv, re, string
import pandas as pd
from pathlib import Path
from gensim.models import Word2Vec
from os import listdir
from os.path import isfile, join
from fpdf import FPDF
from datetime import datetime, timezone, timedelta
from get_n_common_words_english import get_most_common
from streamlit_app import restrict_w2v, wv_restrict_w2v
from clean_text import replace_synonyms

base_compounds = ['cytarabine', 'daunorubicin', 'azacitidine', 'gemtuzumab ozogamicin', 'midostaurin', 'cpx-351', 'ivosidenib', 'venetoclax', 'enasidenib', 'gilteritinib', 'glasdegib']

def get_file_size(file_path):
    size = os.path.getsize(file_path)
    return size

def convert_bytes(size, unit=None):
    if unit == "KB":
        return '{} {}'.format(round(size / 1024), 'KB')
    elif unit == "MB":
        return '{} {}'.format(round(size / 1024 * 1024), 'MB')
    elif unit == "GB":
        return '{} {}'.format(round(size / 1024 * 1024 * 1024), 'GB')
    else:
        return '{} {}'.format(size, 'bytes')

def create_pdf_header(command, filenames, n = 15):
    pdf = FPDF('P', 'mm', 'A4')   
    pdf.add_page()
    pdf.set_font("Arial", size = 12)
    pdf.set_margins(3,10,3)


    diferenca = timedelta(hours=-3)
    data_hora_atuais = datetime.now()
    fuso_horario = timezone(diferenca)
    data_hora_sao_paulo = data_hora_atuais.astimezone(fuso_horario)
    data_hora_sao_paulo_em_texto = data_hora_sao_paulo.strftime('%d/%m/%Y %H:%M')

    pdf.cell(200, 5, txt = "WE4LKD - AML",  ln = 1, align = 'C')
    pdf.cell(200, 10, txt = "Relatório de análise modelos Word2Vec", ln = 2, align = 'C')
    pdf.cell(200, 5, txt = data_hora_sao_paulo_em_texto, ln = 2, align = 'C')
    if isinstance(command, int):
        if command > 0:
            pdf.cell(200, 10, txt = 'Palavras comuns removidas: {}'.format(command), ln = 2, align = 'C')
        elif command == 0:
            pdf.cell(200, 10, txt = 'Palavras comuns removidas: zero', ln = 2, align = 'C')
    elif isinstance(command, str):
        if command == 'nci_cancer_drugs':
            pdf.cell(200, 10, txt = 'Restrição de domínio: drogas anti-câncer NCI', ln = 2, align = 'C')
        elif command == 'fda_drugs':
            pdf.cell(200, 10, txt = 'Restrição de domínio: drogas aprovadas pela FDA', ln = 2, align = 'C')
        else:
            pdf.cell(200, 10, txt = 'Restrição de domínio: {}'.format(command), ln = 2, align = 'C')

    pdf.cell(200, 5, txt = 'Arquivos analisados:', ln = 2, align = 'L')
    for e in filenames:
        size = get_file_size(e)
        pdf.cell(200, 5, txt=e+'    ' + convert_bytes(size, "KB"), ln=2, align='L')

    pdf.cell(0, 10, ' ', ln=2, align='L')
    pdf.cell(200, 5, txt = 'Procedimentos para análise:', ln = 2, align = 'L')
    if isinstance(command, int):
        pdf.multi_cell(0, 5, txt = '1. as {} palavras em inglês mais comuns foram eliminadas retiradas dos modelos antes da análise.'.format(command), align = 'L')
    elif(isinstance(command, str)):
        pdf.multi_cell(0, 5, txt = '1. o vocabulário dos modelos foi restrito, eliminando todas as palavras que não estão incluídas no domínioe escolhido: {}.'.format(command), align = 'L')

    pdf.multi_cell(0, 5, txt = '2. em cada modelo, buscar sinônimos de ativos ainda não descobertos na época, de acordo com a linha do tempo base. Caso os termos estejam no vocabulário, imprimir sua similaridade aos ativos já descobertos, e imprimir suas {} palavras mais próximas.'.format(n), align = 'L')
    pdf.multi_cell(0, 5, txt = '3. caso não sejam encontrados indícios de conhecimento latente, apenas a similaridade aos ativos já descobertos é impressa no relatório.')

    return pdf, data_hora_sao_paulo_em_texto

def investigate_models(pdf, model, model_year, i, n):
    flag = 0
    pdf.cell(0, 10, ' ', ln=2, align='L')
    pdf.cell(0, 5, 'Modelo {}: até o ano de {} (inclusive)'.format(i, model_year), ln=2, align='L')
    pdf.cell(0, 5, 'Tamanho do vocabulário: {}'.format(len(model.wv.vocab)), ln=2, align='L')

    vocab_len = len(model.wv.vocab)

    if vocab_len == 0:
        pdf.cell(0, 5, 'não é possível visualizar as palavras mais próximas a {}, só há 1 termo no vocabulário.'.format(s), ln=2, align='L')
    else:
        if model_year < 1968:
            pdf.multi_cell(0, 5, 'Buscar pela presença de ativos descobertos a partir de 1968: Cytarabine, Daunorubicin, Azacitidine, Gemtuzumab Ozogamicin, Midostaurin, CPX-351, Ivosidenib, Venetoclax, Enasidenib, Gilteritinib e Glasdegib.', align='L')
            flag = search_targets_1968(pdf, model, flag, n, 0)
            flag = search_targets_1978(pdf, model, flag, n)
            flag = search_targets_2000(pdf, model, flag, n)
            flag = search_targets_2002(pdf, model, flag, n)
            flag = search_targets_2010(pdf, model, flag, n)
            flag = search_targets_2012(pdf, model, flag, n)
            flag = search_targets_2014(pdf, model, flag, n)
            flag = search_targets_2015(pdf, model, flag, n)
            flag = search_targets_2017(pdf, model, flag, n)
        elif model_year < 1978:
            pdf.multi_cell(0, 5, 'Buscar pela presença de ativos descobertos a partir de 1978: Azacitidine, Gemtuzumab Ozogamicin, Midostaurin, CPX-351, Ivosidenib, Venetoclax, Enasidenib, Gilteritinib e Glasdegib.', align='L')
            search_targets_1968(pdf, model, flag, n, 1)
            flag = search_targets_1978(pdf, model, flag, n)
            flag = search_targets_2000(pdf, model, flag, n)
            flag = search_targets_2002(pdf, model, flag, n)
            flag = search_targets_2010(pdf, model, flag, n)
            flag = search_targets_2012(pdf, model, flag, n)
            flag = search_targets_2014(pdf, model, flag, n)
            flag = search_targets_2015(pdf, model, flag, n)
            flag = search_targets_2017(pdf, model, flag, n)
        elif model_year < 2000:
            pdf.multi_cell(0, 5, 'Buscar pela presença de ativos descobertos a partir de 2000: Gemtuzumab Ozogamicin, Midostaurin, CPX-351, Ivosidenib, Venetoclax, Enasidenib, Gilteritinib e Glasdegib.', align='L')
            search_targets_1968(pdf, model, flag, n, 1)
            flag = search_targets_2000(pdf, model, flag, n)
            flag = search_targets_2002(pdf, model, flag, n)
            flag = search_targets_2010(pdf, model, flag, n)
            flag = search_targets_2012(pdf, model, flag, n)
            flag = search_targets_2014(pdf, model, flag, n)
            flag = search_targets_2015(pdf, model, flag, n)
            flag = search_targets_2017(pdf, model, flag, n)
        elif model_year < 2002:
            pdf.multi_cell(0, 5, 'Buscar pela presença de ativos descobertos a partir de 2002: Midostaurin, CPX-351, Ivosidenib, Venetoclax, Enasidenib, Gilteritinib e Glasdegib.', align='L')
            search_targets_1968(pdf, model, flag, n, 1)
            flag = search_targets_2002(pdf, model, flag, n)
            flag = search_targets_2010(pdf, model, flag, n)
            flag = search_targets_2012(pdf, model, flag, n)
            flag = search_targets_2014(pdf, model, flag, n)
            flag = search_targets_2015(pdf, model, flag, n)
            flag = search_targets_2017(pdf, model, flag, n)
        elif model_year < 2010:
            pdf.multi_cell(0, 5, 'Buscar pela presença de ativos descobertos a partir de 2010: CPX-351, Ivosidenib, Venetoclax, Enasidenib, Gilteritinib e Glasdegib.', align='L')
            search_targets_1968(pdf, model, flag, n, 1)
            flag = search_targets_2010(pdf, model, flag, n)
            flag = search_targets_2012(pdf, model, flag, n)
            flag = search_targets_2014(pdf, model, flag, n)
            flag = search_targets_2015(pdf, model, flag, n)
            flag = search_targets_2017(pdf, model, flag, n)
        elif model_year < 2012:
            pdf.multi_cell(0, 5, 'Buscar pela presença de ativos descobertos a partir de 2012: Ivosidenib, Venetoclax, Enasidenib, Gilteritinib e Glasdegib.', align='L')
            search_targets_1968(pdf, model, flag, n, 1)
            flag = search_targets_2012(pdf, model, flag, n)
            flag = search_targets_2014(pdf, model, flag, n)
            flag = search_targets_2015(pdf, model, flag, n)
            flag = search_targets_2017(pdf, model, flag, n)
        elif model_year < 2014:
            pdf.multi_cell(0, 5, 'Buscar pela presença de ativos descobertos a partir de 2014: Venetoclax, Enasidenib, Gilteritinib e Glasdegib.', align='L')
            search_targets_1968(pdf, model, flag, n, 1)
            flag = search_targets_2014(pdf, model, flag, n)
            flag = search_targets_2015(pdf, model, flag, n)
            flag = search_targets_2017(pdf, model, flag, n)
        elif model_year < 2015:
            pdf.multi_cell(0, 5, 'Buscar pela presença de ativos descobertos a partir de 2015: Enasidenib, Gilteritinib e Glasdegib.', align='L')
            search_targets_1968(pdf, model, flag, n, 1)
            flag = search_targets_2015(pdf, model, flag, n)
            flag = search_targets_2017(pdf, model, flag, n)
        elif model_year < 2017:
            pdf.multi_cell(0, 5, 'Buscar pela presença de ativos descobertos a partir de 2017: Glasdegib.', align='L')
            search_targets_1968(pdf, model, flag, n, 1)
            flag = search_targets_2017(pdf, model, flag, n)
        
        if flag == 0:
            pdf.cell(0, 10, 'Sem indício de conhecimento latente, os ativos ainda não descobertos não estão no vocabulário.', ln=2, align='L')
            if model_year < 2000:
                similarities_table(base_compounds[:3], model, pdf)
            elif model_year < 2002:
                similarities_table(base_compounds[:4], model, pdf)
            elif model_year < 2010:
                similarities_table(base_compounds[:5], model, pdf)
            elif model_year < 2012:
                similarities_table(base_compounds[:6], model, pdf)
            elif model_year < 2014:
                similarities_table(base_compounds[:7], model, pdf)
            elif model_year < 2015:
                similarities_table(base_compounds[:8], model, pdf)
            elif model_year < 2017:
                similarities_table(base_compounds[:9], model, pdf)

def similarities_table(words_list, model, pdf):
    pdf.set_font("Arial", size = 10)
    table = [['Palavra']]
    for w in base_compounds:
        if w in model.wv.vocab:
            table[0].append(w)

    th = pdf.font_size
    header_len = len(table[0])

    for w in words_list:
        if w in model.wv.vocab:
            row = [w]
            for y in table[0][1:]:
                if w == y:
                    row.append('---')
                else:
                    similarity = round(float(model.wv.similarity(y, w)), 2)
                    rank = model.wv.rank(y, w)
                    row.append('{}, {}ª'.format(similarity, rank))
            table.append(row)

    for row in table:
        for datum in row:
            if header_len <= 7:
                pdf.cell(25, 1.8*th, str(datum), border=1)
            else:
                pdf.cell(20.5, 1.8*th, str(datum), border=1)
    
        pdf.ln(1.8*th)

    pdf.set_font("Arial", size = 12)

def search_targets_1968(pdf, model, flag, n = 15, passado=0):
    synonyms_1968 = ['cytarabine', 'ara c', 'ara-c', 'arac', 'arabinofuranosyl cytosine', 'arabinoside cytosine', 'cytosine arabinoside', 'arabinosylcytosine', 'aracytidine', 'aracytine',
            'beta ara c', 'beta-ara c', 'beta-ara-c', 'cytarabine hydrochloride', 'cytonal', 'cytosar', 'cytosar u', 'cytosar-u', 'cytosine arabinoside', 'daunorubicin',
            'cerubidine', 'dauno rubidomycine', 'dauno-rubidomycine', 'daunoblastin', 'daunoblastine', 'daunomycin', 'daunorubicin hydrochloride',
            'hydrochloride daunorubicin', 'nsc 82151', 'nsc-82151', 'nsc82151', 'rubidomycin', 'rubomycin'
    ]
    vocab_len = len(model.wv.vocab)

    table = [['Palavra', 'Similaridade Cosseno']]
    if passado == 0:
        for s in synonyms_1968:
            if s.lower() in model.wv.vocab:
                flag = 1
                if s.lower() == 'cytarabine' or s.lower() == 'daunorubicin':
                    pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário.'.format(s), align='L')
                else:
                    pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Cytarabine ou Daunorubicin.'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')

                if len(model.wv.vocab) == 1:
                    pdf.cell(0, 5, 'não é possível visualizar as palavras mais próximas a {}, só há 1 termo no vocabulário.'.format(s), ln=2, align='L')
                elif len(model.wv.vocab) == 0:
                    pdf.multi_cell(0, 5, 'não é possível visualizar as palavras mais próximas a {}. A restrição de vocabulário eliminou todos os termos, reveja e corrija a restrição.'.format(s), align='L')
                else:
                    if n <= len(model.wv.vocab):
                        pdf.cell(0, 5, '{} palavras mais próximas a {}:'.format(n, s), ln=2, align='C')
                        near = model.wv.most_similar(positive=[s], topn = n)
                    else:
                        pdf.cell(0, 5, '{} palavras mais próximas a {}:'.format(len(model.wv.vocab) - 1, s), ln=2, align='C')
                        near = model.wv.most_similar(positive=[s], topn = len(model.wv.vocab))

                    pdf.set_font("Arial", size = 10)
                    for token, prox in near:
                        table.append(tuple((token, round(float(prox), 2))))
                    
                    for row in table:
                        for datum in row:
                            pdf.cell(85, 5, str(datum), border=1)
                    
                        pdf.ln(5)
        pdf.set_font("Arial", size = 12)
        return flag

def search_targets_1978(pdf, model, flag, n = 15):
    synonyms_1978 = ['azacitidine', '5-azacytidine', '320-67-2', 'ladakamycin', 'azacytidine', 'vidaza', 'mylosar', '5-azacitidine', 'azacitidinum', 'azacitidina',
            'azacitidinum', '5-azac', 'nsc-102816', 'c8h12n4o5', 'u-18496', 'nsc102816', '5azac', 'm801h13nru'
    ]
    vocab_len = len(model.wv.vocab)

    lk_words = []
    for s in synonyms_1978:
        if s.lower() in model.wv.vocab:
            flag = 1
            if s.lower() == 'azacitidine':
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário.'.format(s), align='L')
            else:
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônim de Azacitidine.'.format(s), align='L')
            lk_words.append(s.lower())

    if len(lk_words) > 0:
        similarities_table(lk_words, model, pdf)

    table = [['Palavra', 'Similaridade Cosseno']]
    for w in lk_words:
        pdf.cell(0, 4, ' ', ln=2, align='L')

        if vocab_len == 1:
            pdf.cell(0, 5, 'não é possível visualizar as palavras mais próximas a {}, só há 1 termo no vocabulário.'.format(s), ln=2, align='L')
        else:
            if n <= vocab_len:
                pdf.cell(0, 5, '{} palavras mais próximas a {}:'.format(n, w), ln=2, align='C')
                near = model.wv.most_similar(positive=[w], topn = n)
            else:
                pdf.cell(0, 5, '{} palavras mais próximas a {}:'.format(vocab_len - 1, w), ln=2, align='C')
                near = model.wv.most_similar(positive=[w], topn = vocab_len)

        pdf.set_font("Arial", size = 10)
        for token, prox in near:
            table.append(tuple((token, round(float(prox), 2))))
        
        for row in table:
            for datum in row:
                pdf.cell(85, 5, str(datum), border=1)
        
            pdf.ln(5)
    pdf.set_font("Arial", size = 12)
    return flag

def search_targets_2000(pdf, model, flag, n = 15):
    synonyms_2000 = ['gemtuzumab ozogamicina', 'gemtuzumab ozogamicine', 'cma-676', 'mylotarg', 'cma676', 'cma 676', 'gemtuzumab ozogamicin']
    lk_words = []
    vocab_len = len(model.wv.vocab)

    for s in synonyms_2000:
        if s.lower() in model.wv.vocab:
            flag = 1
            if s.lower() == 'gemtuzumab ozogamicin':
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário.'.format(s), align='L')
            else:
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Gemtuzumab Ozogamicin.'.format(s), align='L')
            lk_words.append(s.lower())

    if len(lk_words) > 0:
        similarities_table(lk_words, model, pdf)

    table = [['Palavras', 'Similaridade Cosseno']]
    for w in lk_words:
        pdf.cell(0, 4, ' ', ln=2, align='L')

        if vocab_len == 1:
            pdf.cell(0, 5, 'não é possível visualizar as palavras mais próximas a {}, só há 1 termo no vocabulário.'.format(s), ln=2, align='L')
        else:
            if n <= vocab_len:
                pdf.cell(0, 5, '{} palavras mais próximas a {}:'.format(n, w), ln=2, align='C')
                near = model.wv.most_similar(positive=[w], topn = n)
            else:
                pdf.cell(0, 5, '{} palavras mais próximas a {}:'.format(vocab_len - 1, w), ln=2, align='C')
                near = model.wv.most_similar(positive=[w], topn = vocab_len)

        pdf.set_font("Arial", size = 10)
        for token, prox in near:
            table.append(tuple((token, round(float(prox), 2))))
        
        for row in table:
            for datum in row:
                pdf.cell(85, 5, str(datum), border=1)
        
            pdf.ln(5)
    pdf.set_font("Arial", size = 12)
    return flag

def search_targets_2002(pdf, model, flag, n = 15):
    synonyms_2002 = ['midostaurin', 'pkc412', '120685-11-2', 'benzoylstaurosporine', 'cgp 41251', 'pkc-412', 'pkc 412', '4-n-benzoylstaurosporine', 'cgp-41251',
            'rydapt', 'n-benzoylstaurosporine', 'id912s5von', 'chembl608533', 'chebi:63452', 'cgp 41 251'
    ]
    lk_words = []
    vocab_len = len(model.wv.vocab)

    for s in synonyms_2002:
        if s.lower() in model.wv.vocab:
            flag = 1
            if s.lower() == 'midostaurin':
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário.'.format(s), align='L')
            else:
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Midostaurin.'.format(s), align='L')
            lk_words.append(s.lower())

    if len(lk_words) > 0:
        similarities_table(lk_words, model, pdf)

    table = [['Palavra', 'Similaridade Cosseno']]
    for w in lk_words:
        pdf.cell(0, 4, ' ', ln=2, align='L')
        if vocab_len == 1:
            pdf.cell(0, 5, 'não é possível visualizar as palavras mais próximas a {}, só há 1 termo no vocabulário.'.format(s), ln=2, align='L')
        else:
            if n <= vocab_len:
                pdf.cell(0, 5, '{} palavras mais próximas a {}:'.format(n, w), ln=2, align='C')
                near = model.wv.most_similar(positive=[w], topn = n)
            else:
                pdf.cell(0, 5, '{} palavras mais próximas a {}:'.format(vocab_len - 1, w), ln=2, align='C')
                near = model.wv.most_similar(positive=[w], topn = vocab_len)

        pdf.set_font("Arial", size = 10)
        for token, prox in near:
            table.append(tuple((token, round(float(prox), 2))))
        
        for row in table:
            for datum in row:
                pdf.cell(85, 5, str(datum), border=1)
        
            pdf.ln(5)
    pdf.set_font("Arial", size = 12)
    return flag

def search_targets_2010(pdf, model, flag, n = 15):
    synonyms_2010 = ['cpx-351', 'vyxeos', 'vyxeos liposomal', 'cpx 351', 'cpx351']
    lk_words = []
    vocab_len = len(model.wv.vocab)

    for s in synonyms_2010:
        if s.lower() in model.wv.vocab:
            flag = 1
            if s.lower() == 'cpx-351':
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário.'.format(s), align='L')
            else:
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônim de CPX-351.'.format(s), align='L')
            lk_words.append(s.lower())

    if len(lk_words) > 0:
        similarities_table(lk_words, model, pdf)


    table = [['Palavra', 'Similaridade Cosseno']]
    for w in lk_words:
        pdf.cell(0, 4, ' ', ln=2, align='L')
        if vocab_len == 1:
            pdf.cell(0, 5, 'não é possível visualizar as palavras mais próximas a {}, só há 1 termo no vocabulário.'.format(s), ln=2, align='L')
        else:
            if n <= vocab_len:
                pdf.cell(0, 5, '{} palavras mais próximas a {}:'.format(n, w), ln=2, align='C')
                near = model.wv.most_similar(positive=[w], topn = n)
            else:
                pdf.cell(0, 5, '{} palavras mais próximas a {}:'.format(vocab_len - 1, w), ln=2, align='C')
                near = model.wv.most_similar(positive=[w], topn = vocab_len)
        
        pdf.set_font("Arial", size = 10)
        for token, prox in near:
            table.append(tuple((token, round(float(prox), 2))))
        
        for row in table:
            for datum in row:
                pdf.cell(85, 5, str(datum), border=1)
        
            pdf.ln(5)
    pdf.set_font("Arial", size = 12)
    return flag

def search_targets_2012(pdf, model, flag, n = 15):
    synonyms_2012 = ['ivosidenib', '1448347-49-6', 'ag-120', 'ag120', 'tibsovo', 'UNII-Q2PCN8MAM6', 'q2pcn8mam6', 'ivosidenibum', '1448346-63-1', '1448347-49-6', 'gtpl9217',
            'chembl3989958', 'schembl15122512', 'ex-a992', 'chebi:145430', 'bdbm363689', 'amy38924', 'mfcd29036964', 'nsc789102', 'ccg-270141', 'cs-5122'
    ]
    lk_words = []
    vocab_len = len(model.wv.vocab)

    for s in synonyms_2012:
        if s.lower() in model.wv.vocab:
            flag = 1
            if s.lower() == 'ivosidenib':
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário.'.format(s), align='L')
            else:
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônim de Ivosidenib.'.format(s), align='L')
            lk_words.append(s.lower())

    if len(lk_words) > 0:
        similarities_table(lk_words, model, pdf)

    table = [['Palavra', 'Similaridade Cosseno']]
    for w in lk_words:
        pdf.cell(0, 4, ' ', ln=2, align='L')
        if vocab_len == 1:
            pdf.cell(0, 5, 'não é possível visualizar as palavras mais próximas a {}, só há 1 termo no vocabulário.'.format(s), ln=2, align='L')
        else:
            if n <= vocab_len:
                pdf.cell(0, 5, '{} palavras mais próximas a {}:'.format(n, w), ln=2, align='C')
                near = model.wv.most_similar(positive=[w], topn = n)
            else:
                pdf.cell(0, 5, '{} palavras mais próximas a {}:'.format(vocab_len - 1, w), ln=2, align='C')
                near = model.wv.most_similar(positive=[w], topn = vocab_len)
        
        pdf.set_font("Arial", size = 10)
        for token, prox in near:
            table.append(tuple((token, round(float(prox), 2))))
        
        for row in table:
            for datum in row:
                pdf.cell(85, 5, str(datum), border=1)
        
            pdf.ln(5)
    pdf.set_font("Arial", size = 12)
    return flag

def search_targets_2014(pdf, model, flag, n = 15):
    synonyms_2014 = ['venetoclax', 'abt-199', '1257044-40-8', 'venclexta', 'gdc-0199', 'abt199', 'abt 199', 'UNII-N54AIC43PW', 'gdc 0199', 'rg7601', 'rg-7601', 'n54aic43pw',
            'venclyxto', 'bdbm189459'
    ]
    lk_words = []
    vocab_len = len(model.wv.vocab)

    for s in synonyms_2014:
        if s.lower() in model.wv.vocab:
            flag = 1
            if s.lower() == 'venetoclax':
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário.'.format(s), align='L')
            else:
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônim de Venetoclax.'.format(s), align='L')
            lk_words.append(s.lower())

    if len(lk_words) > 0:
        similarities_table(lk_words, model, pdf)

    table = [['Palavra', 'Similaridade Cosseno']]
    for w in lk_words:
        pdf.cell(0, 4, ' ', ln=2, align='L')
        if vocab_len == 1:
            pdf.cell(0, 5, 'não é possível visualizar as palavras mais próximas a {}, só há 1 termo no vocabulário.'.format(s), ln=2, align='L')
        else:
            if n <= vocab_len:
                pdf.cell(0, 5, '{} palavras mais próximas a {}:'.format(n, w), ln=2, align='C')
                near = model.wv.most_similar(positive=[w], topn = n)
            else:
                pdf.cell(0, 5, '{} palavras mais próximas a {}:'.format(vocab_len - 1, w), ln=2, align='C')
                near = model.wv.most_similar(positive=[w], topn = vocab_len)
        
        pdf.set_font("Arial", size = 10)
        for token, prox in near:
            table.append(tuple((token, round(float(prox), 2))))
        
        for row in table:
            for datum in row:
                pdf.cell(85, 5, str(datum), border=1)
        
            pdf.ln(5)
    pdf.set_font("Arial", size = 12)
    return flag

def search_targets_2015(pdf, model, flag, n = 15):
    synonyms_2015 = ['enasidenib', '1446502-11-9', 'ag-221', 'idhifa', 'unii-3t1ss4e7ag', 'ag 221', 'cc-90007', '3t1ss4e7ag', 'enasidenibum', 'ag221', 'gtpl8960',
            'chembl3989908', 'schembl15102202', 'ex-A654', 'chebi:145374', 'hms3873d03', 'amy38698', 'bcp16041', 'bdbm50503251', 'mfcd29472245', 'nsc788120',
            's8205', 'akos026750439', 'zinc222731806', 'ccg-269476', 'cs-5017', 'db13874', 'nsc-788120', 'sb19193', 'ac-31318', 'as-75164', 'hy-18690', 'ft-0700204',
            'd10901', 'j-690181', 'q27077182', 'gilteritinib', '1254053-43-4', 'asp2215', 'asp-2215', 'xospata', 'asp 2215', '66d92mgc8m', 'gilteritinib hcl',
            'gilteritinibum', 'c6f', 'schembl282229', 'gtpl8708', 'chembl3301622', 'chebi:145372', 'bdbm144315', 'c29h44n8o3', 'bcp28756', 'ex-a2775', '3694ah',
            'mfcd28144685', 'nsc787846', 'nsc787854', 'nsc788454', 'nsc800106', 's7754', 'ccg-270016', 'cs-3885', 'db12141', 'nsc-787846', 'nsc-787854', 'nsc-788454',
            'nsc-800106', 'sb16988', 'ncgc00481652-01', 'ncgc00481652-02', 'ac-29030', 'as-35199', 'hy-12432', 'qc-11768', 'db-108103', 'a14411', 'd10709', 'q27077802'
    ]
    lk_words = []
    vocab_len = len(model.wv.vocab)

    for s in synonyms_2015:
        if s.lower() in model.wv.vocab:
            flag = 1
            if s.lower() == 'enasidenib' or s.lower() == 'gilteritnib':
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário.'.format(s), align='L')
            else:
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Enasidenib ou Gilteritinib.'.format(s), align='L')
            lk_words.append(s.lower())

    if len(lk_words) > 0:
        similarities_table(lk_words, model, pdf)

    table = [['Palavra', 'Similaridade Cosseno']]
    for w in lk_words:
        pdf.cell(0, 4, ' ', ln=2, align='L')
        if vocab_len == 1:
            pdf.cell(0, 5, 'não é possível visualizar as palavras mais próximas a {}, só há 1 termo no vocabulário.'.format(s), ln=2, align='L')
        else:
            if n <= vocab_len:
                pdf.cell(0, 5, '{} palavras mais próximas a {}:'.format(n, w), ln=2, align='C')
                near = model.wv.most_similar(positive=[w], topn = n)
            else:
                pdf.cell(0, 5, '{} palavras mais próximas a {}:'.format(vocab_len - 1, w), ln=2, align='C')
                near = model.wv.most_similar(positive=[w], topn = vocab_len)
        
        pdf.set_font("Arial", size = 10)
        for token, prox in near:
            table.append(tuple((token, round(float(prox), 2))))
        
        for row in table:
            for datum in row:
                pdf.cell(85, 5, str(datum), border=1)
        
            pdf.ln(5)
    pdf.set_font("Arial", size = 12)
    return flag

def search_targets_2017(pdf, model, flag, n = 15):
    synonyms_2017 = ['glasdegib', '1095173-27-5', 'pf-04449913', 'pf 04449913', 'daurismo', 'k673dmo5h9', 'chembl2043437', 'c21h22n6o', 'pf-913', 'glasdegibum',
            'gtpl8201', 'schembl2068480', 'ex-a858', 'chebi:145428', 'amy38164', 'vtb17327', '2640ah', 'bdbm50385635', 'mfcd25976839', 'nsc775772', 
            'zinc68251434', '1095173-27-5', 'ccg-268350', 'db11978', 'nsc-775772', 'sb16679', 'ncgc00378600-02', 'bs-14357', 'hy-16391', 'qc-11459', 's7160', 'd10636',
            'z-3230', 'j-690029', 'q27077810'
    ]
    lk_words = []
    vocab_len = len(model.wv.vocab)

    for s in synonyms_2017:
        if s.lower() in model.wv.vocab:
            flag = 1
            if s.lower() == 'glasdegib':
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário.'.format(s), align='L')
            else:
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Glasdegib.'.format(s), align='L')
            lk_words.append(s.lower())

    if len(lk_words) > 0:
        similarities_table(lk_words, model, pdf)

    table = [['Palavra', 'Similaridade Cosseno']]
    for w in lk_words:
        pdf.cell(0, 4, ' ', ln=2, align='L')
        if vocab_len == 1:
            pdf.cell(0, 5, 'não é possível visualizar as palavras mais próximas a {}, só há 1 termo no vocabulário.'.format(s), ln=2, align='L')
        else:
            if n <= vocab_len:
                pdf.cell(0, 5, '{} palavras mais próximas a {}:'.format(n, w), ln=2, align='C')
                near = model.wv.most_similar(positive=[w], topn = n)
            else:
                pdf.cell(0, 5, '{} palavras mais próximas a {}:'.format(vocab_len - 1, w), ln=2, align='C')
                near = model.wv.most_similar(positive=[w], topn = vocab_len)
        
        pdf.set_font("Arial", size = 10)
        for token, prox in near:
            table.append(tuple((token, round(float(prox), 2))))
        
        for row in table:
            for datum in row:
                pdf.cell(85, 5, str(datum), border=1)
        
            pdf.ln(5)
    pdf.set_font("Arial", size = 12)
    return flag

try:
    command_arg = int(sys.argv[1])
    os.mkdir("./relatorios/")
except IndexError:
    command_arg = 0
except ValueError:
    command_arg = sys.argv[1]
except OSError:
    pass

filenames = sorted([str(x) for x in Path('./word2vec/').glob('*.model')])
n = 15
i = 1
pdf, data_hora = create_pdf_header(command_arg, filenames, n)

for f in filenames:
    model = pickle.load(open(f, 'rb'))
    model.init_sims()
    if isinstance(command_arg, int):
        common_words = get_most_common(command_arg)
        wv_restrict_w2v(model, set(common_words))
    elif command_arg == 'nci_cancer_drugs':
        specific_domain = []
        domains_table = pd.read_csv('https://docs.google.com/spreadsheets/d/' + 
                        '1SgYG4gZuL3grEHHAZt49dAUw_jFAc4LADajFeGAf2-w' +
                        '/export?gid=0&format=csv',
                        )
        specific_domain = domains_table['name'].tolist()
        specific_domain = list(dict.fromkeys(specific_domain))
        wv_restrict_w2v(model, set(specific_domain), True)
    elif command_arg == 'fda_drugs':
        specific_domain = []
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

    print(f)            
    model_year = int(f[33:37])
    investigate_models(pdf, model, model_year, i, n)
    i += 1

pdf.output("relatorio_{}.{}.{}_{}-{}_{}.pdf".format(data_hora[0:2], data_hora[3:5], data_hora[6:11], data_hora[11:13], data_hora[14:16], command_arg))
