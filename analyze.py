import os, gensim, sys, pickle
from pathlib import Path
from gensim.models import Word2Vec
from os import listdir
from os.path import isfile, join
from prettytable import PrettyTable
from fpdf import FPDF
from datetime import datetime, timezone, timedelta
from get_n_common_words_english import get_most_common
from streamlit_app import restrict_w2v, wv_restrict_w2v

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

synonyms_1968 = ['cytarabine', 'ara c', 'ara-c', 'arac', 'arabinofuranosyl cytosine', 'arabinoside cytosine', 'cytosine arabinoside', 'arabinosylcytosine', 'aracytidine', 'aracytine',
        'beta ara c', 'beta-ara c', 'beta-ara-c', 'cytarabine hydrochloride', 'cytonal', 'cytosar', 'cytosar u', 'cytosar-u', 'cytosine arabinoside', 'daunorubicin',
        'cerubidine', 'dauno rubidomycine', 'dauno-rubidomycine', 'daunoblastin', 'daunoblastine', 'daunomycin', 'daunorubicin hydrochloride',
        'hydrochloride daunorubicin', 'nsc 82151', 'nsc-82151', 'nsc82151', 'rubidomycin', 'rubomycin'
]

synonyms_1978 = ['azacitidine', '5-azacytidine', '320-67-2', 'ladakamycin', 'azacytidine', 'vidaza', 'mylosar', '5-azacitidine', 'azacitidinum', 'azacitidina',
        'azacitidinum', '5-azac', 'nsc-102816', 'c8h12n4o5', 'u-18496', 'nsc102816', '5azac', 'm801h13nru'
]

synonyms_2000 = ['gemtuzumab ozogamicina', 'cma-676', 'gemtuzumab-ozogamicin', 'mylotarg', 'cma676', 'cma 676', 'gemtuzumab ozogamicin']

synonyms_2002 = ['midostaurin', 'pkc412', '120685-11-2', 'benzoylstaurosporine', 'cgp 41251', 'pkc-412', 'pkc 412', '4-n-benzoylstaurosporine', 'cgp-41251',
        'rydapt', 'n-benzoylstaurosporine', 'id912s5von', 'chembl608533', 'chebi:63452', 'cgp 41 251'
]

synonyms_2010 = ['cpx-351', 'vyxeos', 'vyxeos liposomal', 'cpx 351']

synonyms_2012 = ['ivosidenib', '1448347-49-6', 'ag-120', 'ag120', 'tibsovo', 'UNII-Q2PCN8MAM6', 'q2pcn8mam6', 'ivosidenibum', '1448346-63-1', '1448347-49-6', 'gtpl9217',
        'chembl3989958', 'schembl15122512', 'ex-a992', 'chebi:145430', 'bdbm363689', 'amy38924', 'mfcd29036964', 'nsc789102', 'ccg-270141', 'cs-5122'
]

synonyms_2014 = ['venetoclax', 'abt-199', '1257044-40-8', 'venclexta', 'gdc-0199', 'abt199', 'abt 199', 'UNII-N54AIC43PW', 'gdc 0199', 'rg7601', 'rg-7601', 'n54aic43pw',
        'venclyxto', 'bdbm189459'
]

synonyms_2015 = ['enasidenib', '1446502-11-9', 'ag-221', 'idhifa', 'unii-3t1ss4e7ag', 'ag 221', 'cc-90007', '3t1ss4e7ag', 'enasidenibum', 'ag221', 'gtpl8960',
        'chembl3989908', 'schembl15102202', 'ex-A654', 'chebi:145374', 'hms3873d03', 'amy38698', 'bcp16041', 'bdbm50503251', 'mfcd29472245', 'nsc788120',
        's8205', 'akos026750439', 'zinc222731806', 'ccg-269476', 'cs-5017', 'db13874', 'nsc-788120', 'sb19193', 'ac-31318', 'as-75164', 'hy-18690', 'ft-0700204',
        'd10901', 'j-690181', 'q27077182', 'gilteritinib', '1254053-43-4', 'asp2215', 'asp-2215', 'xospata', 'asp 2215', '66d92mgc8m', 'gilteritinib hcl',
        'gilteritinibum', 'c6f', 'schembl282229', 'gtpl8708', 'chembl3301622', 'chebi:145372', 'bdbm144315', 'c29h44n8o3', 'bcp28756', 'ex-a2775', '3694ah',
        'mfcd28144685', 'nsc787846', 'nsc787854', 'nsc788454', 'nsc800106', 's7754', 'ccg-270016', 'cs-3885', 'db12141', 'nsc-787846', 'nsc-787854', 'nsc-788454',
        'nsc-800106', 'sb16988', 'ncgc00481652-01', 'ncgc00481652-02', 'ac-29030', 'as-35199', 'hy-12432', 'qc-11768', 'db-108103', 'a14411', 'd10709', 'q27077802'
]

synonyms_2017 = ['glasdegib', '1095173-27-5', 'pf-04449913', 'pf 04449913', 'daurismo', 'k673dmo5h9', 'chembl2043437', 'c21h22n6o', 'pf-913', 'glasdegibum',
        'gtpl8201', 'schembl2068480', 'ex-a858', 'chebi:145428', 'amy38164', 'vtb17327', '2640ah', 'bdbm50385635', 'mfcd25976839', 'nsc775772', 
        'zinc68251434', '1095173-27-5', 'ccg-268350', 'db11978', 'nsc-775772', 'sb16679', 'ncgc00378600-02', 'bs-14357', 'hy-16391', 'qc-11459', 's7160', 'd10636',
        'z-3230', 'j-690029', 'q27077810'
]

try:
	os.mkdir("./relatorios/")
except OSError as error:
	pass

diferenca = timedelta(hours=-3)
data_hora_atuais = datetime.now()
fuso_horario = timezone(diferenca)
data_hora_sao_paulo = data_hora_atuais.astimezone(fuso_horario)
data_hora_sao_paulo_em_texto = data_hora_sao_paulo.strftime('%d/%m/%Y %H:%M')

# a variable pdf
pdf = FPDF()   
# Add a page
pdf.add_page()
pdf.set_font("Arial", size = 12)

pdf.cell(200, 10, txt = "WE4LKD - AML",  ln = 1, align = 'C')
  
# add another cell
pdf.cell(200, 10, txt = "Relatório de análise modelos Word2Vec", ln = 2, align = 'C')

pdf.cell(200, 5, txt = data_hora_sao_paulo_em_texto, ln = 2, align = 'C')

#sys.stdout = open("./relatorios/latent_knowledge.txt", "w", encoding='utf-8')
files = sorted([str(x) for x in Path('./word2vec/').glob('*.model')])
pdf.cell(200, 5, txt = 'Arquivos analisados:', ln = 2, align = 'L')
for e in files:
    size = get_file_size(e)
    pdf.cell(200, 5, txt=e+'    '+convert_bytes(size, "KB"), ln=2, align='L')

n = 15

#n first common words
try:
    common_number = int(sys.argv[1])
except:
    common_number = 0

pdf.cell(0, 10, ' ', ln=2, align='L')
pdf.cell(200, 5, txt = 'Procedimentos para análise:', ln = 2, align = 'L')
pdf.multi_cell(0, 5, txt = '1. em todos os modelos, foram impressas as {} palavras mais próximas a "cytarabine" e "daunorubicin";'.format(n), align = 'L')
pdf.multi_cell(0, 5, txt = '2. em cada modelo, foram buscados sinônimos de ativos que ainda não tinham sido descobertos na época, de acordo com a linha do tempo base.', align = 'L')

flag = 0
table = [['Token', 'Similaridade Cosseno']]
for f in files:
    model = pickle.load(open(f, 'rb'))
    model.init_sims()
    if common_number != 0:
        common_words = get_most_common(int(common_number))
        wv_restrict_w2v(model, set(common_words))
    model_year = int(f[33:37])
    pdf.cell(0, 10, ' ', ln=2, align='L')
    pdf.cell(0, 5, 'Modelo até o ano de {} (inclusive)'.format(model_year), ln=2, align='L')
    flag = 0
    if model_year < 1968:
        pdf.multi_cell(0, 5, 'buscar pela presenca de ativos descobertos a partir de 1968: Cytarabine, Daunorubicin, Azacitidine, Gemtuzumab Ozogamicine, Midostaurine, CPX-351, Ivosidenib, Venetoclax, Enasidenib, Gilteritinib e Glasdegib', align='L')
        for s in synonyms_1968:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Cytarabine ou Daunorubicin'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = []
                table = [['Token', 'Similaridade Cosseno']]
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_1978:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Azacitidine'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2000:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Gemtuzumab Ozogamicine'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2002:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Midostaurine'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2010:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de CPX-351'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2012:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Ivosidenib'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2014:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Venetoclax'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2015:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Enasidenib ou Gilteritinib'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2017:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Glasdegib'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        if flag == 1:
            pdf.cell(0, 5, '----', ln=2, align='L')
        elif flag == 0:
            pdf.cell(0, 5, 'sem indício de conhecimento latente', ln=2, align='L')
    elif model_year < 1978:
        pdf.multi_cell(0, 5, 'buscar pela presenca de ativos descobertos a partir de 1978: Azacitidine, Gemtuzumab Ozogamicine, Midostaurine, CPX-351, Ivosidenib, Venetoclax, Enasidenib, Gilteritinib e Glasdegib', align='L')
        for s in synonyms_1968:
            if s.lower() in model.wv.vocab:
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = []
                table = [['Token', 'Similaridade Cosseno']]
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_1978:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Azacitidine'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2000:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Gemtuzumab Ozogamicine'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2002:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Midostaurine'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2010:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de CPX-351'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2012:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Ivosidenib'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2014:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Venetoclax'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2015:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Enasidenib ou Gilteritinib'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2017:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Glasdegib'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        if flag == 1:
            pdf.cell(0, 5, '----', ln=2, align='L')
        elif flag == 0:
            pdf.cell(0, 5, 'sem indício de conhecimento latente', ln=2, align='L')
    elif model_year < 2000:
        pdf.multi_cell(0, 5, 'buscar pela presenca de ativos descobertos a partir de 2000: Gemtuzumab Ozogamicine, Midostaurine, CPX-351, Ivosidenib, Venetoclax, Enasidenib, Gilteritinib e Glasdegib', align='L')
        for s in synonyms_1968:
            if s.lower() in model.wv.vocab:
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = []
                table = [['Token', 'Similaridade Cosseno']]
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2000:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Gemtuzumab Ozogamicine'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2002:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Midostaurine'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2010:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de CPX-351'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2012:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Ivosidenib'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2014:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Venetoclax'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2015:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Enasidenib ou Gilteritinib'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2017:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Glasdegib'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        if flag == 1:
            pdf.cell(0, 5, '----', ln=2, align='L')
        elif flag == 0:
            pdf.cell(0, 5, 'sem indício de conhecimento latente', ln=2, align='L')
    elif model_year < 2002:
        pdf.multi_cell(0, 5, 'buscar pela presenca de ativos descobertos a partir de 2002: Midostaurine, CPX-351, Ivosidenib, Venetoclax, Enasidenib, Gilteritinib e Glasdegib', align='L')
        for s in synonyms_1968:
            if s.lower() in model.wv.vocab:
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = []
                table = [['Token', 'Similaridade Cosseno']]
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2002:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Midostaurine'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2010:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de CPX-351'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2012:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Ivosidenib'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2014:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Venetoclax'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2015:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Enasidenib ou Gilteritinib'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2017:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Glasdegib'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        if flag == 1:
            pdf.cell(0, 5, '----', ln=2, align='L')
        elif flag == 0:
            pdf.cell(0, 5, 'sem indício de conhecimento latente', ln=2, align='L')
    elif model_year < 2010:
        pdf.multi_cell(0, 5, 'buscar pela presenca de ativos descobertos a partir de 2010: CPX-351, Ivosidenib, Venetoclax, Enasidenib, Gilteritinib e Glasdegib', align='L')
        for s in synonyms_1968:
            if s.lower() in model.wv.vocab:
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = []
                table = [['Token', 'Similaridade Cosseno']]
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2010:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de CPX-351'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2012:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Ivosidenib'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2014:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Venetoclax'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2015:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Enasidenib ou Gilteritinib'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2017:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Glasdegib'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        if flag == 1:
            pdf.cell(0, 5, '----', ln=2, align='L')
        elif flag == 0:
            pdf.cell(0, 5, 'sem indício de conhecimento latente', ln=2, align='L')
    elif model_year < 2012:
        pdf.multi_cell(0, 5, 'buscar pela presenca de ativos descobertos a partir de 2012: Ivosidenib, Venetoclax, Enasidenib, Gilteritinib e Glasdegib', align='L')
        for s in synonyms_1968:
            if s.lower() in model.wv.vocab:
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = []
                table = [['Token', 'Similaridade Cosseno']]
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2012:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Ivosidenib'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2014:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Venetoclax'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2015:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Enasidenib ou Gilteritinib'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2017:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Glasdegib'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        if flag == 1:
            pdf.cell(0, 5, '----', ln=2, align='L')
        elif flag == 0:
            pdf.cell(0, 5, 'sem indício de conhecimento latente', ln=2, align='L')
    elif model_year < 2014:
        pdf.multi_cell(0, 5, 'buscar pela presenca de ativos descobertos a partir de 2014: Venetoclax, Enasidenib, Gilteritinib e Glasdegib', align='L')
        for s in synonyms_1968:
            if s.lower() in model.wv.vocab:
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = []
                table = [['Token', 'Similaridade Cosseno']]
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2014:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Venetoclax'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2015:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Enasidenib ou Gilteritinib'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2017:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Glasdegib'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        if flag == 1:
            pdf.cell(0, 5, '----', ln=2, align='L')
        elif flag == 0:
            pdf.cell(0, 5, 'sem indício de conhecimento latente', ln=2, align='L')
    elif model_year < 2015:
        pdf.multi_cell(0, 5, 'buscar pela presenca de ativos descobertos a partir de 2015: Enasidenib, Gilteritinib e Glasdegib', align='L')
        for s in synonyms_1968:
            if s.lower() in model.wv.vocab:
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = []
                table = [['Token', 'Similaridade Cosseno']]
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2015:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Enasidenib ou Gilteritinib'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2017:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Glasdegib'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        if flag == 1:
            pdf.cell(0, 5, '----', ln=2, align='L')
        elif flag == 0:
            pdf.cell(0, 5, 'sem indício de conhecimento latente', ln=2, align='L')
    elif model_year < 2017:
        pdf.multi_cell(0, 5, 'buscar pela presenca de ativos descobertos a partir de 2017: Glasdegib', align='L')
        for s in synonyms_1968:
            if s.lower() in model.wv.vocab:
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = []
                table = [['Token', 'Similaridade Cosseno']]
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        for s in synonyms_2017:
            if s.lower() in model.wv.vocab:
                flag = 1
                pdf.multi_cell(0, 5, '\nIndício de conhecimento latente: {} está no vocabulário, é sinônimo de Glasdegib'.format(s), align='L')
                pdf.cell(0, 4, ' ', ln=2, align='L')
                pdf.cell(0, 5, '{} palavras próximas a {}:'.format(n, s), ln=2, align='C')
                near = model.wv.most_similar(positive=[s], topn = n)
                for token, prox in near:
                    table.append(tuple((token, round(float(prox), 2))))
                
                for row in table:
                    for datum in row:
                        pdf.cell(85, 5, str(datum), border=1)
                
                    pdf.ln(5)
        table = [['Token', 'Similaridade Cosseno']]

        if flag == 1:
            pdf.cell(0, 5, '----', ln=2, align='L')
        elif flag == 0:
            pdf.cell(0, 5, 'sem indício de conhecimento latente', ln=2, align='L')
    else:
        pdf.cell(0, 10, 'sem indício de conhecimento latente', ln=2, align='L')
pdf.output("./relatorios/relatorio_{}.{}.{}_{}-{}_{}.pdf".format(data_hora_sao_paulo_em_texto[0:2], data_hora_sao_paulo_em_texto[3:5], data_hora_sao_paulo_em_texto[6:11], data_hora_sao_paulo_em_texto[11:13], data_hora_sao_paulo_em_texto[14:16], common_number))