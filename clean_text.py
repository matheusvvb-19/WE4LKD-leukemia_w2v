import nltk, sys, string, re
from os import listdir
from os.path import join
from pathlib import Path
from nltk.tokenize import PunktSentenceTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet, stopwords

nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

lemmatizer = nltk.wordnet.WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

fix_typos_dict = {'remarkablely': 'remarkably',
                      'leukaemia': 'leukemia',
                      'leukaemias': 'leukemias',
                      'efficiacy': 'efficiency',
                      'carboxylpheny': 'carboxyphenyl',
                      'dimethylthiazole': 'dimethylthiazol',
                      'anthracyclines': 'anthracycline',
}

units_and_symbols = ['μ', 'Ω', 'ω', 'μm', 'mol', '°c', '≥', '≤', '±', 
                        'day', 'month', 'year', '·', 'week', 'μ', 'days',
                        'weeks', 'years', 'mg', 'mm', '®', 'µl'
]

synonyms_cytarabine = ['ara c', 'ara-c', 'arac', 'cytosar', 'cytosar-u',
                    'arabinofuranosyl cytosine', 'arabinoside cytosine', 
                    'cytosine arabinoside', 'beta-arac',
                    'arabinosylcytosine', 'aracytidine', 'aracytine',
                    'beta ara c', 'beta-ara c', 'beta-ara-c', 'beta arac',
                    'beta ara-c', 'arabinofuranosyl', '1-beta-d-cytarabinecytosine', 'cytosine-arabinoside',
                    '1-beta-d-cytarabine cytosine', 'hdcytarabine'
]

synonyms_daunorubicin = ['cerubidine', 'dauno rubidomycine', 'dauno-rubidomycine', 'daunoblastin',
                    'daunoblastine', 'daunomycin',
                    'daunorubicin', 'daunorubicin hydrochloride', 'hydrochloride daunorubicin',
                    'nsc 82151', 'nsc-82151', 'nsc82151', 'rubidomycin',
                    'rubomycin'
]

synonyms_azacitidine = ['5-azacytidine', '5 azacytidine', '320-67-2', 'ladakamycin',
                    'azacytidine', 'vidaza',
                    'mylosar', 'azacitidinum', 'azacitidina',
                    'azacitidinum', '5-azac', 'nsc-102816', 'nsc 102816', 'nsc102816', 'c8h12n4o5',
                    'u-18496', '5azac', 'm801h13nru'
]

synonyms_gemtuzumab_ozogamicin = ['cma-676', 'gemtuzumab-ozogamicine', 'mylotarg', 'cma676',
                    'cma 676',
]

synonyms_midostaurin = ['pkc412', '120685-11-2', 'benzoylstaurosporine',
                    'cgp 41251', 'pkc-412', 'pkc 412', '4-n-benzoylstaurosporine', 'cgp-41251',
                    'rydapt', 'n-benzoylstaurosporine', 'id912s5von', 'chembl608533',
                    'chebi:63452', 'cgp 41 251',
]

synonyms_cpx_351 = ['vyxeos', 'vyxeos liposomal', 'cpx 351', 'cpx351']

synonyms_ivosidenib = ['1448347-49-6', 'ag-120', 'ag120', 'tibsovo',
                    'unii-q2pcn8mam6', 'q2pcn8mam6', 'ivosidenibum',
                    '1448346-63-1', '1448347-49-6', 'gtpl9217', 'chembl3989958', 'schembl15122512',
                    'ex-a992', 'chebi:145430', 'bdbm363689', 'amy38924',
                    'mfcd29036964', 'nsc789102', 'ccg-270141', 'cs-5122', 'ccg270141',
                    'ccg 270141', 'nsc 789102', 'nsc-789102'
]

synonyms_venetoclax = ['abt-199', 'abt199', 'abt 199', '1257044-40-8', 'venclexta', 'gdc-0199', 'gdc0199',
                    'gdc 0199','unii-n54aic43pw','rg7601','rg-7601','rg 7601','n54aic43pw','venclyxto','bdbm189459'
]

synonyms_enasidenib = ['1446502-11-9', 'ag-221', 'idhifa', 'unii-3t1ss4e7ag', 'ag 221', 'cc-90007', '3t1ss4e7ag', 'enasidenibum', 'ag221', 'gtpl8960',
    'chembl3989908', 'schembl15102202', 'ex-A654', 'chebi:145374', 'hms3873d03', 'amy38698', 'bcp16041', 'bdbm50503251', 'mfcd29472245', 'nsc788120',
    's8205', 'akos026750439', 'zinc222731806', 'ccg-269476', 'cs-5017', 'db13874', 'nsc-788120', 'sb19193', 'ac-31318', 'as-75164', 'hy-18690', 'ft-0700204',
    'd10901', 'j-690181', 'q27077182'
]

synonyms_gilteritinib = ['gilteritinib', '1254053-43-4', 'asp2215', 'asp-2215', 'xospata', 'asp 2215', '66d92mgc8m', 'gilteritinib hcl',
    'gilteritinibum', 'c6f', 'schembl282229', 'gtpl8708', 'chembl3301622', 'chebi:145372', 'bdbm144315', 'c29h44n8o3', 'bcp28756', 'ex-a2775', '3694ah',
    'mfcd28144685', 'nsc787846', 'nsc787854', 'nsc788454', 'nsc800106', 's7754', 'ccg-270016', 'cs-3885', 'db12141', 'nsc-787846', 'nsc-787854', 'nsc-788454',
    'nsc-800106', 'sb16988', 'ncgc00481652-01', 'ncgc00481652-02', 'ac-29030', 'as-35199', 'hy-12432', 'qc-11768', 'db-108103', 'a14411', 'd10709', 'q27077802'
]

synonyms_glasdegib = ['1095173-27-5', 'pf-04449913', 'pf 04449913', 'daurismo', 'k673dmo5h9', 'chembl2043437', 'c21h22n6o', 'pf-913', 'glasdegibum',
    'gtpl8201', 'schembl2068480', 'ex-a858', 'chebi:145428', 'amy38164', 'vtb17327', '2640ah', 'bdbm50385635', 'mfcd25976839', 'nsc775772', 
    'zinc68251434', '1095173-27-5', 'ccg-268350', 'db11978', 'nsc-775772', 'sb16679', 'ncgc00378600-02', 'bs-14357', 'hy-16391', 'qc-11459', 's7160', 'd10636',
    'z-3230', 'j-690029', 'q27077810'
]

synonyms_doxorubicin = ['adriamycin', 'doxorubicine']

synonyms_thioguanine = ['6-thioguanine', '6-tg']

synonyms_teniposide = ['vm-26']

synonyms_mercaptopurine = ['6-mp']

def contains(str, seq):
    '''Verify if a string (str) contains any of the characaters in a list.
  
    Args:
      str: string
      seq: list of characters
  
    Returns:
      0 if the string does not contain any of the characters, 1 otherwise.
    '''
    for c in seq:
        if c in str: 
            return 1
    return 0


def write_file(text, file_path):
    '''Write the result (word_list) of cleaning function into a .txt file, called "results_file_clean.txt"

    Args: 
      text: all pre-processed text 
    '''
    with open("./{}_clean.txt".format(file_path[:-4]), "w+", encoding="utf-8") as outfile:
      outfile.write("\n".join(text))


def clean_file(file_path):
    '''Pre-process text file with all the articles abstracts. This function removes from text stop words, units, 
    symbols, punctuation and unwanted regular expressions and isolated numbers from the text. It uses the two other 
    functions above.

    Args:
      file_path: the path of the original .txt file. 
    '''

    summaries = [s.strip() for s in open(file_path, encoding="utf-8")]
    word_list = []
    for s in summaries:
        s = re.sub('<[^>]+>', '', s)
        s = re.sub('\\s+', ' ', s)
        s = re.sub('([--:\w?@%&+~#=]*\.[a-z]{2,4}\/{0,2})((?:[?&](?:\w+)=(?:\w+))+|[--:\w?@%&+~#=]+)?', '', s)
        s = re.sub('\d+\W+\d+', '', s)
        s = s.lower()
        s = re.sub("|".join(sorted(synonyms_cytarabine, key = len, reverse = True)), 'cytarabine', s)
        s = re.sub("|".join(sorted(synonyms_daunorubicin, key = len, reverse = True)), 'daunorubicin', s)
        s = re.sub("|".join(sorted(synonyms_azacitidine, key = len, reverse = True)), 'azacitidine', s)
        s = re.sub("|".join(sorted(synonyms_gemtuzumab_ozogamicin, key = len, reverse = True)), 'gentuzumab ozogamicin', s)
        s = re.sub("|".join(sorted(synonyms_midostaurin, key = len, reverse = True)), 'midostaurin', s)
        s = re.sub("|".join(sorted(synonyms_cpx_351, key = len, reverse = True)), 'cpx-351', s)
        s = re.sub("|".join(sorted(synonyms_ivosidenib, key = len, reverse = True)), 'ivosidenib', s)
        s = re.sub("|".join(sorted(synonyms_venetoclax, key = len, reverse = True)), 'venetoclax', s)
        s = re.sub("|".join(sorted(synonyms_enasidenib, key = len, reverse = True)), 'enasidenib', s)
        s = re.sub("|".join(sorted(synonyms_gilteritinib, key = len, reverse = True)), 'gilteritinib', s)
        s = re.sub("|".join(sorted(synonyms_glasdegib, key = len, reverse = True)), 'glasdegib', s)
        s = re.sub("|".join(sorted(synonyms_doxorubicin, key = len, reverse = True)), 'doxorubicin', s)
        s = re.sub("|".join(sorted(synonyms_thioguanine, key = len, reverse = True)), 'thioguanine', s)
        s = re.sub("|".join(sorted(synonyms_teniposide, key = len, reverse = True)), 'teniposide', s)
        s = re.sub("|".join(sorted(synonyms_mercaptopurine, key = len, reverse = True)), 'mercaptopurine', s)
        s = s.translate(str.maketrans('', '', string.punctuation.replace('-', '')))
        s = nltk.sent_tokenize(s)
        for sent in s:
            words = nltk.word_tokenize(sent)
            words = [w for w in words if not w.isdigit()]
            words = [w for w in words if contains(w, units_and_symbols)==0]
            words = [w if w not in fix_typos_dict else fix_typos_dict[w] for w in words]
            #words = [stemmer.stem(word) for (word, pos) in nltk.pos_tag(words) if (is_verb(pos)==1) or word if (is_verb(pos)==0)]
            words = [lemmatizer.lemmatize(word, wordnet.VERB) for (word, pos) in nltk.pos_tag(words)]

        words = [word for word in words if not word in stop_words]
        word_list.append(words)

    res = list(map(' '.join, word_list))
    write_file(res, file_path)

filenames = sorted([str(x) for x in Path('./results_aggregated/').glob('*.txt')])

for f in filenames:
    print('cleaning {}'.format(f))
    clean_file(f)
