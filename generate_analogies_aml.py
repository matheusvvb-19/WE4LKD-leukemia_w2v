##################################################
## Generates a txt file containing the word-pair analogies about AML.
##################################################
## Author: {name}
## Copyright: Copyright 2022, Discovering Latent Knowledge in medical paper on Acute Myeloid Leukemia
## Email: {contact_email}
##################################################

# IMPORTS:
import pandas as pd
import operator
sys.path.append('./pubchem/')
from clean_summaries import get_target_compounds

# FUNCTIONS:
def get_AML_mutated_genes():
    return sorted(['idh1', 'runx1', 'tp53', 'ptpn11', 'kit', 'u2af1', 'smc1a', 'stag2', 'phf6', 'brinp3', 'rad21', 'plce1', 'ezh2', 'calr', 'dclk1', 'pkd1l2', 'cadm2', 'csf3r', 'ctnna2', 'epha3', 'lpa', 'srsf2', 'trpc1', 'rif1'])

def get_target_compounds_side_effects(remove_common_effects):
    """ Returns a Python dictionary containing the target compounds and their related main side effects.
        The side effects were mannualy selected from http://www.bccancer.bc.ca/health-professionals/clinical-resources/cancer-drug-manual/drug-index#"""

    dictionary = {
        'cytarabine': ['anemia', 'leukopenia', 'neutropenia', 'thrombocytopenia', 'rash', 'nausea', 'vomiting', 'infection', 'sepsis', 'neurotoxicity'],
        'daunorubicin': ['myelosuppression', 'cardiomyopathy', 'stomatitis'],
        'azacitidine': ['anemia', 'neutropenia', 'thrombocytopenia', 'constipation', 'nausea', 'vomiting', 'pyrexia'],
        'midostaurin': ['anemia', 'leukopenia', 'lymphopenia', 'neutropenia', 'thrombocytopenia', 'nausea', 'stomatitis', 'vomiting', 'pyrexia', 'hypersensitivity', 'pneumonia', 'sepsis', 'hypocalcemia', 'hypokalemia', 'hypomagnesemia', 'rash', 'hypotension'],
        'gemtuzumab-ozogamicin': ['anemia', 'neutropenia', 'pancytopenia', 'thrombocytopenia', 'nausea', 'vomiting', 'chills', 'fatigue', 'pyrexia', 'infection', 'sepsis', 'hemorrhage'],
        'vyxeos': ['neutropenia', 'thrombocytopenia', 'cardiotoxicity', 'constipation', 'diarrhea', 'mucositis', 'nausea', 'vomiting', 'edema', 'fatigue', 'infection', 'headache', 'dyspnea', 'epistaxis', 'hemorrhage'],
        'ivosidenib': ['chills', 'confusion', 'cough', 'fainting', 'fever', 'headache', 'rash'],
        'venetoclax': ['anemia', 'neutropenia', 'thrombocytopenia', 'diarrhea', 'nausea', 'fatigue', 'pyrexia', 'pneumonia', 'hyperkalemia', 'hyperphosphatemia', 'hyperruricemia', 'hypocalcemia'],
        'enasidenib': ['agitation', 'chills', 'confusion', 'cough', 'depression', 'dizziness', 'fainting', 'fever', 'headache', 'hostility', 'irritability', 'lightheadedness', 'nausea'],
        'gilteritinib': ['thrombocytopenia', 'pancreatitis', 'fatigue', 'pneumonia', 'hypotension'],
        'glasdegib': ['chills', 'confusion', 'cough', 'drowsiness', 'fainting', 'fever', 'headache', 'lightheadedness', 'nausea', 'nervousness', 'nosebleeds', 'paralysis'],
        'arsenictrioxide': ['hyperleukocytosis', 'thrombocytopenia', 'tachycardia', 'diarrhea', 'nausea', 'vomiting', 'chills', 'fatigue', 'pyrexia', 'hyperglycemia', 'headache', 'paresthesia', 'insomnia', 'cough', 'dyspnea', 'dermatitis'],
        'cyclophosphamide': ['myelosuppression', 'nausea', 'vomiting'],
        'dexamethasone': ['hypokalemia', 'hyperglycemia', 'hypertension', 'psychosis'],
        'idarubicin': ['anemia', 'hemorrhage', 'leukopenia', 'neutropenia', 'thrombocytopenia', 'anorexia', 'diarrhea', 'nausea', 'vomiting', 'fever', 'infection', 'alopecia'],
        'mitoxantrone': ['anemia', 'leukopenia', 'myelosuppression', 'cardiomyopathy'],
        'pemigatinib': ['confusion', 'diarrhea', 'lightheadedness', 'dizziness', 'fainting', 'seizures', 'thirst', 'tremor'],
        'prednisone': ['heartburn', 'nausea', 'infection'],
        'rituximab': ['neutropenia', 'nausea', 'chills', 'fever', 'hypersensitivity', 'infection', 'rash'],
        'thioguanine': ['anemia', 'leukopenia', 'thrombocytopenia', 'hepatoxicity'],
        'vincristine': ['alopecia'],
    }

    if remove_common_effects > 0 and remove_common_effects <= 1:
        number_of_compounds = len(dictionary.keys())
        effects_frequency = {}

        for key, analogue_words in dictionary.items():
            if len(analogue_words) > 0:
                for aw in analogue_words:
                    try:
                        effects_frequency[aw] += 1
                    
                    except:
                        effects_frequency[aw] = 1

        for key, frequency in effects_frequency.items():
            effects_frequency[key] = effects_frequency[key] / number_of_compounds

        dictionary_copy = dictionary
        for key, analogue_words in dictionary.items():
            if len(analogue_words) > 0:
                for aw in analogue_words:
                    if effects_frequency[aw] >= remove_common_effects:
                        dictionary_copy[key].remove(aw)        

        return dictionary_copy

    else:
        return dictionary 

def get_words_frequency():
    """ Creates and returns a Python dictionary containing the frequency of the words in corpus, sorted in decrescent order."""
    
    df = pd.read_csv('/data/doubleblind/doubleblind/pubchem/results_pandas.csv', escapechar='\\')
    abstracts = df['summary'].to_list()
    abstracts = [x.split() for x in abstracts]

    wanted_words = set(get_AML_mutated_genes() + get_target_compounds() + list(dict.fromkeys([item for sublist in get_target_compounds_side_effects(0.25).values() for item in sublist])) + ['anthracycline', 'cancer', 'aml', 'malignancies', 'tumor', 'mutation'])
    words_frequency = {}
    for abst in abstracts:
        for word in abst:
            if word in wanted_words:
                if word not in words_frequency:
                    words_frequency[word] = 1
                
                else:
                    words_frequency[word] += 1
            
            else:
                continue
    
    return dict(sorted(words_frequency.items(), key=operator.itemgetter(1), reverse=True))

def generate_analogies(percentage=0.25, frequency_threshold=300):
    """ Generates a list of analogies considering the 'percentage' most common side effects and only tokens that occurs more than 'frequency_threshold' times in the corpus.
    
    Args:
        percentage: percentage of most common side effects to be eliminated from the analogies;
        frequency_threshold: eleiminates analogies that contains words with less than this determined number of occurrences in corpus.
    
    Returns: a list of analogies with the format 'word1' 'word2' 'word3' 'word4'
    """
    
    words_frequency = get_words_frequency()
    analogies = []

    for key, analogue_words in get_target_compounds_side_effects(percentage).items():
        if len(analogue_words) > 0:
            for aw in analogue_words:
                for key_2, analogue_words_2 in get_target_compounds_side_effects(percentage).items():
                    if len(analogue_words_2) > 0:
                        for aw_2 in analogue_words_2:
                            if aw != aw_2 and key != key_2:
                                try:
                                    if words_frequency[aw] < frequency_threshold or words_frequency[aw_2] < frequency_threshold or words_frequency[key] < frequency_threshold or words_frequency[key_2] < frequency_threshold:
                                        continue

                                    else:
                                        analogies.append('{} {} {} {}\n'.format(key.lower(), aw.lower(), key_2.lower(), aw_2.lower()))
                                        analogies.append('{} {} {} {}\n'.format(aw.lower(), key.lower(), aw_2.lower(), key_2.lower()))
                                
                                except:
                                    continue
    
    return analogies

# MAIN PROGRAM:
if __name__ == '__main__':
    with open('./data/analogies_aml.txt', 'w') as f:
        for c in get_target_compounds():
            f.write('anthracycline cancer {} aml\n'.format(c))
            f.write('anthracycline malignancies {} aml\n'.format(c))
            f.write('anthracycline tumor {} aml\n'.format(c))

            f.write('cancer anthracycline aml {}\n'.format(c))
            f.write('malignancies anthracycline aml {}\n'.format(c))
            f.write('tumor anthracycline aml {}\n'.format(c))

        for g in get_AML_mutated_genes():
            f.write('mutation cancer {} aml\n'.format(g))
            f.write('mutation malignancies {} aml\n'.format(g))
            f.write('mutation tumor {} aml\n'.format(g))

            f.write('cancer mutation aml {}\n'.format(g))
            f.write('malignancies mutation aml {}\n'.format(g))
            f.write('tumor mutation aml {}\n'.format(g))

        for a in generate_analogies():
            f.write(a)

    print('END!')
