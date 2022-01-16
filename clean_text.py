import nltk
import string
import re
from crawler import list_from_txt
from pathlib import Path

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
    nltk.download('stopwords')
    from nltk.corpus import stopwords 

    stop_words = set(stopwords.words('english'))
    personal_stop_words = list_from_txt('personal_stop_words.txt')

    fix_typos_dict = {'remarkablely': 'remarkably',
                      'leukaemia': 'leukemia',
                      'leukaemias': 'leukemias',
                      'efficiacy': 'efficiency',
                      'carboxylpheny': 'carboxyphenyl',
                      'dimethylthiazole': 'dimethylthiazol',
    }

    units_and_symbols = ['μ', 'Ω', 'ω', 'μm', 'mol', '°c', '≥', '≤', '±', 
                         'day', 'month', 'year', '·', 'week', 'μ', 'days',
                         'weeks', 'years', 'mg', 'mm', '®', 'µl'
    ]

    for i in personal_stop_words:
        stop_words.add(i)

    # define training data
    summaries = [s.strip() for s in open(file_path, encoding="utf-8")]
    remove_chars = list(string.punctuation)
    remove_chars.remove('-')

    word_list = []
    for s in summaries:
        s = re.sub('<[^>]+>', '', s)
        s = re.sub('\\s+', ' ', s)
        s = re.sub('([--:\w?@%&+~#=]*\.[a-z]{2,4}\/{0,2})((?:[?&](?:\w+)=(?:\w+))+|[--:\w?@%&+~#=]+)?', '', s)
        s = re.sub('\d+\W+\d+', '', s)
        s = s.split(' ')
        s = [w.lower().translate({ord(x): '' for x in remove_chars}) for w in s]
        s = [w for w in s if not w.isdigit()]
        s = [w for w in s if contains(w, units_and_symbols)==0]
        s = [w for w in s if not w in stop_words]
        s = [w if w not in fix_typos_dict else fix_typos_dict[w] for w in s]
        word_list.append(s)

    res = list(map(' '.join, word_list))
    write_file(res, file_path)


filenames = [str(x) for x in Path('./results_aggregated/').glob('*.txt')]

for f in filenames:
    print('cleaning {}'.format(f))
    clean_file(f)
