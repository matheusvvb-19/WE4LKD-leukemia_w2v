from gensim.models import Word2Vec
from pathlib import Path

filenames = [str(x) for x in Path('../results_aggregated/').glob('*_clean.txt')]

# define training data
for f in filenames:
    print(f)
    summaries = [s.strip() for s in open(f, encoding='utf-8')]

    word_list = []
    for s in summaries:
        s = s.split(' ')
        word_list.append(s)    

    # train model
    model = Word2Vec(word_list, min_count=2, size=30)

    # save model
    file_name = f.split('/')[2]
    model.save('model_{}.bin'.format(file_name[:-4]))
