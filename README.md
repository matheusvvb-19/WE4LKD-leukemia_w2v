# Word embeddings capture latent knowledge discovery
Inspired by [this application](https://github.com/materialsintelligence/mat2vec)

## Setup
```
#create venv
python3 -m venv venv
#activate venv
source venv/bin/activate
#install requirements
pip3 install --ignore-installed -r requirements.txt
```

## Change search strings on `search_strings.txt` (optional)

## Run crawler or download [results](https://drive.google.com/drive/folders/1Ryx6QjV0FIAD19mPBF4lTkMWfKo_CH4G?usp=sharing)
```
mkdir results
python3 crawler.py
```

## Merge abstract files
This will generate `results_file.txt` from abstracts on results folder
```
python3 merge_txt.py
```

## Clean abstract files
This will generate `results_file_clean.txt`
```
python3 clean_text.py
```
## Word embeddings

### Word2Vec
Training the model
```
cd word2vec
python3 train.py
```
Loading the model
```
from gensim.models import Word2Vec
model = Word2Vec.load('model.bin')
```
Access vector for one word
```
model.wv['cytarabin']
```

### GloVe
```
git clone http://github.com/stanfordnlp/glove
cd glove && make
```
Making changes to `demo.sh`:
- Remove the script from if to fi after 'make'
- Replace the CORPUS name with `"../results_file_clean.txt"`
- On `if [ "$CORPUS" = 'text8' ]; then` replace `text8` with `"../results_file_clean.txt"`

Training the model
```./demo.sh```

Word vectors will be placed on `vectors.txt` 

## Create inputs for [Projector](https://projector.tensorflow.org/) inside tensorboard_inputs folder 
After training the model, choose which one will be transformed to tensorboard format. Pass as a second argument the number of common words that will not be plotted. The number of words must be between 1 and 333.333, the default value is 10.000
```
python3 to_tensorboard_format.py glove <your number>
```

or

```
python3 to_tensorboard_format.py word2vec <your number>
```


