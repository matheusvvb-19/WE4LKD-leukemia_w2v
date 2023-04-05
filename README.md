<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">WE4LKD</h3>

  <p align="center">
    Word Embeddings For Latent Knowledge Discovery
    <br />
    <br />
    <a href="https://drive.google.com/drive/folders/1Fq5HkZx8DmWWAXnhkYSuX7r_GZjd6jGh?usp=sharing">Google Drive Folder</a>
    ·
    <a href="https://github.com/matheusvvb-19/WE4LKD-leukemia_w2v/issues">Report Bug</a>
    ·
    <a href="https://github.com/matheusvvb-19/WE4LKD-leukemia_w2v/issues">Request Feature</a>
  </p>
</div>

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://embedding-viewer.streamlit.app/)

<!-- CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about">About</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#references">References</a></li>
  </ol>
</details>


<!-- ABOUT -->
## About

This project aims to study and analyze the possible existence of latent knowledge in medical articles about Acute Myeloid Leukemia (AML), an aggressive type of cancer without very effective treatments. For this, Word2Vec distributed representation models are generated from article prefaces available in [PubMed](https://pubmed.ncbi.nlm.nih.gov/) and through dimensionality reduction techniques their vectors are plotted and analyzed.

<p align="right"><a href="#top">⬆️</a></p>


### Built With

* [Gensim](https://radimrehurek.com/gensim/)
* [NLTK](https://www.nltk.org/)
* [Numpy](https://numpy.org/)
* [Plotly](https://plotly.com/)
* [Scikit-learn](https://scikit-learn.org/stable/)
* [Streamlit](https://streamlit.io/)

<p align="right"><a href="#top">⬆️</a></p>


### Prerequisites

To use this project, you need to have Pyhton installed on your machine. This project used [Python version 3.6](https://www.python.org/downloads/release/python-360/).
* Download and install Python [here](https://www.python.org/downloads/).

In addition, you will also need [Pip](https://pypi.org/project/pip/), the Python package manager to install the other requirements of the project.

<p align="right"><a href="#top">⬆️</a></p>

<!-- INICIALIZAÇÃO -->
## Getting Started

To get started, create a Python virtual environment, activate it and install the project requirements:
```sh
python3 -m venv venv
source venv/bin/activate
pip3 install --ignore-installed -r requirements.txt
```

### Installation

Clone the repository
```sh
git clone https://github.com/matheusvvb-19/WE4LKD-leukemia.git
```

<p align="right"><a href="#top">⬆️</a></p>



<!-- USAGE -->
## Usage

After initialization and installation, it is time to run the project:
1. If you like, you can change the search phrases in the `search_strings.txt` file
2. Run `crawler.py`
  ```sh
  mkdir results
  python3 crawler.py
  ```
3. Execute the script `merge_txt.py`, this will generate the _.txt_ files with all articles between periods
  ```sh
  mkdir results_aggregated
  python3 merge_txt.py
  ```
4. Execute the script `clean_text.py`, which will clean the merged _.txt_ files
```sh
  python3 clean_text.py
```
5. Train the Word2Vec models
```sh
  cd word2vec
  python3 train.py
```
6. View and explore your generated distributed representation models using the [Embeddings Viewer](https://share.streamlit.io/matheusvvb-19/we4lkd-leukemia_w2v/main) built with Streamlit.<br> If the viewer is not available, run it locally on your machine with the command
```sh
  cd ..
  streamlit run streamlit_app.py
```
7. To generate latent knowledge analysis reports, run the command
```sh
  python3 analyze.py <palavras_comuns_eliminadas>
```
The last argument of the command line indicates the domain constraint applied to the models. This constraint eliminates certain words from the vocabularies of the models, decreasing spurious words and making the analysis easier. The possible values are:
* nci_cancer_drugs
* fda_drugs
* an integer representing the number of most common English words - according to [Beautiful Soup](https://norvig.com/ngrams/count_1w.txt) - that you want to remove from the model.

This argument can also be empty.

<p align="right"><a href="#top">⬆️</a></p>


<!-- ROADMAP -->
## Roadmap

- [x] Deploy Embedding Viewer with Streamlit Cloud
- [x] Exclude common words from the models
- [x] Apply domain constraint
- [x] Generate _.pdf_ reports automatically, also with the domain constraint
- [x] Reduce long words of plotly bar plots in Embedding Viewer
- [x] Iterative search on Embedding Viewer - analyze context
- [ ] Use Named Entity Recognition to filter words
- [ ] Standardize synonyms of compounds automatically

<p align="right"><a href="#top">⬆️</a></p>


<!-- CONTACT -->
## Contact

Matheus Vargas Volpon Berto<br>
matheusvvb@hotmail.com<br>
[LinkedIn](https://www.linkedin.com/in/matheus-volpon/)

<p align="right"><a href="#top">⬆️</a></p>


<!-- REFERENCES -->
## References

* [Unsupervised word embeddings capture latent knowledge from materials science literature", Nature 571, 95–98 (2019)](https://github.com/materialsintelligence/mat2vec)

<p align="right"><a href="#top">⬆️</a></p>
