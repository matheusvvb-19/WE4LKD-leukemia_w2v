<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">WE4LKD</h3>

  <p align="center">
    Word Embeddings For Latent Knowledge Discovery
    <br />
    <br />
    ·
    <a href="https://github.com/matheusvvb-19/WE4LKD-leukemia_w2v/issues">Report Bug</a>
    ·
    <a href="https://github.com/matheusvvb-19/WE4LKD-leukemia_w2v/issues">Request Feature</a>
  </p>
</div>

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](doubleblind)

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

This project aims to study and analyze the possible existence of latent knowledge in medical articles about Acute Myeloid Leukemia (AML), an aggressive type of cancer without very effective treatments. For this, Word2Vec and FastText distributed representation models are generated from article prefaces available in [PubMed](https://pubmed.ncbi.nlm.nih.gov/) and their vectors are analyzed trough vector operations with keywords.

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
4. Download [this file](https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-Title.gz) and [this file](https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-Synonym-filtered.gz), put both inside ./pubchem/ folder and excute `titles_and_synonyms.py`
```sh
  python3 titles_and_synonyms.py
```
5. Execute the script `clean_summaries.py`, which will clean the merged _.txt_ files into a PySPark DataFrame
```sh
  python3 clean_summaries.py
```
6. Train the Word2Vec and FastText models, run the script twice (doing the appropriate change in code)
```sh
  cd word2vec
  python3 train.py
```
7. To generate csv files containing the correlation (dot product) between compounds and "AML", execute
```sh
  cd ..
  python3 generate_dotproducts_csv.py
```

<p align="right"><a href="#top">⬆️</a></p>

<!-- CONTACT -->
## Contact

doubleblind<br>
doubleblind<br>
[LinkedIn](doubleblind)

<p align="right"><a href="#top">⬆️</a></p>


<!-- REFERENCES -->
## References

* [Unsupervised word embeddings capture latent knowledge from materials science literature, Nature 571, 95–98 (2019)](https://github.com/materialsintelligence/mat2vec)

<p align="right"><a href="#top">⬆️</a></p>
