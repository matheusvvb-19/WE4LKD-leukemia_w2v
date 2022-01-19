<div id="topo"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">WE4LKD</h3>

  <p align="center">
    Word Embeddings For Latent Knowledge Discovery
    <br />
    <a href="https://share.streamlit.io/matheusvvb-19/we4lkd-leukemia_w2v/main"><strong>Visualizar Embeddings</strong></a>
    <br />
    <br />
    <a href="https://drive.google.com/drive/folders/1Fq5HkZx8DmWWAXnhkYSuX7r_GZjd6jGh?usp=sharing">Arquivos do Projeto</a>
    ·
    <a href="https://github.com/matheusvvb-19/WE4LKD-leukemia_w2v/issues">Reportar Bug</a>
    ·
    <a href="https://github.com/matheusvvb-19/WE4LKD-leukemia_w2v/issues">Sugerir Funcionalidade</a>
  </p>
</div>



<!-- SUMÁRIO -->
<details>
  <summary>Sumário</summary>
  <ol>
    <li>
      <a href="#sobre-o-projeto">Sobre o projeto</a>
      <ul>
        <li><a href="#bibliotecas">Bibliotecas</a></li>
      </ul>
    </li>
    <li>
      <a href="#inicialização">Inicialização</a>
      <ul>
        <li><a href="#pré-requisitos">Pré-requisitos</a></li>
        <li><a href="#instalação">Instalação</a></li>
      </ul>
    </li>
    <li><a href="#uso">Uso</a></li>
    <li><a href="#futuras-melhorias">Futuras Melhorias</a></li>
    <li><a href="#contato">Contato</a></li>
    <li><a href="#inspiração">Inspiração</a></li>
  </ol>
</details>


<!-- SOBRE -->
## Sobre o projeto

Este projeto tem por objetivo estudar e analisar a possível existência de conhecimento latente em artigos médicos sobre Leucemia Mielóide Aguda (AML), um tipo de câncer agressivo e sem tratamentos muito eficazes. Para isso, são gerados modelos de representação distribuída Word2Vec a partir de prefácios de artigos disponíveis no [PubMed](https://pubmed.ncbi.nlm.nih.gov/).

_descrição mais detalhada em breve_

<p align="right">(<a href="#topo">voltar ao topo</a>)</p>


### Bibliotecas

* [Gensim](https://radimrehurek.com/gensim/)
* [NLTK](https://www.nltk.org/)
* [Numpy](https://numpy.org/)
* [Plotly](https://plotly.com/)
* [Scikit-learn](https://scikit-learn.org/stable/)
* [Streamlit](https://streamlit.io/)
* [TensorFlow](https://www.tensorflow.org/?hl=pt-br)

<p align="right">(<a href="#topo">voltar ao topo</a>)</p>


### Pré-requisitos

Para utilizar este projeto, você precisa ter o Pyhton instalado em sua máquina. Este projeto se utilizou da [versão 3.6 do Python](https://www.python.org/downloads/release/python-360/).
* Baixe e instale o Python [aqui](https://www.python.org/downloads/).

Além disso, você também precisará do [Pip](https://pypi.org/project/pip/), gerenciador de pacotes Python para instalar os demais requerimentos do projeto.


<!-- INICIALIZAÇÃO -->
## Inicialização

Para iniciar, crie um ambiente virtual Python, ative-o e instale os requerimentos do projeto:
```sh
python3 -m venv venv
source venv/bin/activate
pip3 install --ignore-installed -r requirements.txt
```


### Instalação

1. Clone o repositório
   ```sh
   git clone https://github.com/matheusvvb-19/WE4LKD-leukemia.git
   ```
2. Crie o ambiente virtual Python e instale os pacotes necessários (veja a seção anterior)

<p align="right">(<a href="#topo">voltar ao topo</a>)</p>



<!-- USO -->
## Uso

Após fazer a inicialização e instalação, chegou a hora de executar o projeto:
1. Caso queira, você pode trocar as frases de busca no arquivo `search_strings.txt`
2. Execute o arquivo `crawler.py` ou baixe os resultados
  ```sh
  mkdir results
  python3 crawler.py
  ```
3. Execute o arquivo `merge_txt.py`
  ```sh
  mkdir results_aggregated
  python3 merge_txt.py
  ```
4. Execute o arquivo `clean_text.py`
```sh
  python3 clean_text.py
```
5. Treine os modelos Word2Vec
```sh
  cd word2vec
  python3 train.py
```
6. Visualize e explore seus modelos de representação distribuída gerados, utilizando o [Visualizador de Embeddings](https://share.streamlit.io/matheusvvb-19/we4lkd-leukemia_w2v/main) construído com o Streamlit.<br> Caso o visualizador não esteja disponível, execute-o localmente em sua máquina com o comando:
```sh
  cd ..
  streamlit run streamlit_app.py
```
7. Para gerar relatórios de análise de conhecimento latente, execute o comando:
```sh
  python3 analyze.py <palavras_comuns_eliminadas>
```
O último argumnento da linha de comando indica a quantidade de palavras em inglês mais comuns - segundo [Beautiful Soup](https://norvig.com/ngrams/count_1w.txt) -  que se deseja remover do modelo no momento da visualização de vizinhança. O valor desse arguento pode ser nenhum (vazio), para quando não se deseja remover nenhuma palavra, ou um número inteiro. No Visualizador de Embeddings, as opções são 5, 10, 15 ou 20 mil palavras.

<p align="right">(<a href="#topo">voltar ao topo</a>)</p>



<!-- FUTURAS MELHORIAS -->
## Futuras Melhorias

- [x] Manter o Visualizador de Embeddings online com o Streamlit
- [x] Excluir palavras omuns na visualização das embeddings
- [x] Gerar relatórios em .pdf, também com exclusão de palavras comuns
- [ ] _ainda em construção_...

<p align="right">(<a href="#topo">voltar ao topo</a>)</p>


<!-- CONTATO -->
## Contato

Matheus Vargas Volpon Berto<br>
matheusvvb@hotmail.com<br>
[LinkedIn](https://www.linkedin.com/in/matheus-volpon/)

<p align="right">(<a href="#topo">voltar ao topo</a>)</p>


<!-- INSPIRAÇÃO -->
## Inspiração

* [Unsupervised word embeddings capture latent knowledge from materials science literature", Nature 571, 95–98 (2019)](https://github.com/materialsintelligence/mat2vec)

<p align="right">(<a href="#topo">voltar ao topo</a>)</p>
