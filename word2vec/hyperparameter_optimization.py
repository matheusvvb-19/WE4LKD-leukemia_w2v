import os, jinja2

from gensim import models
from gensim.models import Word2Vec, FastText

from jinja2 import Template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import date

import tikzplotlib
from tikzplotlib import get_tikz_code

def list_from_txt(file_path):
    '''Creates a list of itens based on a .txt file, each line becomes an item.
    
    Args: 
      file_path: the path where the .txt file was created. 
    '''
    
    strings_list = []
    with open (file_path, 'rt', encoding='utf-8') as file:
        for line in file:
            strings_list.append(line.rstrip('\n'))
    return strings_list

def score_func(model, analogies_aml, number_of_analogies_aml, analogies_general, number_of_analogies_general, analogies_biomedical, number_of_analogies_biomedical, topn):
    number_of_analogies_all = number_of_analogies_aml + number_of_analogies_general + number_of_analogies_biomedical

    number_of_correct_analogies_all = 0
    number_of_correct_analogies_aml = 0
    number_of_correct_analogies_general = 0
    number_of_correct_analogies_biomedical = 0

    for analogie in analogies_aml:
        words = analogie.split(' ')
        try:
            for pair in model.wv.most_similar(positive=[model.wv[words[1]], model.wv[words[2]]], negative=[model.wv[words[0]]], topn=topn):
                if words[3] == pair[0]:
                    number_of_correct_analogies_all += 1
                    number_of_correct_analogies_aml += 1

        except:
            continue

    for analogie in analogies_general:
        words = analogie.split(' ')
        try:
            for pair in model.wv.most_similar(positive=[model.wv[words[1]], model.wv[words[2]]], negative=[model.wv[words[0]]], topn=topn):
                if words[3] == pair[0]:
                    number_of_correct_analogies_all += 1
                    number_of_correct_analogies_general += 1

        except:
            continue

    for analogie in analogies_biomedical:
        words = analogie.split(' ')
        try:
            for pair in model.wv.most_similar(positive=[model.wv[words[1]], model.wv[words[2]]], negative=[model.wv[words[0]]], topn=topn):
                if words[3] == pair[0]:
                    number_of_correct_analogies_all += 1
                    number_of_correct_analogies_biomedical += 1

        except:
            continue

    return (
        (number_of_correct_analogies_all * 100) / number_of_analogies_all,
        (number_of_correct_analogies_aml * 100) / number_of_analogies_aml,
        (number_of_correct_analogies_general * 100) / number_of_analogies_general,
        (number_of_correct_analogies_biomedical * 100) / number_of_analogies_biomedical
    )

def contains(string, unwanted_words):
    for x in string.split(' '):
        if x in unwanted_words:
            return True
    
    return False

def get_valid_analogies(model):
    analogies_aml = list_from_txt('../analogies_aml.txt')
    analogies_general = list_from_txt('../analogies_general.txt')
    analogies_biomedical = list_from_txt('../analogies_biomedical.txt')

    analogie_words_present_in_model_vocab = set()
    remove_analogies_with_the_words = []
    for analogie in analogies_aml:
        words = [x.lower() for x in analogie.split(' ')]
        if ':' in words:
            continue

        else:
            for w in words:
                if w not in analogie_words_present_in_model_vocab:
                    if w in list(model.wv.vocab):
                        analogie_words_present_in_model_vocab.add(w)
                    else:
                        remove_analogies_with_the_words.append(w)
                else:
                    continue
    
    analogies_aml = [x for x in analogies_aml if contains(x, remove_analogies_with_the_words)==False]
    
    analogie_words_present_in_model_vocab = set()
    remove_analogies_with_the_words = []
    for analogie in analogies_general:
        words = [x.lower() for x in analogie.split(' ')]
        if ':' in words:
            continue

        else:
            for w in words:
                if w not in analogie_words_present_in_model_vocab:
                    if w in list(model.wv.vocab):
                        analogie_words_present_in_model_vocab.add(w)
                    else:
                        remove_analogies_with_the_words.append(w)
                else:
                    continue
    
    analogies_general = [x for x in analogies_general if contains(x, remove_analogies_with_the_words)==False]

    analogie_words_present_in_model_vocab = set()
    remove_analogies_with_the_words = []
    for analogie in analogies_biomedical:
        words = [x.lower() for x in analogie.split(' ')]
        if ':' in words:
            continue

        else:
            for w in words:
                if w not in analogie_words_present_in_model_vocab:
                    if w in list(model.wv.vocab):
                        analogie_words_present_in_model_vocab.add(w)
                    else:
                        remove_analogies_with_the_words.append(w)
                else:
                    continue
    
    analogies_biomedical = [x for x in analogies_biomedical if contains(x, remove_analogies_with_the_words)==False]

    # removing duplicates:
    analogies_aml = list(dict.fromkeys(analogies_aml))
    analogies_general = list(dict.fromkeys(analogies_general))
    analogies_biomedical = list(dict.fromkeys(analogies_biomedical))

    print(len(analogies_aml))
    print(len(analogies_general))
    print(len(analogies_biomedical))
    return analogies_aml, analogies_general, analogies_biomedical

def get_performance_bar_plot_from_df(df, topn_values, colors):
    models_names = df['Model name'].to_list()
    performance = {
        'All': df['All'].to_list(),
        'AML': df['AML'].to_list(),
        'Grammar': df['Grammar'].to_list(),
        'Biomedical': df['Biomedical'].to_list(),
    }

    x = np.arange(len(models_names))  # the label locations
    width = 0.2  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(figsize=(25, 8))

    for attribute, measurement in performance.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute, color=colors[multiplier])
        #ax.bar_label(rects, padding=3, rotation=45)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Score (%)')
    ax.set_xlabel('Models')
    #ax.set_title('Performance of each Word2Vec model: topn {}'.format(TOPN_VALUES[i]))
    ax.set_xticks(x + width, models_names)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True)
    ax.set_ylim(0, 100)

    caption = 'Performance of each model over the analogies, topn={}'.format(topn_values[i])

    return fig, caption

if __name__ == '__main__':
    print('Starting')

    # Jinja2 settings, commands to use in the .tex template file:
    latex_jinja_env = jinja2.Environment(
        block_start_string = '\BLOCK{',
        block_end_string = '}',
        variable_start_string = '\VAR{',
        variable_end_string = '}',
        comment_start_string = '\#{',
        comment_end_string = '}',
        line_statement_prefix = '%%',
        line_comment_prefix = '%#',
        trim_blocks = True,
        autoescape = False,
        loader = jinja2.FileSystemLoader(os.path.abspath('.'))
    )

    # loading template from .tex file:
    template = latex_jinja_env.get_template('./hyperparameter_optimization_template.tex')
    
    # constants:
    TESTING_DEFAULT_MODEL = True
    ANALYZING_FASTTEXT_MODELS = True
    if ANALYZING_FASTTEXT_MODELS:
        MODELS = sorted([f.path for f in os.scandir('../fasttext/models_nolemma/') if f.name.endswith('.model')])
        model_type = 'FastText'

    else:
        MODELS = sorted([f.path for f in os.scandir('./models_nolemma/') if f.name.endswith('.model')])         # paths to the Word2Vec models
        model_type = 'Word2Vec'
        
    TOPN_VALUES = [10, 20, 30, 40]                                                                              # different values of `topn` to use during the validation of the analogies
    COLORS = ['blue', 'mediumseagreen', 'orange', 'hotpink']                                                # friendly color blidness palette for bar plots
    
    # table of models names and respective files:
    data_models_files = {
        'model name': [],
        'filepath': [],
    }

    for index, model_name in enumerate(MODELS):
        if 'default' in model_name.split('/')[-1]:
            data_models_files['model name'].append('Default')
            TESTING_DEFAULT_MODEL = True

        else:
            data_models_files['model name'].append('Model {}'.format(index+1))
        
        data_models_files['filepath'].append(model_name)
    
    df_models_files = pd.DataFrame(data=data_models_files)
    df_models_files_latex = df_models_files.style.to_latex(position_float='centering', caption="Models' names and their respective filepath.", position='ht', label='models_names_filepaths')

    # table of variable hyperarameters values from each model to be analyzed:
    data_optimization_hyperparameters = {
        'model name': [],
        'vector size': [],
        'learning rate': [],
        'negative sampling': [],
    }

    for index, model_name in enumerate(MODELS):
        model_name_splitted = model_name.split('/')[-1].split('.model')[0].split('_')[3:]
        if 'default' in  model_name.split('/')[-1]:
            data_optimization_hyperparameters['model name'].append('Default')
            TESTING_DEFAULT_MODEL  = True

        else:
            data_optimization_hyperparameters['model name'].append('Model {}'.format(index+1))
        data_optimization_hyperparameters['vector size'].append(float(model_name_splitted[0][1:]))
        data_optimization_hyperparameters['learning rate'].append(float(model_name_splitted[1][1:]))
        data_optimization_hyperparameters['negative sampling'].append(float(model_name_splitted[2][1:]))

    df_optimization_hyperparameters = pd.DataFrame(data=data_optimization_hyperparameters)
    df_optimization_hyperparameters_latex = df_optimization_hyperparameters.style.to_latex(position_float='centering', caption='Variable hyperparameter values.', position='ht', label='variable_hyperparameters')

    models_scores = {model_name: {topn: {'All': 0, 'AML': 0, 'Grammar': 0, 'Biomedical': 0} for topn in TOPN_VALUES} for model_name in MODELS}
    number_of_times_models_were_loaded = 0
    for topn in TOPN_VALUES:
        print('Topn value:', topn)
        for index, model_name in enumerate(MODELS):

            if ANALYZING_FASTTEXT_MODELS:
                model = FastText.load(model_name)
            
            else:
                model = Word2Vec.load(model_name)

            if number_of_times_models_were_loaded == 0:
                print('Getting valid analogies')
                analogies_aml, analogies_general, analogies_biomedical = get_valid_analogies(model)
                df_analogies = pd.DataFrame(data={'Analogies': ['All', 'AML', 'Grammar', 'Biomedical'], 'Amount': [len(analogies_aml)+len(analogies_general), len(analogies_aml), len(analogies_general), len(analogies_biomedical)]})
                df_analogies_latex = df_analogies.style.to_latex(position_float='centering', caption='Amount of analogies per type.', position='ht', label='analogies_amount')

            print('Computing score for model {}/{}'.format(index+1, len(MODELS)))
            model_score = score_func(model, analogies_aml, len(analogies_aml), analogies_general, len(analogies_general), analogies_biomedical, len(analogies_biomedical), topn=topn)

            models_scores[model_name][topn]['All'] = model_score[0]
            models_scores[model_name][topn]['AML'] = model_score[1]
            models_scores[model_name][topn]['Grammar'] = model_score[2]
            models_scores[model_name][topn]['Biomedical'] = model_score[3]

            number_of_times_models_were_loaded += 1

    print('Generating performance tables')
    performance_tables = []
    for topn in TOPN_VALUES:
        topn_scores = {'topn': topn}
        for model_name, model_scores in models_scores.items():
            topn_scores[model_name] = model_scores[topn]
        performance_tables.append(topn_scores)

    for index, topn_results in enumerate(performance_tables):
        data = {
            'Model name': [],
            'All': [],
            'AML': [],
            'Grammar': [],
            'Biomedical': [],
        }

        copy = topn_results
        del copy['topn']

        i = 0
        for key, score_dict in copy.items():
            if i == 0 and TESTING_DEFAULT_MODEL==True:
                data['Model name'].append('Default')
            else:
                data['Model name'].append('Model {}'.format(i+1))
            data['All'].append(score_dict['All'])
            data['AML'].append(score_dict['AML'])
            data['Grammar'].append(score_dict['Grammar'])
            data['Biomedical'].append(score_dict['Biomedical'])

            i += 1
        
        performance_tables[index] = pd.DataFrame(data=data)

    performance_tables_latex = [x.style.highlight_max(props='textbf:--rwrap;', axis=0).to_latex(position_float='centering', caption='Performance of each model over the analogies, topn={}.'.format(TOPN_VALUES[i]), position='ht') for i, x in enumerate(performance_tables)]

    print('Generating plots')
    performance_plots, performace_plots_captions = [], []
    for i, df in enumerate(performance_tables):
        fig, caption = get_performance_bar_plot_from_df(df, TOPN_VALUES, COLORS)
        performance_plots.append(fig)
        performace_plots_captions.append(caption)

    for i, plot in enumerate(performance_plots):
        performance_plots[i] = get_tikz_code(plot, axis_width='490', extra_axis_parameters=[
                                                            'legend style={at={(0.5, 1.2)},\
                                                            anchor=north,legend cell align=left},\
                                                            xticklabel style={font=\\tiny},\
                                                            ytick={0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100},\
                                                            ymax=100'
                                                        ])

    print('Generating .tex file')
    report_latex = template.render(
                        title=model_type,
                        hyperparameter_optimization_df=df_optimization_hyperparameters_latex,
                        model_names_filepaths_df=df_models_files_latex,
                        number_analogies_all=len(analogies_aml)+len(analogies_general)+len(analogies_biomedical),
                        number_analogies_AML=len(analogies_aml),
                        number_analogies_general=len(analogies_general),
                        number_analogies_biomedical=len(analogies_biomedical),
                        performance_dfs=performance_tables_latex,
                        performance_plots_and_captions=zip(performance_plots, performace_plots_captions)
                    )
    
    dat = date.today().strftime("%d/%m/%Y")

    if ANALYZING_FASTTEXT_MODELS:
        with open('../fasttext/ft_hyperparameter_report_{}.{}.tex'.format(dat[0:2], dat[3:5]), 'w') as f:
            f.write(report_latex)
    
    else:
        with open('./w2v_hyperparameter_report_{}.{}.tex'.format(dat[0:2], dat[3:5]), 'w') as f:
            f.write(report_latex)

    print('END!')