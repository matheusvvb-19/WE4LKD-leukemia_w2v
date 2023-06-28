##################################################
## Generates a LaTex file containing the historical plots of each compound, for each embeddings approach, and each algebric operation applied on data (normalization, standartization, softmax).
##################################################
## Author: Matheus Vargas Volpon Berto
## Copyright: Copyright 2022, Discovering Latent Knowledge in medical paper on Acute Myeloid Leukemia
## Email: matheusvvb@hotmail.com
##################################################

# IMPORTS:
import os, jinja2, re, math
import pandas as pd
import numpy as np
from gensim import models
from gensim.models import Word2Vec, FastText
from matplotlib import pyplot as plt
from jinja2 import Template
from datetime import date
import tikzplotlib
from tikzplotlib import get_tikz_code
from itertools import repeat
sys.path.append('./pubchem/')
from clean_summaries import get_target_compounds
from generates_dotproducts_csv import get_w2v_output_embedding

# FUNCTIONS:
def get_dot_products_plot_BERT(folder_path, column='dot_product_result_absolute'):
    """Plots a historical record for the selected compounds.
    
    Args:
        compounds: the name of the compounds to be plotted;
        folder_path: a list of filenames;
        column: the column to be historically plotted. The possbile values are:
            dot_product_result
            dot_product_result_absolute
            softmax
            normalized_dot_product_absolute
            standartized_dot_product_absolute
            softmax_normalization
            softmax_standartization
    """
    
    compounds = get_target_compounds()
    latent_knowledge_compounds = []
    indexes_to_put_marker = []
    
    temporal = {
        'compound': [],
        'year': [],
        column: [],
    }

    for c in folder_path:        
        compound = c.split('.csv')[0].split('/')[-1].split('_')[0]

        df = pd.read_csv(c)
        values = df[column].tolist()

        if len(values) < 60:
            first_year = df['year'].to_list()[0]
            #last_year = df['year'].to_list()[-1]

            if first_year != 1963:
                values = list(repeat(None, first_year - 1963)) + values
            
            #if last_year != 2022: # remover isso quando irpara produção
                #values = values + list(repeat(None, 2022 - last_year))

        temporal['compound'].extend([compound] * 60)
        temporal['year'].extend([x for x in range(1963, 2023)])
        temporal[column].extend(values)

    temporal_df = pd.DataFrame.from_dict(temporal)

    dfs = []
    for c in compounds:
        df_aux = temporal_df[temporal_df['compound'] == c]
        dfs.append(df_aux)

        try:
            df_aux.dropna(inplace=True)
            indexes_to_put_marker.append(df_aux['year'].to_list().index(int(timeline[c])) + 1)
        
        except:
            indexes_to_put_marker.append(None)

    #print(indexes_to_put_marker)
    fig, axs = plt.subplots(int(len(compounds)/2)+1, 2, sharex='all', figsize=(20, 30))
    
    i = 0
    for row in range(0, int(len(compounds)/2)+1):
        for col in range(0, 2):
            if i >= len(compounds):
                break

            if i == len(compounds) - 2:
                axs[row, col].tick_params(axis='x', labelbottom=True)
            
            dp_values = dfs[i][column].to_list()
            marker_on = [False for x in range(1963, 2023)]
            for index, y in enumerate(range(1963, 2023)):
                if str(y) == str(timeline[compounds[i]]):
                    marker_on[index] = True

            """
            for index_dp, dp in enumerate(dp_values):
                try:
                    if math.isnan(dp) == False and index_dp < int(timeline[compounds[i]]) and dp > dp_values[marker_on.index(True)]:
                        print('{} is bigger than {}'.format(dp, dp_values[marker_on.index(True)]))
                        latent_knowledge_compounds.append(compounds[i])
                        break
                except:
                    continue
            """

            axs[row, col].plot('year', column, data=dfs[i])
            #axs[row].set_xlabel('Years')
            #axs[row].set_ylabel(' '.join(column.split('_')).capitalize())
            axs[row, col].grid(visible=True)

            #if compounds[i] in latent_knowledge_compounds:
                #axs[row, col].set_title(compounds[i].capitalize(), fontdict={'fontweight': 'bold'}, color='red')
            
            #else:
                #axs[row, col].set_title(compounds[i].capitalize())
            axs[row, col].set_title(compounds[i].capitalize())

            #axs[row].setylim(compounds[i].capitalize())
            axs[row, col].set_xlim(1962, 2023)
            axs[row, col].tick_params(axis='x', labelrotation=45)
            #axs[row, col].tick_params(axis='both', labelbottom=True)
            axs[row, col].set_xticks([x for x in range(1963, 2023, 3)])  
            
            i = i + 1

    #plt.tight_layout(h_pad=3)

    #plt.xlabel("common X")
    #plt.ylabel("common Y")
    axs[int(len(compounds)/2), 1].set_axis_off()
    fig.tight_layout(pad=3)
    fig.supxlabel('Years', y=-0.03, fontsize=24)
    fig.supylabel(' '.join(column.split('_')).capitalize(), x=-0.03, fontsize=24)

    sep = r"\linewidth/12"
    latex_string = get_tikz_code(fig,
                                 axis_width='450',
                                 axis_height='125',
                                 extra_axis_parameters=[
                                    'xtick={1963, 1966, 1969, 1972, 1975, 1978, 1981, 1984, 1987, 1990, 1993, 1996, 1999, 2002, 2005, 2008, 2011, 2014, 2017, 2020},\
                                    x tick label style={/pgf/number format/1000 sep=},\
                                    y tick label style={\
                                            /pgf/number format/fixed,\
                                            /pgf/number format/precision=15\
                                    },\
                                    scaled y ticks=false'],
                                 extra_groupstyle_parameters={f'horizontal sep={sep}'}
                    )

    for lkc in latent_knowledge_compounds:
        latex_string = re.sub('title={'+ lkc.capitalize() + '}',  'title={'+ lkc.capitalize() + '},\ntitle style={color=red}', latex_string)
    
    for idx in indexes_to_put_marker:
        if idx is not None:
            latex_string = re.sub('\[semithick,[ ]?steelblue31119180\]', '[semithick, steelblue31119180, mark=star, mark size=4, mark indices={' + str(idx) +'}]', latex_string, count=1)
        
        else:
            latex_string = re.sub('\[semithick,[ ]?steelblue31119180\]', '[semithick, steelblue31119181]', latex_string, count=1)

    latex_string = re.sub('steelblue31119181', 'steelblue31119180', latex_string)
    return latex_string

# MAIN PROGRAM:
if __name__ == '__main__':
    pd.options.mode.chained_assignment = None
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
    template = latex_jinja_env.get_template('./latent_knowledge_template.tex')

    # CONSTANTS:
    timeline = {
        'cytarabine': '1968',
        'daunorubicin': '1968',
        'azacitidine': '1978',
        'gemtuzumab-ozogamicin': '1999',
        'midostaurin': '2002',
        'vyxeos': '2010',
        'ivosidenib': '2012',
        'venetoclax': '2014',
        'enasidenib': '2015',
        'gilteritinib': '2015',
        'glasdegib': '2015',
        'arsenictrioxide': '1996',
        'cyclophosphamide': '1970',
        'dexamethasone': '1977',
        'idarubicin': '1984',
        'mitoxantrone': '1983',
        'pemigatinib': '2019',
        'prednisone': '1975',
        'rituximab': '2000',
        'thioguanine': '1972',
        'vincristine': '1964',
    }
    
    DOT_PRODUCTS_PER_COMPOUND_BERT_FIRST_SUBWORD = [x.path for x in os.scandir('./validation/per_compound/bert/') if x.name.endswith('first.csv')]
    DOT_PRODUCTS_PER_COMPOUND_BERT_LAST_SUBWORD = [x.path for x in os.scandir('./validation/per_compound/bert/') if x.name.endswith('last.csv')]
    DOT_PRODUCTS_PER_COMPOUND_BERT_MEAN_SUBWORD = [x.path for x in os.scandir('./validation/per_compound/bert/') if x.name.endswith('mean.csv')]
    DOT_PRODUCTS_PER_COMPOUND_W2V_COMB15 = [x.path for x in os.scandir('./validation/per_compound/w2v/') if x.name.endswith('_comb15.csv')]
    DOT_PRODUCTS_PER_COMPOUND_FT_COMB16 = [x.path for x in os.scandir('./validation/per_compound/ft/') if x.name.endswith('_comb16.csv')]

    # BERT MODELS:
    print('Generating plots for BERT-based models, first subword method')
    # first subword:
    plot_first_subword_dot_product_result_absolute = get_dot_products_plot_BERT(DOT_PRODUCTS_PER_COMPOUND_BERT_FIRST_SUBWORD, column='dot_product_result_absolute')
    plot_first_subword_softmax = get_dot_products_plot_BERT(DOT_PRODUCTS_PER_COMPOUND_BERT_FIRST_SUBWORD, column='softmax')
    plot_first_subword_softmax_normalization = get_dot_products_plot_BERT(DOT_PRODUCTS_PER_COMPOUND_BERT_FIRST_SUBWORD, column='softmax_normalization')
    plot_first_subword_softmax_standartization = get_dot_products_plot_BERT(DOT_PRODUCTS_PER_COMPOUND_BERT_FIRST_SUBWORD, column='softmax_standartization')

    print('Generating plots for BERT-based models, last subword method')
    # last subword:
    plot_last_subword_dot_product_result_absolute = get_dot_products_plot_BERT(DOT_PRODUCTS_PER_COMPOUND_BERT_LAST_SUBWORD, column='dot_product_result_absolute')
    plot_last_subword_softmax = get_dot_products_plot_BERT(DOT_PRODUCTS_PER_COMPOUND_BERT_LAST_SUBWORD, column='softmax')
    plot_last_subword_softmax_normalization = get_dot_products_plot_BERT(DOT_PRODUCTS_PER_COMPOUND_BERT_LAST_SUBWORD, column='softmax_normalization')
    plot_last_subword_softmax_standartization = get_dot_products_plot_BERT(DOT_PRODUCTS_PER_COMPOUND_BERT_LAST_SUBWORD, column='softmax_standartization')

    print('Generating plots for BERT-based models, mean subword method')
    # mean subword:
    plot_mean_subword_dot_product_result_absolute = get_dot_products_plot_BERT(DOT_PRODUCTS_PER_COMPOUND_BERT_MEAN_SUBWORD, column='dot_product_result_absolute')
    plot_mean_subword_softmax = get_dot_products_plot_BERT(DOT_PRODUCTS_PER_COMPOUND_BERT_MEAN_SUBWORD, column='softmax')
    plot_mean_subword_softmax_normalization = get_dot_products_plot_BERT(DOT_PRODUCTS_PER_COMPOUND_BERT_MEAN_SUBWORD, column='softmax_normalization')
    plot_mean_subword_softmax_standartization = get_dot_products_plot_BERT(DOT_PRODUCTS_PER_COMPOUND_BERT_MEAN_SUBWORD, column='softmax_standartization')

    # WORD2VEC MODELS:
    print('Generating plots for Word2Vec models, combination 15')
    plot_comb15_dot_product_result_absolute = get_dot_products_plot_BERT(DOT_PRODUCTS_PER_COMPOUND_W2V_COMB15, column='dot_product_result_absolute')
    plot_comb15_softmax = get_dot_products_plot_BERT(DOT_PRODUCTS_PER_COMPOUND_W2V_COMB15, column='softmax')
    plot_comb15_softmax_normalization = get_dot_products_plot_BERT(DOT_PRODUCTS_PER_COMPOUND_W2V_COMB15, column='softmax_normalization')
    plot_comb15_softmax_standartization = get_dot_products_plot_BERT(DOT_PRODUCTS_PER_COMPOUND_W2V_COMB15, column='softmax_standartization')

    # FASTTEXT MODELS:
    print('Generating plots for FastText models, combination 16')
    plot_comb16_dot_product_result_absolute = get_dot_products_plot_BERT(DOT_PRODUCTS_PER_COMPOUND_FT_COMB16, column='dot_product_result_absolute')
    plot_comb16_softmax = get_dot_products_plot_BERT(DOT_PRODUCTS_PER_COMPOUND_FT_COMB16, column='softmax')
    plot_comb16_softmax_normalization = get_dot_products_plot_BERT(DOT_PRODUCTS_PER_COMPOUND_FT_COMB16, column='softmax_normalization')
    plot_comb16_softmax_standartization = get_dot_products_plot_BERT(DOT_PRODUCTS_PER_COMPOUND_FT_COMB16, column='softmax_standartization')

    print('Generating .tex file')
    report_latex = template.render(
                        # BERT-based:
                        plot_first_subword_dot_product_result_absolute=plot_first_subword_dot_product_result_absolute,
                        plot_first_subword_softmax=plot_first_subword_softmax,
                        plot_first_subword_softmax_normalization=plot_first_subword_softmax_normalization,
                        plot_first_subword_softmax_standartization=plot_first_subword_softmax_standartization,
                        plot_last_subword_dot_product_result_absolute=plot_last_subword_dot_product_result_absolute,
                        plot_last_subword_softmax=plot_last_subword_softmax,
                        plot_last_subword_softmax_normalization=plot_last_subword_softmax_normalization,
                        plot_last_subword_softmax_standartization=plot_last_subword_softmax_standartization,
                        plot_mean_subword_dot_product_result_absolute=plot_mean_subword_dot_product_result_absolute,
                        plot_mean_subword_softmax=plot_mean_subword_softmax,
                        plot_mean_subword_softmax_normalization=plot_mean_subword_softmax_normalization,
                        plot_mean_subword_softmax_standartization=plot_mean_subword_softmax_standartization,
                        # Word2Vec combination 15:
                        plot_comb15_dot_product_result_absolute=plot_comb15_dot_product_result_absolute,
                        plot_comb15_softmax=plot_comb15_softmax,
                        plot_comb15_softmax_normalization=plot_comb15_softmax_normalization,
                        plot_comb15_softmax_standartization=plot_comb15_softmax_standartization,
                        # FastText combination 16:
                        plot_comb16_dot_product_result_absolute=plot_comb16_dot_product_result_absolute,
                        plot_comb16_softmax=plot_comb16_softmax,
                        plot_comb16_softmax_normalization=plot_comb16_softmax_normalization,
                        plot_comb16_softmax_standartization=plot_comb16_softmax_standartization
                    )
    
    dat = date.today().strftime("%d/%m/%Y")

    with open('./latent_knowledge_report_{}.{}.tex'.format(dat[0:2], dat[3:5]), 'w') as f:
        f.write(report_latex)
    
    print('END!')
