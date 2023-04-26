##################################################
## Preprocess the synonyms table (PubChem).
##################################################
## Author: {name}
## Copyright: Copyright 2022, Discovering Latent Knowledge in medical paper on Acute Myeloid Leukemia
## Email: {contact_email}
##################################################

# IMPORTS:
from pubchem.clean_summaries import read_csv_table_files, words_preprocessing, summary_column_preprocessing, to_csv
import pyspark.sql.functions as F

# MAIN PROGRAM:
if __name__ == '__main__':
    synonyms = read_csv_table_files('/home/matheus/synonyms_table.csv')\
                .withColumn('synonym', summary_column_preprocessing(F.col('synonym')))
                
    to_csv(
        words_preprocessing(synonyms, column='synonym'), 
        target_folder='/home/matheus/WE4LKD-leukemia_w2v/pubchem/synonyms/'
    )

    print('END!')
