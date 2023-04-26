##################################################
## Generates a csv file for the CID-Titles and CID-Synonyms DataFrames.
##################################################
## Author: {name}
## Copyright: Copyright 2022, Discovering Latent Knowledge in medical paper on Acute Myeloid Leukemia
## Email: {contact_email}
##################################################

# IMPORTS:
from pubchem.clean_summaries import ss, read_csv_table_files, words_preprocessing, summary_column_preprocessing, to_csv
import pyspark.sql.functions as F
import pyspark.sql.types as T

# MAIN PROGRAM:
if __name__ == '__main__':
    spark = ss()

    # Carregue o arquivo CID-Title.gz
    rdd = spark.sparkContext.textFile("CID-Title.gz")

    # Divida as linhas por tab para separar cid e title
    rdd = rdd.map(lambda line: line.split("\t"))

    # Crie um DataFrame a partir do RDD e renomeie as colunas
    df = rdd.toDF(["cid", "title"])

    # Converta a coluna "cid" em StringType
    df = df.withColumn("cid", F.col("cid").cast(T.StringType()))

    df.to_csv(df, './titles/', sep='|')
    
    # Carregue o arquivo CID-Synonym-filtered.gz
    rdd = spark.sparkContext.textFile("CID-Synonym-filtered.gz")

    # Divida as linhas por tab para separar cid e title
    rdd = rdd.map(lambda line: line.split("\t"))

    # Crie um DataFrame a partir do RDD e renomeie as colunas
    df = rdd.toDF(["cid", "synonym"])

    # Converta a coluna "cid" em StringType
    df = df.withColumn("cid", F.col("cid").cast(T.StringType()))

    df.to_csv(df, './synonyms/', sep='|')

    print('END!')
