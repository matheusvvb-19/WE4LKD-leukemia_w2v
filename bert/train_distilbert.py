##################################################
## Trains incremental Distilbert language models from a preprocessed corpus in a csv file.
##################################################
## Author: {name}
## Copyright: Copyright 2022, Discovering Latent Knowledge in medical paper on Acute Myeloid Leukemia
## Email: {contact_email}
## Based on: https://huggingface.co/course/chapter7/6?fw=tf
##################################################

import os
from os import listdir
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import DistilBertConfig, DistilBertForMaskedLM
from datasets import load_dataset, Features, Value
from pathlib import Path

def tokenize(element):
    outputs = tokenizer(
        element["summary"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )

    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)

    return {"input_ids": input_batch}

def find_csv_filenames(path_to_dir, suffix=".csv"):
    """ Finds files with a determined suffix inside a folder.
    
    Args:
        parth_to_dir: path to the directory where the files will be searched;
        suffix: the suffix od the file(s) to be retrieved.
    
    Returns:
        a list containing all the files inside the folder that contains the wanted suffix.
    """
    
    filenames = listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]

context_length = 150
tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')

if __name__ == '__main__':
    # CONSTANT(s):
    CLEANED_DOCUMENTS_PATH = './results/'
    TRAINED_MODELS_PATH = './distilbert/'
    
    # variable(s):
    csv_files = find_csv_filenames(CLEANED_DOCUMENTS_PATH)
    assert len(csv_files) == 1
    Path(TRAINED_MODELS_PATH).mkdir(exist_ok=True)

    # reading the csv file containg the preprocessed text into a Hugging Face Dataset element:
    features = Features({'summary': Value('string'), 'filename': Value('string'), 'id': Value('int32')})
    dataset = load_dataset('csv', data_files=CLEANED_DOCUMENTS_PATH + csv_files[0], delimiter='|', escapechar='\\', encoding='utf-8', column_names=['summary', 'filename', 'id'], header=0, features=features)
    
    # computing timespan :
    # assuming that the first collected article was published in 1921, and that more articles were published after this one in the following years, we have ranges equal to:
    # [[1921], [1921, 1922], [1921, 1922, 1923], [1921, 1922, 1923, 1924], [1921, 1922, 1923, 1924, 1925], .......]
    years = sorted(dataset['train'].unique('filename'))
    ranges = [years[:i+1] for i in range(len(years))]
    ranges = ranges[-21:]

    # using a data collator for Maked Language Model (MLM):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True
    )

    # retrieving Distilbert configurations to be applied in the Maked Language Models trained:
    configuration = DistilBertConfig()
    model = DistilBertForMaskedLM(configuration)

    # trainig each incremental model:
    for r in ranges:
        print(r)
        
        target_folder = TRAINED_MODELS_PATH + 'distilbert_model_{}_{}/'.format(r[0], r[-1])

        # auxiliar dataset containing only the abstracts published befor or on the actual timespan:
        aux_dataset = dataset['train'].filter(lambda x: x['filename'] in r)

        # tokenizing text:
        tk_dataset = aux_dataset.map(
            tokenize, batched=True, remove_columns=aux_dataset.column_names
        )

        # training arguments:
        training_args = TrainingArguments(
            output_dir=target_folder,
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_device_train_batch_size=32,
            save_steps=10_000,
            save_total_limit=1,
            do_train=True,
            prediction_loss_only=True,
            report_to='none',
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tk_dataset,
        )

        trainer.train()
        trainer.save_model(target_folder)
        tokenizer.save_pretrained(target_folder)
        print('\n')
