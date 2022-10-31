####################################################

# Este script realiza o treinamento dos modelos Distilbert "from scratch" a partir dos prefácios dos artigos.
# Os modelos são gerados ano a ano (de forma incremental), da mesma forma que os modelos Word2Vec.

####################################################
# based on: https://huggingface.co/course/chapter7/6?fw=tf

from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import DistilBertConfig, DistilBertForMaskedLM

from datasets import load_dataset, Features, Value

import os
from os import listdir

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
    filenames = listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]

context_length = 150
tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')

if __name__ == '__main__':
    # variáveis
    CLEANED_DOCUMENTS_PATH = './results/'
    TRAINED_MODELS_PATH = './distilbert/'
    csv_files = find_csv_filenames(CLEANED_DOCUMENTS_PATH)

    assert len(csv_files) == 1
    Path(TRAINED_MODELS_PATH).mkdir(exist_ok=True)


    # realizando a leitura do arquivo .csv proveniente da limpeza/pré-processamento do texto:
    features = Features({'summary': Value('string'), 'filename': Value('string'), 'id': Value('int32')})
    dataset = load_dataset('csv', data_files=CLEANED_DOCUMENTS_PATH + csv_files[0], delimiter='|', escapechar='\\', encoding='utf-8', column_names=['summary', 'filename', 'id'], header=0, features=features)
    
    # calculando os intervalos/janelas de tempo que cada modelo irá abrangir:
    # supondo que o primeiro artigo coleto foi publicado em 1921, e que, depois deste, mais artigos foram publicados nos anos seguintes, temos ranges igual a:
    # [[1921], [1921, 1922], [1921, 1922, 1923], [1921, 1922, 1923, 1924], [1921, 1922, 1923, 1924, 1925], .......]
    years = sorted(dataset['train'].unique('filename'))
    ranges = [years[:i+1] for i in range(len(years))]
    ranges = ranges[-21:]

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True
    )

    # dados do modelo a ser treinado:
    configuration = DistilBertConfig()
    model = DistilBertForMaskedLM(configuration)

    for r in ranges:
        print(r)
        # definindo pasta para escrita/armazenamento do modelo:
        target_folder = TRAINED_MODELS_PATH + 'distilbert_model_{}_{}/'.format(r[0], r[-1])

        # filtrando dataset original, selecionando apenas os artigos dentro da janela de tempo que está sendo processada:
        aux_dataset = dataset['train'].filter(lambda x: x['filename'] in r)

        # fazendo a tokenização do dataset auxiliar:
        tk_dataset = aux_dataset.map(
            tokenize, batched=True, remove_columns=aux_dataset.column_names
        )

        # argumentos de treinamento:
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
