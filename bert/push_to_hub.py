##################################################
## Uploads trained Distilbert language models to Hugging Face Models Hub.
##################################################
## Author: Matheus Vargas Volpon Berto
## Copyright: Copyright 2022, Discovering Latent Knowledge in medical paper on Acute Myeloid Leukemia
## Email: matheusvvb@hotmail.com
## Based on: https://huggingface.co/docs/transformers/model_sharing#use-the-pushtohub-function
##################################################

import os
from transformers import DistilBertForMaskedLM
from transformers import AutoTokenizer

if __name__ == '__main__':
    # CONSTANT(S):
    MODELS = sorted([f.path for f in os.scandir('./distilbert/') if f.is_dir()])
    
    # variable(s):
    model_name_template = 'doubleblind'

    for m in MODELS:
        model = DistilBertForMaskedLM.from_pretrained(m, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(m, local_files_only=True)

        print('Pushing model {} to hub\n'.format(m))
        model.push_to_hub(model_name_template + m[-4:])
        tokenizer.push_to_hub(model_name_template + m[-4:])
