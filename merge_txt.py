##################################################
## Groups together each singular abstract text file into one single file.
##################################################
## Author: {name}
## Copyright: Copyright 2022, Discovering Latent Knowledge in medical paper on Acute Myeloid Leukemia
## Email: {contact_email}
##################################################

# IMPORTS:
from pathlib import Path
import os

# MAIN PROGRAM:
if __name__ == '__main__':
    filenames = [str(x) for x in Path('./results/').glob('**/*.txt')]
    initial_year = 1900
    years = [2023]
    os.makedirs(os.path.dirname('./results_aggregated'), exist_ok=True)

    for y in years:
        print(y)
        
        list_years = [str(i) for i in range(initial_year, y)]

        filtered_filenames = []
        for f in filenames:
            if f.split('/')[2].startswith(tuple(list_years)):
                filtered_filenames.append(f)

        print('number of abstracts between {} and {}: {}'.format(initial_year, y, len(filtered_filenames)))
        abstract_list = []
        for fname in filtered_filenames:
            with open(fname, encoding='utf-8') as infile:
                for line in infile:
                    abstract_list.append(line)

        Path('./results_aggregated/results_file_{}_{}.txt'.format(initial_year, y-1)).touch()
        with open('results_aggregated/results_file_{}_{}.txt'.format(initial_year, y-1),'w+', encoding='utf-8') as f:
            for row in abstract_list:
                f.write(repr(row)+'\n')
