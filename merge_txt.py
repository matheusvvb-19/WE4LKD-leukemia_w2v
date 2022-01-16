from pathlib import Path
import os


filenames = [str(x) for x in Path('./results/').glob('**/*.txt')]

initial_year = 1900

years = [1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025]

for y in years:
    print(y)
    #filter_by_year
    list_years = [str(i) for i in range(initial_year,y)]

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

    os.makedirs(os.path.dirname('./results_aggregated'), exist_ok=True)

    with open('results_aggregated/results_file_{}_{}.txt'.format(initial_year, y-1),'w+', encoding='utf-8') as f:
        for row in abstract_list:
            f.write(repr(row)+'\n')
