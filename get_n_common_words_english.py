from urllib.request import urlopen
from bs4 import BeautifulSoup

def get_most_common(n=10000):
    
    #from https://norvig.com/ngrams/
    url = "https://norvig.com/ngrams/count_1w.txt"
    html = urlopen(url).read()
    soup = BeautifulSoup(html, features="html.parser")

    # remove script and style
    for script in soup(["script", "style"]):
        script.extract()

    # get text
    text = soup.get_text()
    
    
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    text_lines = text.split('\n')
    
    text_first_n = text_lines[:n]
    
    list_n = []

    for t in text_first_n:
        list_n.append(t.split('\t')[0])
        
    return list_n
