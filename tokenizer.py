import re

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9,!?\'\`\.]", " ", string)
    string = re.sub(r"\.{3}", " ...", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

class Tokenizer(): # select one to use 
    def __init__(self, tokenizer='whitespace', clean_string=True):
        self.clean_string = clean_string
        tokenizer = tokenizer.lower()

        # Tokenize with whitespace
        if tokenizer == 'whitespace':
            self.tokenize = lambda string: string.strip().split()

        if tokenizer == 'regex':
            import re
            pattern = r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+"
            self.tokenize = lambda string: re.findall(pattern, string)

        if tokenizer == 'spacy':
            import spacy
            nlp = spacy.load('en')
            self.tokenize = lambda string: [token.text for token in nlp(string)]

        # Tokenize with punctuations other than periods
        if tokenizer == 'nltk':
            from nltk import word_tokenize

            self.tokenize = word_tokenize

    def __call__(self, string):
        if self.clean_string:
            string = clean_str(string)
        return self.tokenize(string)