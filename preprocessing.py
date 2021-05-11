import os
import re
import pandas as pd

data_dir = os.path.join(os.getcwd(), "Dataset")

def read_data(f_name):
    data = pd.read_csv(os.path.join(data_dir, f_name))
    sentences = [clean_str(s) for s in data["Sentence"]]
    categories = data["Category"]
    return sentences, categories
    # return data

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

class Tokenizer():
    def __init__(self, tokenizer='whitespace', clean_string=True):
        self.clean_string = clean_string
        tokenizer = tokenizer.lower()

        # Tokenize with whitespace
        if tokenizer == 'whitespace':
            print('Loading whitespace tokenizer')
            self.tokenize = lambda string: string.strip().split()

        if tokenizer == 'regex':
            print('Loading regex tokenizer')
            import re
            pattern = r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+"
            self.tokenize = lambda string: re.findall(pattern, string)

        if tokenizer == 'spacy':
            print('Loading SpaCy')
            import spacy
            nlp = spacy.load('en')
            self.tokenize = lambda string: [token.text for token in nlp(string)]

        # Tokenize with punctuations other than periods
        if tokenizer == 'nltk':
            print('Loading NLTK word tokenizer')
            from nltk import word_tokenize

            self.tokenize = word_tokenize

    def __call__(self, string):
        if self.clean_string:
            string = clean_str(string)
        return self.tokenize(string)


if __name__ == '__main__':
    tokenizer = Tokenizer()
    sentences, _ = read_data('train_final.csv')
    samples = sentences[:5]
    for sample in samples:
        print(tokenizer(sample))

    tokenizer = Tokenizer("regex")
    for sample in samples:
        print(tokenizer(sample))
    # f_o = open(os.path.join(os.getcwd(), "tokenized_test.txt"), "w")
    # for sentence in sentences:
    #     w_line = " ".join(tokenizer(sentence))
    #     f_o.write(w_line+"\n")
    #     # print(tokenizer(sentence))
    # f_o.close()