import json
import pandas as pd
import numpy as np
from libs.tokenization import annotate_data, build_matrizes, build_dataframe

def read_google_data(data_source):
    data_string = open(data_source, 'rb').read().decode('utf-8')
    sentences = data_string.split('\n\n')
    print('Number of sentences:', len(sentences))
    json_objects = []
    for chunk in sentences:
        try:
            obj = json.loads(chunk)
            current = {'sentence': obj['graph']['sentence'], 'compression': obj['compression']['text']}
            json_objects.append(current)
        except json.decoder.JSONDecodeError:
            print('Found decode error:', chunk)
            continue
    return json_objects

def read_data(data_source):
    data_file = open(data_source, 'rb')
    corpus = pd.read_csv(data_file, sep='\t', encoding='utf-8')
    return corpus

if __name__ == '__main__':
    try:
        data_source = 'master_corpus.txt'
        corpus = read_data('ressources/' + data_source)
        sentences = corpus['message_body'].values
        tokens = annotate_data(sentences)
        X = build_matrizes(tokens)
        header_row = ['word', 'pos', 'dependency_label', 'id', 'parent', 'EOS']
        df = build_dataframe(header_row, X)
        df.to_csv('result/' + data_source + '.csv', sep=',', encoding='utf-8', na_rep='N/A')
        print('Totally saved:{} text messages'.format(len(X)))

    except FileNotFoundError:
        print('TODO add ressource file')
