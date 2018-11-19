import json
import pandas as pd
from libs.tokenization import annotate_data, build_matrizes, build_dataframe

def read_message_corpus(data_source):
    data_file = open(data_source, 'rb')
    corpus = pd.read_csv(data_file, sep='\t', encoding='utf-8')
    return corpus

if __name__ == '__main__':
    try:
        data_source = 'master_corpus.txt'
        corpus = read_message_corpus('ressources/' + data_source)
        sentences = corpus['message_body'].values
        related_types = corpus['type'].values
        related_classes = corpus['class'].values
        related_timestamps = corpus['timestamp'].values
        related_subscribers = corpus['subscriber'].values
        tokens = annotate_data(sentences)
        X = build_matrizes(tokens)
        for toks, rel_sen, rel_class, rel_ts, rel_subscriber, rel_type in zip(X, sentences, related_classes, related_timestamps, related_subscribers, related_types):
            for tok in toks:
                tok.extend([rel_sen, rel_type, rel_ts, rel_subscriber, rel_class])
        header_row = ['Word', 'POS', 'Dependency label', 'ID', 'Parent ID', 'Related Sentence', 'Related Message Type', 'Related Message Timestamp', 'Related Message Subscriber', 'Related Message Class', 'EOS']
        df = build_dataframe(header_row, X)
        df.to_csv('result/' + data_source + '.csv', sep=',', encoding='utf-8', na_rep='N/A')
        print('Totally saved:{} text messages'.format(len(X)))

    except FileNotFoundError:
        print('TODO add ressource file')
