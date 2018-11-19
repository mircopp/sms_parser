import requests
import time
import progressbar
import pandas as pd

def annotate_data(data):
    """
    Annotates the sentences using syntaxnet as a RESTful API.
    :param data: The sentences to be annotated
    :return: The parsed sentences and compressions
    """
    sentences = []
    start = time.time()
    print("Started")

    LEN_DATA = len(data)
    p_bar = progressbar.ProgressBar(max_value=(LEN_DATA))

    for i in range(len(data)):
        sentences.append(get_annotation(data[i]))
        p_bar.update(i + 1)

    end = time.time()
    print('Elapsed time:\t {}:{}'.format(int((end - start) / 60), int((end - start) % 60)))
    return sentences



def get_annotation(sentence):
    """
    Get the annotation for a single sentence.
    :param sentence: The sentence to be parsed.
    :return: The parsed list of word tokens.
    """
    sen_payload = {'sentence': sentence}
    req = requests.post('http://localhost:4000/parse', json=sen_payload)
    sen_tokens = req.json()
    return sen_tokens


def is_punctuation_mark(word):
    """
    Checks whether a word is a punctuation mark.
    :param word: The word to be checked.
    :return: True if the word is a punctuation mark, false otherwise.
    """
    return (word['dependency_label'] == 'punct') and (word['pos'] != 'HYPH') and (word['word'] != '<EOS>')

def filter_punctuation_marks(sequence):
    """
    Filters the punctuation mark of a sequence of parsed tokens.
    :param sequence: The parsed sequence to be filtered.
    :return: The filtered sequence.
    """
    result = []
    sequence.append({'word': '<EOS>', 'pos': '.', 'dependency_label': 'punct',
                   'id': sequence[len(sequence) - 1]['id'] if len(sequence) > 0 else 0,
                   'parent': sequence[len(sequence) - 1]['parent'] if len(sequence) > 0 else -1})
    for tmp in sequence:
        if is_punctuation_mark(tmp):
            continue
        else:
            result.append(tmp)
    return result

def generate_matrix(sentence):
    """
    Transforms given sentence into a matrix.
    :param sentence: The original sentence as a sequence of word tokens.
    :return: The labelled feature matrix
    """

    result = []
    for i in range(len(sentence)):
        result.append([sentence[i]['word'], sentence[i]['pos'], sentence[i]['dependency_label'], sentence[i]['id'],
                       sentence[i]['parent']])
    return result


def build_matrizes(sentences):
    """
    Build feature matrix for sentences and compressions as labelled data.
    :param sentences: The sentences.
    :return: A feature matrix
    """
    dropped = 0
    res = []
    for sentence in sentences:
        sen_matrix = generate_matrix(sentence)
        if sen_matrix:
            res.append(sen_matrix)
        else:
            dropped += 1
    print('Dropped {} sentences'.format(dropped))
    return res


def build_dataframe(header_row, matrizes):
    """
    Transform the data into a csv DataFrame.
    :param header_row: The column types
    :param matrizes: The sequence matrix
    :return: The transformed DataFrame
    """
    data = []
    for sentence in matrizes:
        for i in range(len(sentence)):
            if i == (len(sentence) - 1):
                eos = 1
            else:
                eos = 0
            data.append(
                [sentence[i][0], sentence[i][1], sentence[i][2], sentence[i][3], sentence[i][4], eos])
    indexes = list(range(len(data)))
    dataframe = pd.DataFrame(data=data, index=indexes, columns=header_row)
    return dataframe
