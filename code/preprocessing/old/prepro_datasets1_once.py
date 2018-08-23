import pickle
import argparse
import time
#import datetime
from collections import defaultdict

#import sys
#import os
#sys.path.append(os.path.abspath('../aux'))


word_freq = defaultdict(int)
char_freq = defaultdict(int)    # how many times each character is encountered
p_e_m_position = defaultdict(int)  # for the gold mentions where the ground truth appears in
                                   # the p_e_m dictionary
list_of_p_e_m_position = []

def vocabulary_count_wiki():

    doc_cnt = 0
    #with open(base_folder + "data/mydata/tokenized_toy_wiki_dump.txt") as fin:
    with open(config.base_folder+"data/basic_data/tokenizedWiki.txt") as fin:
        for line in fin:
            if line.startswith('</doc>'):
                doc_cnt += 1
                if doc_cnt % 5000 == 0:
                    print("document counter: ", doc_cnt)
                continue
            if line.startswith('<a\xa0href="') or line.startswith('<doc\xa0id="') or \
                    line.startswith('</a>') or line.startswith('*NL*'):
                continue

            line = line[:-1]       # omit the '\n' character
            word_freq[line.lower() if config.lowercase_emb else line] += 1
            for c in line:
                char_freq[c] += 1

    # print some statistics
    print("some frequency statistics. The bins are [...) ")
    for d, name, edges in zip([word_freq, char_freq], ["word", "character"],
                              [[1, 2, 3, 6, 11, 21, 31, 51, 76, 101, 201, np.inf],
                               [1, 6, 11, 21, 51, 101, 201, 501, 1001, 2001, np.inf]]):
        hist_values, _ = np.histogram(list(d.values()), edges)
        cum_sum = np.cumsum(hist_values[::-1])
        print(name, " frequency histogram, edges: ", edges)
        print("absolute values:        ", hist_values)
        print("absolute cumulative (right to left):    ", cum_sum[::-1])
        print("probabilites cumulative (right to left):", (cum_sum / np.sum(hist_values))[::-1])

    with open(config.base_folder+"data/vocabulary/frequencies{}.pickle".format("_lowercase"
                                                                               if config.lowercase_emb else ""), 'wb') as handle:
        pickle.dump((word_freq, char_freq), handle)


def process_aida_aux(filepath):
    with open(filepath) as fin:
        line_cnt = 0
        for line in fin:
            line = line.rstrip()
            if line.startswith("-DOCSTART-"):
                continue
            #if line == "":
            #    print("*NL*")

            line = line.split('\t')

            word_freq[line.lower() if args.lowercase_emb else line] += 1
            for c in line:
                char_freq[c] += 1

            line_cnt += 1
            if line_cnt == 30:
                break



def process_aida():
    process_aida_aux(datasets+"AIDA/aida_train.txt")
    process_aida_aux(datasets+"AIDA/testa_testb_aggregate_original")


def process_other_datasets(folder_path):
    pass





def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="/home/master_thesis_share/data/basic_data/test_datasets/")
    parser.add_argument("--lowercase_emb", type=bool, default=False,
                        help="if true then lowercase word for counting and embedding")
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    datasets = args.datasets

    process_aida()
    '''
    process_other_datasets(datasets+"/wned-datasets/ace2004/")
    process_other_datasets(datasets+"/wned-datasets/aquaint/")
    process_other_datasets(datasets+"/wned-datasets/clueweb/")
    process_other_datasets(datasets+"/wned-datasets/msnbc/")
    process_other_datasets(datasets+"/wned-datasets/wikipedia/")
    '''







