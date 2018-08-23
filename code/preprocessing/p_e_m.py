import pickle
from collections import defaultdict
from nltk.tokenize import word_tokenize
import time
import sys
import os
import operator

import preprocessing.config as config
import preprocessing.util as util


# TODO delete this one, no longer usefull since there is another one that also handles the conflicts
def tokenize_p_e_m():
    '''
    tokenizes the mention of the p(e|m) dictionary so that it is consistent with our
    tokenized corpus (,.;' all these symbols separate from the previous word with whitespace)
    it only modifies the mention and nothing else.
    '''
    #for dict_file in ["prob_wikipedia_p_e_m.txt", "prob_yago_crosswikis_wikipedia_p_e_m.txt",
    #                  "prob_crosswikis_wikipedia_p_e_m.txt"]:
    for dict_file in ["prob_yago_crosswikis_wikipedia_p_e_m.txt"]:
        with open(config.base_folder+"data/p_e_m/"+dict_file) as fin, \
                open(config.base_folder+"data/p_e_m/" + "tokenized/"+dict_file, "w") as fout:
            diff_cnt = 0
            for line in fin:
                mention, rest = line.split('\t', 1)
                if len(mention.split()) > 1:
                    tokenized_mention = ' '.join(word_tokenize(mention))
                else:
                    tokenized_mention = mention
                if mention != tokenized_mention:
                    diff_cnt += 1
                    #print(mention, "  -------->  ", tokenized_mention)
                fout.write(tokenized_mention + "\t" + rest)
        print(dict_file, "diff_cnt = ", diff_cnt)


def print_p_e_m_dictionary_to_file(p_e_m, full_filepath):
    _, wiki_id_name_map = util.load_wiki_name_id_map()
    with open(full_filepath, "w") as fout:
        for mention, entities in p_e_m.items():
            out_acc = []
            # entities is a   defaultdict(int)
            # so items returns ent2: 10,  ent54:20, ent3:2
            sorted_ = sorted(entities.items(), key=operator.itemgetter(1), reverse=True)
            # a list of tuples   [(ent54,20), (ent2,10), (ent3,2)]
            total_freq = 0
            for ent_id, prob in sorted_:
                if len(out_acc) > 100:    # at most 100 candidate entities
                    break
                total_freq += prob
                out_acc.append(','.join([ent_id, str(prob),
                                         wiki_id_name_map[ent_id].replace(' ', '_')]))
            fout.write(mention + "\t" + str(total_freq) + "\t" + "\t".join(out_acc) + "\n")


def tokenize_p_e_m_and_merge_conflicts(filename, yago=False):
    """takes as input a p_e_m with absolute frequency, tokenizes the mention, handles conflicts
    (same mention after tokenization) with merging. execute that on wiki, crosswiki, yago
    absolute frequency files -> output again absolute frequency."""
    p_e_m = defaultdict(lambda: defaultdict(int))
    with open(config.base_folder+"data/p_e_m/"+filename) as fin:
        diff_cnt = 0
        conflicts_cnt = 0
        for line in fin:
            line = line.rstrip()
            l = line.split("\t")
            mention = l[0]
            tokenized_mention = ' '.join(word_tokenize(mention))

            if mention != tokenized_mention:
                diff_cnt += 1
            if tokenized_mention in p_e_m:
                conflicts_cnt += 1
                #print(mention, "  -------->  ", tokenized_mention)
            for e in l[2:]:
                if yago:
                    ent_id, _ = e.split(',', 1)
                    ent_id = ent_id.strip()   # not necessary
                    freq = 1
                else:
                    ent_id, freq, _ = e.split(',', 2)
                    ent_id = ent_id.strip()   # not necessary
                    freq = int(freq)
                p_e_m[tokenized_mention][ent_id] += freq

    print("conflicts from tokenization counter: ", conflicts_cnt)
    print_p_e_m_dictionary_to_file(p_e_m, config.base_folder+"data/p_e_m/tokenized/"+filename)


def from_freq_to_prob(filename):
    with open(config.base_folder+"data/p_e_m/tokenized/"+filename) as fin, \
            open(config.base_folder+"data/p_e_m/tokenized/prob_"+filename, "w") as fout:
        p_e_m_errors = 0
        for line in fin:
            line = line.rstrip()
            try:
                l = line.split("\t")
                total_freq = int(l[1])
                out_acc = [l[0], str(1)]   # mention and total_prob
                for e in l[2:]:
                    ent_id, freq, title = e.split(',', 2)
                    out_acc.append(','.join([ent_id, str(int(freq)/total_freq), title]))
                fout.write('\t'.join(out_acc) + "\n")
            except Exception as esd:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                p_e_m_errors += 1
                print("error in line: ", repr(line))


def merge_two_prob_dictionaries(filename1, filename2, newfilename):
    """merge two p_e_m dictionaries that are already in probabilities to a new one again
    with probabilities."""
    p_e_m = defaultdict(lambda: defaultdict(float))
    for filename in [filename1, filename2]:
        with open(config.base_folder+"data/p_e_m/tokenized/"+filename) as fin:
            for line in fin:
                line = line.rstrip()
                l = line.split("\t")
                mention = l[0]
                for e in l[2:]:
                    ent_id, prob, _ = e.split(',', 2)
                    ent_id = ent_id.strip()   # not necessary
                    prob = float(prob)
                    #p_e_m[mention][ent_id] = min(1, p_e_m[mention][ent_id] + prob)
                    p_e_m[mention][ent_id] = p_e_m[mention][ent_id] + prob   # without min
                    # even without restricting it still the range of values is [0,2]

    print_p_e_m_dictionary_to_file(p_e_m, config.base_folder+"data/p_e_m/tokenized/"
                                   + newfilename)


if __name__ == "__main__":
    tokenize_p_e_m()

    #tokenize_p_e_m_and_merge_conflicts("wikipedia_p_e_m.txt")
    #tokenize_p_e_m_and_merge_conflicts("crosswikis_wikipedia_p_e_m.txt")
    #tokenize_p_e_m_and_merge_conflicts("yago_p_e_m.txt", yago=True)

    #from_freq_to_prob("wikipedia_p_e_m.txt")
    #from_freq_to_prob("crosswikis_wikipedia_p_e_m.txt")
    #from_freq_to_prob("yago_p_e_m.txt")

    """
    merge_two_prob_dictionaries("prob_crosswikis_wikipedia_p_e_m.txt",
                                "prob_yago_p_e_m.txt",
                                "prob_yago_crosswikis_wikipedia_p_e_m.txt")
    """

    # vocabulary_count_wiki()
    # entity_count_wiki()
    # load_p_e_m()
    # from_freq_to_prob()
