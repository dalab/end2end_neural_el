import pickle
from collections import defaultdict
import numpy as np
import time
import sys
import os

import preprocessing.config as config
from preprocessing.util import *


def vocabulary_count_wiki():
    word_freq = defaultdict(int)
    char_freq = defaultdict(int)    # how many times each character is encountered

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




def entity_count_wiki_aux():
    with open(config.base_folder+"data/entities/entity_frequencies.pickle", 'rb') as handle:
        ent_freq = pickle.load(handle)
    import operator
    sorted_ = sorted(ent_freq.items(), key=operator.itemgetter(1), reverse=True)
    _, wiki_id_name_map = load_wiki_name_id_map()
    wiki_id_name_map["0"] = "unk_entity"
    ent_id_errors = 0
    with open(config.base_folder+"data/entities/entity_frequencies.txt", "w") as fout:
        for ent_id, freq in sorted_:
            if ent_id not in wiki_id_name_map:
                print("ent_id: ", ent_id, " not in wiki_name_id_map")
                ent_id_errors += 1
            else:
                fout.write(ent_id+"\t"+wiki_id_name_map[ent_id]+"\t"+str(freq)+"\n")
    print("ent_id_errors: ", ent_id_errors)


# checks for entities
# equivalent to vocabulary_count but for entities. executed only once
# counts how many hyperlinks point to each entity and prints statistics
def entity_count_wiki():
    ent_freq = defaultdict(int)

    hyperlink2EntityId = EntityNameIdMap()
    hyperlink2EntityId.init_hyperlink2id()
    doc_cnt = 0
    #with open(base_folder + "data/mydata/tokenized_toy_wiki_dump.txt") as fin:
    with open(config.base_folder+"data/basic_data/tokenizedWiki.txt") as fin:
        for line in fin:
            if line.startswith('</doc>'):
                doc_cnt += 1
                if doc_cnt % 5000 == 0:
                    print("document counter: ", doc_cnt)
                continue
            if line.startswith('<a\xa0href="'):
                ent_id = hyperlink2EntityId.hyperlink2id(line)
                ent_freq[ent_id] += 1

    # print some statistics

    print("number of entities that are linked at least once, len(ent_freq) = ", len(ent_freq))
    print("hyperlinks to unknown entities = ", ent_freq[config.unk_ent_id])
    print("hyperlinks_to_dismabiguation_pages = ", hyperlink2EntityId.hyperlinks_to_dismabiguation_pages)
    print("some entity frequency statistics. The bins are [...) ")
    for d, name, edges in zip([ent_freq], ["entity frequencies"],
                              [[1, 2, 3, 6, 11, 21, 31, 51, 76, 101, 201, 501, np.inf]]):
        hist_values, _ = np.histogram(list(d.values()), edges)
        cum_sum = np.cumsum(hist_values[::-1])
        print(name, " frequency histogram, edges: ", edges)
        print("absolute values:        ", hist_values)
        print("absolute cumulative (right to left):    ", cum_sum[::-1])
        print("probabilites cumulative (right to left):", (cum_sum / np.sum(hist_values))[::-1])

    with open(config.base_folder+"data/entities/entity_frequencies.pickle", 'wb') as handle:
        pickle.dump(ent_freq, handle)


def get_frequent_entities_set(top=None, freq_thr=None, return_freq=False):
    """returns a set with ent_id of the top most frequent entities (in terms of hyperlinks
    pointing to them). e.g. top=10000 returns a set with the top 10000 most frequent entities"""
    with open(config.base_folder+"data/entities/entity_frequencies.pickle", 'rb') as handle:
        ent_freq = pickle.load(handle)
    import operator
    sorted_ = sorted(ent_freq.items(), key=operator.itemgetter(1), reverse=True)
    sorted_ = sorted_[1:] # in order to omit the most frequent entity the "unknown entity" 0
    if top is not None:
        sorted_ = sorted_[:top]    # [('ent43', 9), ('ent12', 8), ('ent58', 5)]
    if freq_thr is not None:
        last_element = len(sorted_)
        for i, (_, freq) in enumerate(sorted_):
            if freq < freq_thr:
                last_element = i
                break
        sorted_ = sorted_[:last_element]
    if return_freq:
        return sorted_
    else:
        return list(list(zip(*sorted_))[0])   # ['ent43', 'ent12', 'ent58']


def entity_name_id_map_from_dump():
    """goes through the tokenized wiki dump and by processing the lines <doc id=..
    it creates a map from name to id and the inverse. It outputs these in a file similar
    to wiki_name_id_map.txt with name my_wiki_name_id_fromdump_map.txt
    In theory this should be the same with the wiki_name_id_map.txt but it isn't..."""
    doc_cnt = 0
    my_ent_name_id_map = dict()
    my_ent_id_name_map = dict()
    duplicate_names = 0    # different articles with identical title
    duplicate_ids = 0      # with the same id
    with open(config.base_folder+"data/basic_data/tokenizedWiki.txt") as fin:
        for line in fin:
            line = line[:-1]       # omit the '\n' character
            if line.startswith('<doc\xa0id="'):
                docid = line[9:line.find('"', 9)]
                doctitle = line[line.rfind('="') + 2:-2].replace('\xa0', ' ')

                if doctitle in my_ent_name_id_map:
                    duplicate_names += 1
                if docid in my_ent_id_name_map:
                    duplicate_ids += 1

                my_ent_name_id_map[doctitle] = docid  # i can convert it to int as well
                my_ent_id_name_map[docid] = doctitle
                #print("docid: ", docid, "\t doctitle: ", doctitle)
                doc_cnt += 1
                if doc_cnt % 10000 == 0:
                    print("doc_cnt = ", doc_cnt)
    print("len(ent_name_id_map) = ", len(my_ent_name_id_map))
    print("duplicate names: ", duplicate_names)
    print("duplicate ids: ", duplicate_ids)
    with open(config.base_folder+"data/basic_data/my_wiki_name_id_fromdump.txt", 'w') as fout:
        for doc_title, doc_id in my_ent_name_id_map.items():
            fout.write(doc_title + "\t" + doc_id + "\n")

def compare_name_id_maps():
    wiki_name_id_map, _ = load_wiki_name_id_map(lowercase=False)
    my_wiki_name_id_map, _ = load_wiki_name_id_map(
        lowercase=False, filepath="/home/master_thesis_share/data/basic_data/my_wiki_name_id_fromdump.txt")
    print("len(wiki_name_id_map) = ", len(wiki_name_id_map))
    print("len(my_wiki_name_id_map) = ", len(my_wiki_name_id_map))

    wiki_names = set(wiki_name_id_map.keys())
    my_wiki_names = set(my_wiki_name_id_map.keys())
    given_minus_mine = wiki_names - my_wiki_names
    print("len(given_minus_mine)", len(given_minus_mine))
    mine_minus_given = my_wiki_names - wiki_names
    print("len(mine_minus_given)", len(mine_minus_given))

    for dict1, dict2 in [(wiki_name_id_map, my_wiki_name_id_map), (my_wiki_name_id_map, wiki_name_id_map)]:
        print("Loop:")
        different_id_cnt = 0
        not_included_cnt = 0   # inside my dictionary but not inside the wiki_name_id_map.txt
        for wiki_title, wiki_id in dict1.items():
            if wiki_title not in dict2:
                not_included_cnt += 1
                print("not included: ", wiki_title)
            elif dict2[wiki_title] != wiki_id:
                different_id_cnt += 1
                print("different id: ", wiki_title, "myid: ", wiki_id, " other id: ", wiki_name_id_map[wiki_title])
        print("Loop statistics:")
        print("different_id_cnt = ", different_id_cnt)
        print("not_included_cnt = ", not_included_cnt)



def test_wiki_name_id_map_txt_conflicts_when_lowering():
    lowercase_maps_original_value = config.lowercase_maps
    config.lowercase_maps = False
    print("args.lowercase_maps = ", config.lowercase_maps)
    wiki_name_id_map, wiki_id_name_map = load_wiki_name_id_map()
    print(len(wiki_name_id_map), len(wiki_id_name_map))
    #assert(len(wiki_name_id_map) == len(wiki_id_name_map))

    config.lowercase_maps = True
    print("args.lowercase_maps = ", config.lowercase_maps)
    wiki_name_id_map_l, wiki_id_name_map_l = load_wiki_name_id_map()
    print(len(wiki_name_id_map_l), len(wiki_id_name_map_l))
    #assert(len(wiki_name_id_map_l) == len(wiki_id_name_map_l))







def create_p_e_m():
    wall_start = time.time()
    p_e_m = dict()  # for each mention we have a list of tuples (ent_id, score)
    p_e_m_errors = 0
    entityNameIdMap = EntityNameIdMap()
    entityNameIdMap.init_compatible_ent_id()
    incompatible_ent_ids = 0
    with open(config.base_folder + "data/p_e_m/tokenized/prob_yago_crosswikis_wikipedia_p_e_m.txt") as fin:
        for line in fin:
            line = line.rstrip()
            try:
                temp = line.split("\t")
                mention, entities = temp[0],  temp[2:2+config.cand_ent_num]
                res = []
                for e in entities:
                    ent_id, score, _ = e.split(',', 2)
                    if not entityNameIdMap.is_valid_entity_id(ent_id):
                        incompatible_ent_ids += 1
                    else:
                        res.append((ent_id, float(score)))
                p_e_m[mention] = res    # for each mention we have a list of tuples (ent_id, score)

            except Exception as esd:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                p_e_m_errors += 1
                print("error in line: ", repr(line))

    print("end of p_e_m reading. wall time:", (time.time() - wall_start)/60, " minutes")
    print("p_e_m_errors: ", p_e_m_errors)
    print("incompatible_ent_ids: ", incompatible_ent_ids)
    # TODO store also the value cand_ent_num. so later you can perform a check if the
    # serialization file exists and is about the same cand_ent_num then retrieve it and not
    # recompute it
    with open(config.base_folder+"data/p_e_m/serializations/p_e_m_{}.pickle".format(
            config.cand_ent_num), 'wb') as handle:
        pickle.dump(p_e_m, handle)

    if not config.lowercase_p_e_m:   # do not build lowercase dictionary
        return p_e_m

    wall_start = time.time()
    # two different p(e|m) mentions can be the same after lower() so we merge the two candidate
    # entities lists. But the two lists can have the same candidate entity with different score
    # we keep the highest score. For example if "Obama" mention gives 0.9 to entity Obama and
    # OBAMA gives 0.7 then we keep the 0.9 . Also we keep as before only the cand_ent_num entities
    # with the highest score
    p_e_m_lowercased = defaultdict(lambda: defaultdict(int))

    for mention, res in p_e_m.items():
        l_mention = mention.lower()
        # if l_mention != mention and l_mention not in p_e_m:
        #   the same so do nothing      already exist in dictionary
        #   e.g. p(e|m) has Obama and obama. So when i convert Obama to lowercase
        # I find that obama already exist so i will prefer this.
        if l_mention not in p_e_m:
            for r in res:
                ent_id, score = r
                p_e_m_lowercased[l_mention][ent_id] = max(score, p_e_m_lowercased[l_mention][ent_id])

    print("end of p_e_m lowercase. wall time:", (time.time() - wall_start)/60, " minutes")
    del p_e_m      # important in order to free memory because pickle ask for more

    import operator
    p_e_m_lowercased_trim = dict()
    for mention, ent_score_map in p_e_m_lowercased.items():
        sorted_ = sorted(ent_score_map.items(), key=operator.itemgetter(1), reverse=True)
        p_e_m_lowercased_trim[mention] = sorted_[:config.cand_ent_num]

    del p_e_m_lowercased
    with open(config.base_folder+"data/p_e_m/serializations/p_e_m_{}_low.pickle".format(
            config.cand_ent_num), 'wb') as handle:
        pickle.dump(p_e_m_lowercased_trim, handle)


def load_p_e_m():   # need 82 % of my memory   and  real	0m50.133s whereas to
    # create it i need 1.65 minutes
    if not os.path.exists(config.base_folder+"data/p_e_m/serializations/p_e_m_{}.pickle".format(
            config.cand_ent_num)) or (config.lowercase_p_e_m and not
    os.path.exists(config.base_folder+"data/p_e_m/serializations/p_e_m_{}_low.pickle".format(
        config.cand_ent_num))):
        create_p_e_m()
    with open(config.base_folder+"data/p_e_m/serializations/p_e_m_{}.pickle".format(
            config.cand_ent_num), 'rb') as handle:
        p_e_m = pickle.load(handle)
    with open(config.base_folder+"data/p_e_m/serializations/p_e_m_{}_low.pickle".format(
            config.cand_ent_num), 'rb') as handle:
        p_e_m_low = pickle.load(handle)
    return p_e_m, p_e_m_low









