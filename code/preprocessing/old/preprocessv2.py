import pickle
import argparse
from collections import defaultdict
import numpy as np
import time
import sys
import os

# TODO put all these in a args file
import config
import util




def build_word_char_maps():
    with open(config.base_folder+"data/vocabulary/frequencies.pickle", 'rb') as handle:
        word_freq, char_freq = pickle.load(handle)
    word2id = dict()
    id2word = dict()
    char2id = dict()
    id2char = dict()

    import gensim
    model = gensim.models.KeyedVectors.load_word2vec_format(
        "/home/master_thesis_share/data/basic_data/wordEmbeddings/Word2Vec/GoogleNews-vectors-negative300.bin", binary=True)
    embedding_dim = len(model['queen'])

    wcnt = 0   # unknown word
    word2id["<wunk>"] = wcnt
    id2word[wcnt] = "<wunk>"
    wcnt += 1
    ccnt = 0   # unknown character
    char2id["<u>"] = ccnt
    id2char[ccnt] = "<u>"
    ccnt += 1

    for word in word_freq:          # for every word in the corpus
        if word in model:           # has a pretrained embeddings
            if config.word_freq_thr and word_freq[word] < config.word_freq_thr:
                pass       # rare word, so don't put it in our vocabulary
            else:
                word2id[word] = wcnt
                id2word[wcnt] = word
                wcnt += 1

    for c in char_freq:
        if config.char_freq_thr and char_freq[c] < config.char_freq_thr:
            pass       # rare character, so don't put it in our vocabulary
        else:
            char2id[c] = ccnt
            id2char[ccnt] = c
            ccnt += 1
    # the space ' ' and the EOL '\n' are not inside since they are discarded by the tokenizer
    # maybe they are useful later?
    assert(len(word2id) == wcnt)
    assert(len(char2id) == ccnt)
    print("words in vocabulary: ", wcnt)
    print("characters in vocabulary: ", ccnt)
    with open(config.base_folder+"data/vocabulary/word_char_maps.pickle", 'wb') as handle:
        pickle.dump((word2id, id2word, char2id, id2char, config.word_freq_thr, config.char_freq_thr), handle)

    return word2id, id2word, char2id, id2char, config.word_freq_thr, config.char_freq_thr




def samples_load(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

# https://stackoverflow.com/questions/20716812/saving-and-loading-multiple-objects-in-pickle-file
#samples = load(myfilename)

class Encoding(object):
    def __init__(self):
        self.entity_freq = defaultdict(int)  # how many hyperlinks point to this entity
        self.ent_name_id_map = dict()        # just to verify that it is the same with the wiki_name_id_map.txt
        self.ent_id_name_map = dict()
        with open(config.base_folder+"data/vocabulary/word_char_maps.pickle", 'rb') as handle:
            self.word2id, _, self.char2id, _, _, _ = pickle.load(handle)
            #_word2id, id2word, _char2id, id2char, word_freq_thr, char_freq_thr = pickle.load(handle)

        with open(config.base_folder+"data/serializations/p_e_m_disamb_redirect_wikinameid_maps.pickle", 'rb') as handle:
            self.p_e_m, _, self.disambiguations_ids, _, self.redirections, self.wiki_name_id_map, _,\
                self.wiki_name_id_map_lower = pickle.load(handle)
        #  p_e_m, args.cand_ent_num, disambiguations_ids, disambiguations_titles, redirections,\
        #  wiki_name_id_map, wiki_id_name_map

        self.unk_ent_id = 0
        #assert(0 not in self.wiki_id_name_map)

        self.chunk_ending = {'</doc>'}
        if config.per_paragraph:
            self.chunk_ending.add('*NL*')
        if config.per_sentence:
            self.chunk_ending.add('*NL*')
            self.chunk_ending.add('.')


    def new_chunk(self):
        self.chunk_words = []
        self.chunk_chars = []        # a list of lists
        self.begin_idx = []          # the start positions of mentions
        self.end_idx = []            # the end positions of mentions
        self.candidate_entities = [] # a list of lists

    def process_word(self, word):
        pass

    def serialize(self):
        if self.begin_idx != []: # this chunk has something to train on
            self.serialize_tfrecords()
            #pickle.dump(((docid, par_cnt, sent_cnt), self.chunk_words, self.chunk_chars,
            #            self.begin_idx, self.end_idx, self.candidate_entities), handle)
        self.new_chunk()

    def serialize_tfrecords(self):
        pass


        pass

    def encode_wikidump(self):
        doc_cnt = 0
        #with open(base_folder+"data/basic_data/tokenizedWiki.txt") as fin,\
        with open(config.base_folder + "data/mydata/tokenized_toy_wiki_dump.txt") as fin,\
            open(config.base_folder+"data/basic_data/samples.pickle", 'wb') as handle:
            self.new_chunk()

            # paragraph and sentence counter are not actually useful. only for debugging purposes.
            par_cnt = 0      # paragraph counter (useful if we work per paragraph)
            sent_cnt = 0      # sentence counter (useful if we work per sentence)
            self.in_mention = False
            for line in fin:
                line = line[:-1]       # omit the '\n' character
                if line.startswith('<doc\xa0id="'):
                    docid = int(line[9:line.find('"', 9)])
                    doctitle = line[line.rfind('="') + 2:-2]
                    self.ent_name_id_map[doctitle] = docid  # i can convert it to int as well
                    self.ent_id_name_map[docid] = doctitle
                    # print("docid: ", docid, "\t doctitle: ", doctitle)
                    par_cnt = 0
                    sent_cnt = 0
                    continue
                if line == '*NL*':
                    par_cnt += 1
                    sent_cnt = 0
                if line == '.':
                    sent_cnt += 1

                if line in self.chunk_ending:
                    # end of chunk. serialize this sample to file
                    # if last word was '.' add this as well
                    if line == '.':
                        # process word
                        self.process_word(line)
                        pass
                    self.serialize()
                    # TODO also find the candidate entities!!!!! serialize those as well!!!
                    # TODO begin and end indexes of all hyperlinks
                    # TODO correct entity linking

                    if line == '</doc>':
                        doc_cnt += 1
                        if doc_cnt % 5000 == 0:
                            print("document counter: ", doc_cnt)
                    continue

                if line.startswith('<a\xa0href="'):
                    # a new mention begins
                    self.in_mention = True
                    self.correct_entity = self.from_hyperlink_text_to_entity_id(line)


                if line.startswith('<a href="') or line.startswith('<doc id="') or \
                        line.startswith('</a>') or line.startswith('*NL*'):
                    continue


            #if line.startswith('<a\xa0href="') or line.startswith('<doc\xa0id="') or \
            #        line.startswith('</a>') or line.startswith('*NL*'):






'''

        # first pass through the corpus. now we just count the frequency of words, chars and entities
        #text_chunk = []  # it can be a sentence, paragraph or the whole article
        #text_chunk = ''.join([`num` for num in xrange(loop_count)])
        chunks = []    # this is where we accumulate all the data samples. but since the
                       # corpus is too big we flush it at the end of each document that we read
        for line in fin:
            if line.startswith('<doc id="'):
                chunks = []
                result = re.search(docid_title_pattern, line)
                if result:  # it should always match so this is redundant
                    docid, title = result.group(1), result.group(2)
                    ent_name_id_map[title] = docid
                    print("id = ", docid, " title = ", title)
                else:
                    print("could not detect title pattern in line\n"+line)
                continue
            if line.startswith('</doc>'):
                # TODO flush the chunks to file train dev test
            # read consecutive lines until we form the desired chunking article, par, sent
            if per_sentence:
                cur_chunks = sent_tokenize(line)



            # in the second pass we do that (after encoding)
            #if line.startswith('</doc>'):
                # TODO flush the chunks to file train dev test

'''



if __name__ == "__main__":
    # util.tokenize_p_e_m()
    # util.vocabulary_count()
    util.p_e_m_disamb_redirect_wikinameid_maps()
    # the above commands can be executed only once. the results are serialized so no need to
    # execute them every time.

    #build_word_char_maps()

    #temp = Encoding()
    #temp.encode_wikidump()
    #main()
