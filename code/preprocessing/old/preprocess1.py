import pickle
import argparse
from collections import defaultdict
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import numpy as np
import time

# TODO put all these in a config file
data_folder = "../data/mydata/"  # "../data/basic_data/"
dev_ratio = 0.1
test_ratio = 0.05
per_sentence = False
per_paragraph = False   # or per article, or per sentence
per_article = True

word_freq_thr = 20   # word frequency threshold (if freq less than that then replace it with the
                           # special word <wunk>. if it has higher than that frequency but not in Word2Vec again
                           # replace it with <wunk>
char_freq_thr = 100   # character frequency threshold (if encountered less times than that then
                      # replace it with the char <cunk>

lowercase_wordemb = False   # before we look for its emb we lowercase




def main():
    word_freq = defaultdict(int)
    char_freq = defaultdict(int)    # how many times each character is encountered
    entity_freq = defaultdict(int)  # how many hyperlinks point to this entity
    ent_name_id_map = dict()        # just to verify that it is the same with the wiki_name_id_map.txt


    import re
    docid_title_pattern = re.compile(r'curid=(\d+)" title="(.*)">')
    hyperlink_pattern = re.compile(r'(.*?)<a href="(.*?)">(.*?)</a>')


    doc_cnt = 0
#    with open(data_folder + "toy_wiki_dump.txt") as fin: # "textWithAnchorsFromAllWikipedia2014Feb.txt") as f:
    with open("../data/basic_data/textWithAnchorsFromAllWikipedia2014Feb.txt") as fin:
        for line in fin:
            if line.startswith('<doc id="'): continue
            if line.startswith('</doc>'):
                doc_cnt += 1
                if doc_cnt % 500 == 0:
                    print("document counter: ", doc_cnt)
                continue
            result = re.findall(hyperlink_pattern, line)
            clean_text = []
            for res in result:
                clean_text.append(res[0])
                clean_text.append(res[2])
            clean_text.append(line.rsplit('</a>', 1)[-1]) # text after last hyperlink
            clean_text = ''.join(clean_text)
            for w in word_tokenize(clean_text):
                word_freq[w] += 1
            for c in clean_text:
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




def re_faster(line):
    result = []
    pass

def profiler():


    import re
    docid_title_pattern = re.compile(r'curid=(\d+)" title="(.*)">')
    hyperlink_pattern = re.compile(r'(.*?)<a href="(.*?)">(.*?)</a>')

    wall_start = time.time()
    experiments_times = []
    doc_cnt = 0
    for tokenization_flag, counting_flag, name in [(False, False, "clean text"), (True, False, "clean text & tokenization"), (True, True, "clean text & tokenization & counting")]: 
        word_freq = defaultdict(int)
        char_freq = defaultdict(int)    # how many times each character is encountered
        entity_freq = defaultdict(int)  # how many hyperlinks point to this entity
        ent_name_id_map = dict()        # just to verify that it is the same with the wiki_name_id_map.txt
        with open("../data/basic_data/textWithAnchorsFromAllWikipedia2014Feb.txt") as fin:
            for line in fin:
                if line.startswith('<doc id="'): continue
                if line.startswith('</doc>'):
                    doc_cnt += 1
                    if doc_cnt % 500 == 0:
                        print("document counter: ", doc_cnt)
                    if doc_cnt == 10000:
                        doc_cnt = 0
                        break
                    continue
                result = re.findall(hyperlink_pattern, line)
                clean_text = []
                for res in result:
                    clean_text.append(res[0])
                    clean_text.append(res[2])
                clean_text.append(line.rsplit('</a>', 1)[-1]) # text after last hyperlink
                clean_text = ''.join(clean_text)
                
                if tokenization_flag:
                    all_tokens = word_tokenize(clean_text)
                    if counting_flag: 
                        for w in all_tokens:
                            word_freq[w] += 1
                        for c in clean_text:
                            char_freq[c] += 1
	
        cur_exp_time = (time.time() - wall_start)/60
        print("wall time for ", name, " : ", cur_exp_time, " minutes") 
        experiments_times.append(cur_exp_time)
        wall_start = time.time()

    print("clean text time: ", experiments_times[0])
    print("tokenization time: ", experiments_times[1]-experiments_times[0])
    print("counting time: ", experiments_times[2]-experiments_times[1])

'''
clean text time:  2.8340925057729085
tokenization time:  2.915117104848226
counting time:  0.2761443416277567
'''


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
    #main()
    profiler()
