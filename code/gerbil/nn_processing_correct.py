
from model.model_ablations import Model
from time import sleep
import tensorflow as tf
import pickle
from nltk.tokenize import word_tokenize
from preprocessing.prepro_util import SamplesGenerator
from evaluation.metrics import _filtered_spans_and_gm_gt_list
import numpy as np
from preprocessing.util import load_wikiid2nnid, reverse_dict, load_wiki_name_id_map, FetchCandidateEntities, load_persons

class StreamingSamples(object):
    def __init__(self):
        # those are not used here
        #self.chunk_id, self.ground_truth, self.ground_truth_len, self.begin_gm, self.end_gm
        #self.cand_entities_labels,
        """only those are used:
        self.words, self.words_len, self.chars, self.chars_len, \
        self.begin_span, self.end_span, self.spans_len, \
        self.cand_entities, self.cand_entities_scores,  \
        self.cand_entities_len"""
        self.sample = None
        self.empty = True

    def new_sample(self, sample):
        self.sample = sample
        self.empty = False

    def gen(self):
        while True:
            if not self.empty:
                self.empty = True
                yield self.sample
            else:
                print("sleep")
                sleep(0.5)

"""
  DT_INT64,   DT_INT64, 	DT_INT64, DT_INT64, 	DT_INT64, 	DT_INT64, 	    DT_INT64, 	DT_INT64, 	     DT_FLOAT, 		          DT_INT64, 
   words,      words_len,    chars,    chars_len,  	begin_span,  	end_span,   spans_len,   cand_entities,  cand_entities_scores,  cand_entities_len, 

[?,?],     [?],       [?,?,?],    [?,?],      [?,?],       [?,?],        [?],        [?,?,?],          [?,?,?],                   [?,?],            
words,   words_len,    chars,    chars_len,  begin_span,  end_span,   spans_len,   cand_entities,   cand_entities_scores,    cand_entities_len, 
"""

"""(tf.TensorShape([None, None]), tf.TensorShape([None]), tf.TensorShape([None, None, None]), tf.TensorShape([None, None]),
tf.TensorShape([None, None]), tf.TensorShape([None, None]), tf.TensorShape([None]),
tf.TensorShape([None, None, None]), tf.TensorShape([None, None, None]), tf.TensorShape([None, None])))
    
"""

class NNProcessing(object):
    def __init__(self, train_args, args):
        self.args = args
        # input pipeline
        self.streaming_samples = StreamingSamples()
        ds = tf.data.Dataset.from_generator(
            self.streaming_samples.gen, (tf.int64, tf.int64, tf.int64, tf.int64,  #words, words_len, chars, chars_len
                                    tf.int64, tf.int64, tf.int64,   # begin_span, end_span, span_len
                                    tf.int64, tf.float32, tf.int64),  #cand_entities, cand_entities_scores, cand_entities_len
            (tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([None, None]), tf.TensorShape([None]),
             tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([]),
             tf.TensorShape([None, None]), tf.TensorShape([None, None]), tf.TensorShape([None])))
        next_element = ds.make_one_shot_iterator().get_next()
        # batch size = 1   i expand the dims now to match the training that has batch dimension
        next_element = [tf.expand_dims(t, 0) for t in next_element]
        next_element = [None, *next_element[:-1], None, next_element[-1],
                        None, None, None, None]

        # restore model
        print("loading Model:", train_args.output_folder)
        model = Model(train_args, next_element)
        model.build()
        checkpoint_path = model.restore_session("el" if args.el_mode else "ed")
        self.model = model
        if args.hardcoded_thr:
            self.thr = args.hardcoded_thr
            print("threshold used:", self.thr)
        else:
            # optimal threshold recovery from log files.
            # based on the checkpoint selected look at the log file for threshold (otherwise recompute it)
            self.thr = retrieve_optimal_threshold_from_logfile(train_args.output_folder, checkpoint_path, args.el_mode)
            print("optimal threshold selected = ", self.thr)

        if args.running_mode == "el_mode":
            args.el_mode = True
        elif args.running_mode == "ed_mode":
            args.el_mode = False
        # else it remains as it is
        if args.manual_thr:
            self.thr = args.manual_thr
            print("threshold manually overriden = ", self.thr)

        # convert text to tensors for the NN
        with open(args.experiment_folder+"word_char_maps.pickle", 'rb') as handle:
            self.word2id, _, self.char2id, _, _, _ = pickle.load(handle)

        self.wikiid2nnid = load_wikiid2nnid(extension_name=args.entity_extension)
        self.nnid2wikiid = reverse_dict(self.wikiid2nnid, unique_values=True)
        _, self.wiki_id_name_map = load_wiki_name_id_map()

        with open(args.experiment_folder+"prepro_args.pickle", 'rb') as handle:
            self.prepro_args = pickle.load(handle)
            if args.lowercase_spans_pem:
                self.prepro_args.lowercase_p_e_m = True
                self.prepro_args.lowercase_spans = True
        print("prepro_args:", self.prepro_args)
        self.fetchCandidateEntities = FetchCandidateEntities(self.prepro_args)

        self.special_tokenized_words = {"``", '"', "''"}
        self.special_words_assertion_errors = 0
        self.gm_idx_errors = 0
        if self.args.el_with_stanfordner_and_our_ed:
            from nltk.tag import StanfordNERTagger
            self.st = StanfordNERTagger(
                '/home/programs/stanford-ner-2018-02-27/classifiers/english.all.3class.distsim.crf.ser.gz',
                '/home/programs/stanford-ner-2018-02-27/stanford-ner.jar', encoding='utf-8')
        if self.args.persons_coreference:
            self.persons_wikiids = load_persons()
            self.persons_mentions_seen = set()

    def process(self, text, given_spans):
        self.persons_mentions_seen = set()
        original_words = word_tokenize(text)
        #print("tokenized chunk_words =", chunk_words)
        words2charidx = []
        idx = 0
        self.given_spans = given_spans
        #chunk_words = parsing_errors(chunk_words)
        # do not use this, not useful for ed either #el has slightly better results without using this
        #aida_test 0.8126 without, 0.8114 with.
        if not self.args.el_mode:
            startidx2wordnum = dict()
            endidx2wordnum = dict()
        chunk_words = []   # correct the special words
        for word_num, word in enumerate(original_words):
            original_word = word
            if word in self.special_tokenized_words:
                smallest_idx = len(text)
                for special_word in self.special_tokenized_words:
                    start = text.find(special_word, idx)
                    if start != -1 and start < smallest_idx:
                        word = special_word
                        smallest_idx = start
                if word != '"':
                    pass
                    #print("special word replacement: ", original_words[max(0, word_num-2):word_num+2], "new word:", word)
            start = text.find(word, idx)
            if start == -1 or start > idx + 10:
                print("Assertion Error! in words2charidx. word={}, original_word={}".format(word, original_word),
                      "near_text={}\nsnippet={}".format(text[idx:idx+20], text[idx-50:idx+50]))
                self.special_words_assertion_errors += 1
                for special_word in self.special_tokenized_words:
                    start = text.find(special_word, idx)
                    print("idx=", idx, "special_word =", special_word, "start=", start)
            else:
                chunk_words.append(word)
                end = start + len(word)
                idx = end
            assert(len(words2charidx) == word_num)
            words2charidx.append((start, end))  # [..)
            if not self.args.el_mode:
                startidx2wordnum[start] = word_num
                endidx2wordnum[end] = word_num

        if self.args.el_with_stanfordner_and_our_ed:  # el test, use stanford ner to extract spans and decide with our ed system
            self.given_spans, myspans = self.stanford_ner_spans(chunk_words, words2charidx)
        # from given given_spans (start, length) in characters convert them to given_spans in word num
        elif not self.args.el_mode:  # simple ed mode. use the spans provided
            self.given_spans = sorted(self.given_spans)
            myspans = []
            for span in self.given_spans:
                try:
                    start, length = span
                    end = start+length
                    if start not in startidx2wordnum:
                        start = self.nearest_idx(start, startidx2wordnum.keys())
                    if end not in endidx2wordnum:
                        end = self.nearest_idx(end, endidx2wordnum.keys())
                    if (start, end-start) != span:
                        print("given span:", text[span[0]:span[0]+span[1]], " new span:",
                              text[start:end])
                    myspans.append((startidx2wordnum[start], endidx2wordnum[end]+1))
                except KeyError:
                    print("Exception KeyError!!!!")
                    print("original_words =", original_words)
                    print("chunk_words =", chunk_words)
                    print("start={}, length={}, left={}, span={}, right={}".format(start, length,
                            text[start-30:start], text[start:start+length], text[start+length:start+length+30]))
                    print("text =", text)
                    print("start= {}".format("in" if start in startidx2wordnum else "out"))
                    print("end=   {}".format("in" if start + length in endidx2wordnum else "out"))
        else: # simple el mode
            # consider all possible given_spans
            myspans = SamplesGenerator.all_spans(chunk_words)
        # at this point whether we do ed or el by stanfordner_plus_oured we must have myspans  [word_num_b, word_num_end)
        # and self.given_spans which are those spans but with characters [begin_char, length)

        begin_spans, end_spans, cand_entities, cand_entities_scores = [], [], [], []
        for left, right in myspans:
            span_text = ' '.join(chunk_words[left:right])
            cand_ent, scores = self.fetchCandidateEntities.process(span_text)
            if self.args.persons_coreference:
                coreference_supermention = self.find_corefence_person(span_text)
                if coreference_supermention:
                    print("original text:", chunk_words[max(0, left-4):max(len(chunk_words), right+4)])
                    if not self.args.persons_coreference_merge:
                        cand_ent, scores = self.fetchCandidateEntities.process(coreference_supermention)
                    else: # merge with cand_ent and scores
                        cand_ent2, scores2 = self.fetchCandidateEntities.process(coreference_supermention)
                        temp1 = list(zip(scores, cand_ent))
                        temp2 = list(zip(scores2, cand_ent2))
                        temp3 = sorted(temp1 + temp2, reverse=True)
                        scores, cand_ent = map(list, zip(*temp3[:self.prepro_args.cand_ent_num]))

            # ['Obama_e', 'ent2', 'ent3']   , [0.9, 0.2, 0.8]
            # filter out entities that are not in our universe (and its corresponding scores)
            # then encode it from wikiid2nnid
            # similar to prepro_util._encode_cand_entities_and_labels
            cand_ent_filtered, scores_filtered = [], []
            if cand_ent is not None and scores is not None:
                if self.args.persons_coreference and not coreference_supermention and \
                                cand_ent[0] in self.persons_wikiids and len(span_text) >= 3:
                    self.persons_mentions_seen.add(span_text)
                for e, s in zip(cand_ent, scores):
                    if e in self.wikiid2nnid:
                        cand_ent_filtered.append(self.wikiid2nnid[e])
                        scores_filtered.append(s)

            if cand_ent_filtered:
                begin_spans.append(left)
                end_spans.append(right)
                cand_entities.append(cand_ent_filtered)
                cand_entities_scores.append(scores_filtered)

        if begin_spans == []:
            return []  # this document has no annotation

        words = []
        chars = []
        for word in chunk_words:
            words.append(self.word2id[word] if word in self.word2id
                         else self.word2id["<wunk>"])
            chars.append([self.char2id[c] if c in self.char2id else self.char2id["<u>"]
                          for c in word])
        chars_len = [len(word) for word in chars]
        new_sample = (words, len(words), list_of_lists_to_2darray(chars), chars_len,
                                           begin_spans, end_spans, len(begin_spans),
                                           list_of_lists_to_2darray(cand_entities),
                                           list_of_lists_to_2darray(cand_entities_scores),
                                           [len(t) for t in cand_entities])
        self.streaming_samples.new_sample(new_sample)

        result_l = self.model.sess.run([self.model.final_scores, self.model.cand_entities_len,
                            self.model.cand_entities, self.model.begin_span, self.model.end_span,
                            self.model.spans_len], feed_dict={self.model.dropout: 1})
        filtered_spans, _ = _filtered_spans_and_gm_gt_list(0, *result_l, None, None, None, [0], [len(words)])

        # based on final_scores and thr return annotations. also translate my given_spans to char given_spans

        print("self.special_words_assertion_errors =", self.special_words_assertion_errors)
        print("gm_idx_errors =", self.gm_idx_errors)

        if self.args.each_entity_only_once or self.args.each_mention_only_once or \
            self.args.omit_first_sentence:
            return self.custom_response(filtered_spans, text, words2charidx, chunk_words)
        response = []
        for span in filtered_spans:
            score, begin_idx, end_idx, nnid = span
            if score >= self.thr:
                self._add_response_span(response, span, words2charidx)
        print("self.persons_mentions_seen =", self.persons_mentions_seen)
        return response

    def find_corefence_person(self, span_text):
        """if span_text is substring of another person mention found before. it should be
        substring of words. so check next character and previous character to be non alphanumeric"""
        if len(span_text) < 3:
            return None
        for mention in self.persons_mentions_seen:
            idx = mention.find(span_text)
            if idx != -1:
                #print("find_coreference_person substring initial match")
                if len(mention) == len(span_text):
                    continue   # they are identical so no point in substituting them
                if idx > 0 and mention[idx-1].isalpha():
                    continue
                if idx < len(mention) - 1 and mention[idx+1].isalpha():
                    continue
                print("persons coreference, before:", span_text, "after:", mention)
                return mention
        return None


    def nearest_idx(self, key, values):
        self.gm_idx_errors += 1
        # find the value in values that is nearest to key
        nearest_value = None
        min_distance = 1e+6
        for value in values:
            if abs(key - value) < min_distance:
                nearest_value = value
                min_distance = abs(key-value)
        return nearest_value

    def _add_response_span(self, response, span, words2charidx):
        score, begin_idx, end_idx, nnid = span
        start = words2charidx[begin_idx][0]  # the word begin_idx starts at this character
        end = words2charidx[end_idx-1][1]  # the word begin_idx starts at this character
        wikiid = self.nnid2wikiid[nnid]
        wikiname = self.wiki_id_name_map[wikiid].replace(' ', '_')
        if not self.args.el_mode:   # try to match it with a given span
            start, end = self.nearest_given_span(start, end)
        response.append((start, end-start, wikiname))

    def nearest_given_span(self, begin_idx, end_idx):    # [begin_idx, end_idx)  end_idx points to the next character after mention
        min_distance = 1e+6
        nearest_idxes = (-1, -1)
        for (start, length) in self.given_spans:
            distance = abs(begin_idx - start) + abs(end_idx - (start + length))
            if distance < min_distance:
                nearest_idxes = (start, start + length)
                min_distance = distance
        return nearest_idxes

    def custom_response(self, filtered_spans, text, words2charidx, chunk_words):
        from operator import itemgetter
        filtered_spans = sorted(filtered_spans, key=itemgetter(1))
        response = []
        # omit title
        if self.args.omit_first_sentence:
            start = text.find(self.args.first_sentence_separator)
            if start > 100:  # this dataset doesn't have a title so do not omit anything
                start = 0
            print("omit first sentence:", text[:start])
            for i, span in enumerate(filtered_spans):
                if words2charidx[span[1]][0] > start:
                    break
            print("omitted annotations:", [chunk_words[span[1]:span[2]] for span in filtered_spans[:i] if span[0] > self.thr])
            filtered_spans = filtered_spans[i:]

        # each entity only once
        if self.args.each_entity_only_once:
            used_entities = set()
            for span in filtered_spans:
                score, begin_idx, end_idx, nnid = span
                if score >= self.thr and nnid not in used_entities:
                    self._add_response_span(response, span, words2charidx)
                    used_entities.add(nnid)
        elif self.args.each_mention_only_once:
            used_mentions = set()
            for span in filtered_spans:
                score, begin_idx, end_idx, nnid = span
                mention = text[words2charidx[begin_idx][0]:words2charidx[end_idx-1][1]]
                if score >= self.thr and mention not in used_mentions:
                    self._add_response_span(response, span, words2charidx)
                    used_mentions.add(mention)

        return response

    def stanford_ner_spans(self, words_l, words2charidx):
        """returns a list of tuples (start_idx, length)"""
        tags = self.st.tag(words_l)
        begin_spans, end_spans, prev_tag = [], [], 'O'
        for i, (_, tag) in enumerate(tags):
            if tag == 'O' and prev_tag != 'O':
                end_spans.append(i)
            elif tag == 'O' and prev_tag == 'O':
                pass
            elif tag != 'O' and prev_tag == 'O':
                begin_spans.append(i)
            elif tag != 'O' and prev_tag == tag:
                pass
            elif tag != 'O' and prev_tag != tag: # and prev_tag != 'O'
                end_spans.append(i)
                begin_spans.append(i)
            prev_tag = tag

        char_spans = []   # (begin_char, length)
        word_spans = []  # [begin_word, end_word)
        for bw, ew in zip(begin_spans, end_spans):
            word_spans.append((bw, ew))
            bc = words2charidx[bw][0]
            ec = words2charidx[ew-1][1]
            char_spans.append((bc, ec - bc))
        return char_spans, word_spans


""" not used
def parsing_errors(chunk_words):
    # check each chunk_word if it is alpha. if not then try to split it
    # 'U.S' '.' merge them to U.S.
    # alpha notalaphcharacter alpha    do not split them
    temp = []
    i = 0
    while i < len(chunk_words):
        word = chunk_words[i]
        if word == 'U.S.':
            temp.append(word)
        elif word == 'U.S' and chunk_words[i+1] == '.':
            temp.append('U.S.')
            i += 1
        elif word.isalpha():
            temp.append(word)
        else:
            temp_idx = 0
            for c in word:
                if not c.isalpha():
                    break
                temp_idx += 1
            if 0 < temp_idx < len(word) and not word[temp_idx+1:].isalpha():
                temp.append(word[:temp_idx])
                temp.append(word[temp_idx:])
            else:
                temp.append(word)
        i += 1
    return temp
"""

def list_of_lists_to_2darray(a):
    # with padding zeros
    b = np.zeros([len(a), len(max(a, key=lambda x: len(x)))])
    for i, j in enumerate(a):
        b[i][0:len(j)] = j
    return b


def retrieve_optimal_threshold_from_logfile(model_folder, checkpoint_path, el_mode):
    eval_cnt = checkpoint_path[checkpoint_path.rfind("-")+1:]  # fixed_no_wikidump_entvecsl2/checkpoints/model-7
    print("eval_cnt from checkpoint_path =", eval_cnt)
    with open(model_folder+"log.txt", "r") as fin:
        line = next(fin).strip()
        while line != "args.eval_cnt =  " + eval_cnt:
            line = next(fin).strip()
        line = next(fin).strip()
        while line != "Evaluating {} datasets".format("EL" if el_mode else "ED"):
            line = next(fin).strip()
        line = next(fin).strip()  # Best validation threshold = -0.112 with F1=91.8
        line = line.split()
        assert line[3] == "=" and line[5] == "with", line
        return float(line[4])



if __name__ == "__main__":
    debug = True