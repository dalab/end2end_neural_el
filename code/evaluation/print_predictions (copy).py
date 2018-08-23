
from termcolor import colored
import pickle
from preprocessing.util import load_wikiid2nnid, reverse_dict, load_wiki_name_id_map
from collections import defaultdict
import operator

class PrintPredictions(object):
    def __init__(self, output_folder, predictions_folder, entity_extension=None):
        self.thr = None
        self.output_folder = output_folder
        self.predictions_folder = predictions_folder
        with open(output_folder+"word_char_maps.pickle", 'rb') as handle:
            _, self.id2word, _, self.id2char, _, _ = pickle.load(handle)

        self.nnid2wikiid = reverse_dict(load_wikiid2nnid(entity_extension), unique_values=True)
        _, self.wiki_id_name_map = load_wiki_name_id_map()
        self.extra_info = ""

    def map_entity(self, nnid):
        wikiid = self.nnid2wikiid[nnid]
        wikiname = self.wiki_id_name_map[wikiid].replace(' ', '_') if wikiid != "<u>" else "<u>"
        return "{} {}".format(wikiid, wikiname)

    def process_file(self, el_mode, name, opt_thr):
        self.thr = opt_thr
        self.el_mode = el_mode
        filepath = self.predictions_folder + ("el/" if el_mode else "ed/") + name
        self.fout = open(filepath, "w")

    def file_ended(self):
        self.fout.close()

    def process_sample(self, chunkid,
                       tp_pred, fp_pred, fn_pred, gt_minus_fn_pred,
                       words, words_len, chars, chars_len,
                       cand_entities, log_cand_entities_scores, cand_entities_len,
                       final_scores, similarity_scores):
        """words: [None] 1d the words of a sample, words_len: scalar,
        chars: [None, None] 2d  words, chars of each word, chars_len: [None] for each word
        the length in terms of characters.
        cand_entities: [None, None]  gold_mentions, candidates for each gm,
        cand_entitites_len: [None]  how many cand ent each gm has."""
        reconstructed_words = []
        for i in range(words_len):
            word = words[i]
            if word != 0:
                reconstructed_words.append(self.id2word[word])
            else:  # <wunk>
                word_chars = []
                for j in range(chars_len[i]):
                    word_chars.append(self.id2char[chars[i][j]])
                reconstructed_words.append(''.join(word_chars))

        text_tags = defaultdict(list)
        gt_legend = []

        for mylist, mycolor in zip([gt_minus_fn_pred, fn_pred], ["green", "red"]):
            for i, (gm_num, b, e, gt) in enumerate(mylist, 1):
                text_tags[b].append((1, colored("[{}".format(i), mycolor)))
                text_tags[e].append((0, colored("]", mycolor)))

                gt_text = ""
                if self.el_mode is False:  # find the position and the score of the ground truth
                    gt_text = "gt not in candidate entities (recall miss)"
                    for j in range(cand_entities_len[gm_num]):
                        if cand_entities[gm_num][j] == gt:
                            gt_text = "gt_p_e_m_pos={}, gt_logpem_score={}".format(j,
                                            log_cand_entities_scores[gm_num][j])
                            break
                text = colored("{}: {} {}".format(i, self.map_entity(gt), gt_text), mycolor)
                gt_legend.append(text)

        tp_legend = []
        tp_pred = sorted(tp_pred, key=operator.itemgetter(1))
        for i, (score, b, e, nnid, sim_score, p_e_m_pos, span_num) in enumerate(tp_pred, 1):
            text_tags[b].append((1, colored("[{}".format(i), "blue")))
            text_tags[e].append((0, colored("]", "blue")))

            text = colored("{}: {}, score={}, sim_score={}, logpem={}, pem_pos={}".format(i,
                        self.map_entity(nnid), score, sim_score,
                            log_cand_entities_scores[span_num][p_e_m_pos], p_e_m_pos), "blue")
            tp_legend.append(text)

        fp_legend = []
        fp_pred = sorted(fp_pred, key=operator.itemgetter(1))
        if len(fp_pred) > 0:
            fpWeakMatcherLogging = FPWeakMatcherLogging(self, fn_pred+gt_minus_fn_pred,
                                    cand_entities, cand_entities_len, log_cand_entities_scores,
                                    final_scores, similarity_scores)
        for i, (score, b, e, nnid, sim_score, p_e_m_pos, span_num) in enumerate(fp_pred, 1):
            text_tags[b].append((1, colored("[{}".format(i), "magenta")))
            text_tags[e].append((0, colored("]", "magenta")))

            fp_gt_text = fpWeakMatcherLogging.check(b, e, span_num)
            text = colored("{}: {}, score={}, sim_score={}, logpem={}, pem_pos={} {} ".format(i,
                            self.map_entity(nnid), score, sim_score,
                            log_cand_entities_scores[span_num][p_e_m_pos], p_e_m_pos,
                            fp_gt_text), "magenta")
            fp_legend.append(text)

        final_acc = ["new sample " + chunkid+"\n"]
        for i in range(words_len+1):
            final_acc.extend([text for _, text in sorted(text_tags[i])])
            if i < words_len:
                final_acc.append(reconstructed_words[i])
        self.fout.write(" ".join(final_acc)+"\n")
        self.fout.write("\n".join(gt_legend + tp_legend + fp_legend))
        self.fout.write("\n")


class FPWeakMatcherLogging(object):
    """is initialized with the gm_gt_list i.e. a list of tuples
    (begin_idx, end_idx, gt) and from the list of tuples it builds a data structure. We already
    know that our tuple doesn't match a ground truth. Now we want to find out what exactly happens.
    cases: 1)) doesn't overlap with any gm  2)) overlap with one or more gm. In this case for each gm
    that it overlaps with find a) which is the gt of this gm, b) final_score, sim_score, p_e_m position
    of the gt in my fp tuple.
    structure used: just a list of (begin_idx, end_idx, gt) tuples.
    This one is used only during evaluation.py from the
    metrics_calculation_and_prediction_printing in order to produce logging text
    for the fp"""
    def __init__(self, printPredictions, b_e_gt_iterator, cand_entities, cand_entities_len,
                 log_cand_entities_scores, final_scores, similarity_scores):
        self.printPredictions = printPredictions
        self.data = b_e_gt_iterator
        self.cand_entities = cand_entities
        self.cand_entities_len = cand_entities_len
        self.log_cand_entities_scores = log_cand_entities_scores
        self.final_scores = final_scores
        self.similarity_scores = similarity_scores

    def check(self, s, e, span_num):
        # all the above information that i have for my best_cand_id, now i have to find them
        # for the gt of the gm that overlap with my fp tuple.
        # compare my tuple s, e with all the gm

        acc = []
        for (gm_num, s2, e2, gt) in self.data:
            overlap = False  # overlap with this specific gm of the for loop
            if s<=s2 and e<=e2 and s2<e:
                overlap = True
            elif s>=s2 and e>=e2 and s<e2:
                overlap = True
            elif s<=s2 and e>=e2:
                overlap = True
            elif s>=s2 and e<=e2:
                overlap = True

            if not overlap:
                continue

            # add to the text accumulator the info for this gt
            # find gt_score, gt_similarity_score, gt_cand_position
            # check all the candidate entities of this span and find where is the gt
            # of course we may not find it at all (recall miss)
            gt_cand_position = -1
            for j in range(self.cand_entities_len[span_num]):
                if self.cand_entities[span_num][j] == gt:
                    gt_cand_position = j
                    break

            if gt_cand_position >= 0:
                acc.append("| {}, score={}, sim_score={}, logpem={}, pem_pos={}".format(
                        self.printPredictions.map_entity(gt),
                        self.final_scores[span_num][gt_cand_position],
                        self.similarity_scores[span_num][gt_cand_position],
                        self.log_cand_entities_scores[span_num][gt_cand_position],
                        gt_cand_position))
            else:
                acc.append("| {}, recall miss".format(self.printPredictions.map_entity(gt)))

        if acc == []:
            acc.append("| no overlap with gm")

        return ' '.join(acc)
