
from termcolor import colored
import pickle
from preprocessing.util import load_wikiid2nnid, reverse_dict, load_wiki_name_id_map
from collections import defaultdict
import operator
import numpy as np


class GMBucketingResults(object):
    def __init__(self, gm_bucketing_pempos):
        gm_bucketing_pempos.append(200)  # [0,1,2,7,200]
        self.gm_buckets = gm_bucketing_pempos
        self.gm_cnt = defaultdict(int)   # how many gold mentions fall in that frquency. one counter for each bucket
        self.fn_cnt = defaultdict(int)   # for many false negative in that bucket
        self.fn_nowinnermatch_cnt = defaultdict(int)  # from the fn we exclude the ones that our winner was identical
                                                      # to gt even if we decided not to annotate in the end
        self.gm_to_gt_unique_mapping = 0   # for this gold mention we have only one candidate entity which is the gt.

    def reinitialize(self):
        self.gm_cnt = defaultdict(int)
        self.fn_cnt = defaultdict(int)
        self.fn_nowinnermatch_cnt = defaultdict(int)
        self.gm_to_gt_unique_mapping = 0

    def process_fn(self, pos, match_with_winner, num_of_cand_entities):
        if pos == 0 and num_of_cand_entities == 1:
            self.gm_to_gt_unique_mapping += 1
        for t in self.gm_buckets:
            if pos <= t:
                self.gm_cnt[t] += 1
                self.fn_cnt[t] += 1
                if not match_with_winner:
                    self.fn_nowinnermatch_cnt[t] += 1
                break

    def process_tp(self, pos, num_of_cand_entities):
        if pos == 0 and num_of_cand_entities == 1:
            self.gm_to_gt_unique_mapping += 1
        for t in self.gm_buckets:
            if pos <= t:
                self.gm_cnt[t] += 1
                break

    def print(self):
        print("gm_to_gt_unique_mapping =", self.gm_to_gt_unique_mapping)
        for t in self.gm_buckets:
            print(str(t), "]", "gm_cnt=", str(self.gm_cnt[t]),
                  "solved=%.1f" % (100*(self.gm_cnt[t] - self.fn_cnt[t])/self.gm_cnt[t]),
                   "winner_match=%.1f" % (100*(self.gm_cnt[t] - self.fn_nowinnermatch_cnt[t])/self.gm_cnt[t]))



class PrintPredictions(object):
    def __init__(self, output_folder, predictions_folder, entity_extension=None, gm_bucketing_pempos=None,
                 print_global_voters=False, print_global_pairwise_scores=False):
        self.thr = None
        self.output_folder = output_folder
        self.predictions_folder = predictions_folder
        with open(output_folder+"word_char_maps.pickle", 'rb') as handle:
            _, self.id2word, _, self.id2char, _, _ = pickle.load(handle)

        self.nnid2wikiid = reverse_dict(load_wikiid2nnid(entity_extension), unique_values=True)
        _, self.wiki_id_name_map = load_wiki_name_id_map()
        self.extra_info = ""
        self.gm_bucketing = GMBucketingResults(gm_bucketing_pempos) if gm_bucketing_pempos else None
        self.print_global_pairwise_scores = print_global_pairwise_scores
        self.print_global_voters = print_global_voters

    def map_entity(self, nnid, onlyname=False):
        wikiid = self.nnid2wikiid[nnid]
        wikiname = self.wiki_id_name_map[wikiid].replace(' ', '_') if wikiid != "<u>" else "<u>"
        return wikiname if onlyname else "{} {}".format(wikiid, wikiname)

    def process_file(self, el_mode, name, opt_thr):
        self.thr = opt_thr
        self.el_mode = el_mode
        filepath = self.predictions_folder + ("el/" if el_mode else "ed/") + name
        self.fout = open(filepath, "w")
        if self.gm_bucketing:
            self.gm_bucketing.reinitialize()

    def file_ended(self):
        self.fout.close()
        if self.gm_bucketing:
            self.gm_bucketing.print()

    def scores_text(self, scores_l, scores_names_l, i, j):
        return ' '.join([scores_name + "=" + str(score[i][j]) for scores_name, score in zip(scores_names_l, scores_l)])

    def process_sample(self, chunkid,
                       tp_pred, fp_pred, fn_pred, gt_minus_fn_pred,
                       words, words_len, chars, chars_len,
                       cand_entities, cand_entities_len,
                       final_scores, filtered_spans, scores_l, scores_names_l, gmask, entity_embeddings):
        """words: [None] 1d the words of a sample, words_len: scalar,
        chars: [None, None] 2d  words, chars of each word, chars_len: [None] for each word
        the length in terms of characters.
        cand_entities: [None, None]  gold_mentions, candidates for each gm,
        cand_entitites_len: [None]  how many cand ent each gm has.
        filtered_spans = [span1, span2,...] sorted in terms of score. each span is a tuple
        (score, begin_idx, end_idx, best_nnid, simil_score, best_position 1-30, span_num)
        tp_pred and fp_pred is also a list of spans like above and it is also sorted for score.
        fn_pred is a [(gm_num, begin_gm, end_gm, gt)]"""
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

        span_num_b_e_gt = sorted(fn_pred+gt_minus_fn_pred)

        text_tags = defaultdict(list)
        gt_legend = []
        if len(fn_pred) > 0:
            fnWeakMatcherLogging = FNWeakMatcherLogging(self, filtered_spans, cand_entities,
                                        cand_entities_len, final_scores, scores_l, scores_names_l,
                                        reconstructed_words, self.gm_bucketing, gmask, entity_embeddings,
                                        span_num_b_e_gt)
        for mylist, mycolor in zip([gt_minus_fn_pred, fn_pred], ["green", "red"]):
            for i, (gm_num, b, e, gt) in enumerate(mylist, 1):
                text_tags[b].append((1, colored("[{}".format(i), mycolor)))
                text_tags[e].append((0, colored("]", mycolor)))

                gt_text = ""
                if self.el_mode is False:  # find the position and the score of the ground truth
                    gt_text = "gt not in candidate entities (recall miss)"
                    for j in range(cand_entities_len[gm_num]):
                        if cand_entities[gm_num][j] == gt:
                            gt_text = "gt_p_e_m_pos={}, {}".format(j,
                                            self.scores_text(scores_l, scores_names_l, gm_num, j))
                            break
                text = "{}: {} {}".format(i, self.map_entity(gt), gt_text)
                if mycolor == "red":
                    text += fnWeakMatcherLogging.check(gm_num, b, e, gt)
                text = colored(text, mycolor)
                gt_legend.append(text)

        tp_legend = []
        tp_pred = sorted(tp_pred, key=operator.itemgetter(1))
        for i, (score, b, e, nnid, scores_text, p_e_m_pos, span_num) in enumerate(tp_pred, 1):
            text_tags[b].append((1, colored("[{}".format(i), "blue")))
            text_tags[e].append((0, colored("]", "blue")))

            text = colored("{}: {}, score={}, {}, pem_pos={}".format(i,
                        self.map_entity(nnid), score, scores_text, p_e_m_pos), "blue")
            tp_legend.append(text)
            if self.gm_bucketing:
                self.gm_bucketing.process_tp(p_e_m_pos, cand_entities_len[span_num])

        fp_legend = []
        fp_pairwise_scores_legend = []
        fp_pred = sorted(fp_pred, key=operator.itemgetter(1))
        if len(fp_pred) > 0:
            fpWeakMatcherLogging = FPWeakMatcherLogging(self, span_num_b_e_gt, #fn_pred+gt_minus_fn_pred,
                                    cand_entities, cand_entities_len,
                                    final_scores, scores_l, scores_names_l, reconstructed_words, self.gm_bucketing,
                                                        gmask, entity_embeddings)
        for i, (score, b, e, nnid, scores_text, p_e_m_pos, span_num) in enumerate(fp_pred, 1):
            text_tags[b].append((1, colored("[{}".format(i), "magenta")))
            text_tags[e].append((0, colored("]", "magenta")))

            fp_gt_text, pairwise_score_text = fpWeakMatcherLogging.check(b, e, span_num, p_e_m_pos)
            text = "{}: {}, score={}, {}, pem_pos={} {} ".format(i,
                            self.map_entity(nnid), score, scores_text, p_e_m_pos,
                            fp_gt_text)
            fp_legend.append(colored(text, "magenta"))
            fp_pairwise_scores_legend.append("\n"+text)
            fp_pairwise_scores_legend.append(pairwise_score_text)

        final_acc = ["new sample " + chunkid+"\n"+self.extra_info+"\n"]
        for i in range(words_len+1):
            final_acc.extend([text for _, text in sorted(text_tags[i])])
            if i < words_len:
                final_acc.append(reconstructed_words[i])
        self.fout.write(" ".join(final_acc)+"\n")
        if self.print_global_voters:
            self.fout.write("global score voters and weights:\n")
            gmask_print_string = self.print_gmask(gmask, span_num_b_e_gt, reconstructed_words, cand_entities)
            self.fout.write(gmask_print_string+"\n")
        self.fout.write("\n".join(gt_legend + tp_legend + fp_legend))
        if self.print_global_pairwise_scores:
            self.fout.write(colored("\n".join(fp_pairwise_scores_legend), "grey"))
        self.fout.write("\n")

    def print_gmask(self, gmask, span_num_b_e_gt, reconstructed_words, cand_entities):
        i = 0
        document_gmask_acc = []
        for span_num, b, e, gt in span_num_b_e_gt:
            assert(i == span_num)
            text_acc = ["mention {} {}: ".format(span_num, ' '.join(reconstructed_words[b:e]))]
            for cand_ent_pos in range(gmask.shape[1]):
                mask_value = gmask[span_num][cand_ent_pos]
                assert(mask_value >= 0)
                if mask_value > 0:
                    text_acc.append("{} {:.2f} | ".format(self.map_entity(cand_entities[span_num][cand_ent_pos]),
                                                          mask_value))
            i += 1
            document_gmask_acc.append(' '.join(text_acc))
        return '\n'.join(document_gmask_acc)


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
    def __init__(self, printPredictions, span_num_b_e_gt, cand_entities, cand_entities_len,
                 final_scores, scores_l, scores_names_l, reconstructed_words, gm_bucketing=None,
                 gmask=None, entity_embeddings=None):
        self.printPredictions = printPredictions
        self.data = span_num_b_e_gt
        self.cand_entities = cand_entities
        self.cand_entities_len = cand_entities_len
        self.final_scores = final_scores
        self.scores_l = scores_l
        self.scores_names_l = scores_names_l
        self.reconstructed_words = reconstructed_words
        self.gm_bucketing = gm_bucketing
        self.gmask = gmask
        self.entity_embeddings = entity_embeddings

    def check(self, s, e, span_num, winner_pos=None):
        # all the above information that i have for my best_cand_id, now i have to find them
        # for the gt of the gm that overlap with my fp tuple.
        # compare my tuple s, e with all the gm

        acc = []
        pairwise_scores_text = ""
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
            # check all the candidate entities of this fp span and find where is the gt
            # of course we may not find it at all (recall miss)
            gt_cand_position = -1
            for j in range(self.cand_entities_len[span_num]):
                if self.cand_entities[span_num][j] == gt:
                    gt_cand_position = j
                    break

            if gt_cand_position >= 0:
                acc.append("| {}, score={}, {}, pem_pos={}".format(
                        self.printPredictions.map_entity(gt),
                        self.final_scores[span_num][gt_cand_position],
                        self.printPredictions.scores_text(self.scores_l, self.scores_names_l, span_num, gt_cand_position),
                        gt_cand_position))
            else:
                acc.append("| {}, recall miss".format(self.printPredictions.map_entity(gt)))
            if self.printPredictions.print_global_pairwise_scores:
                pairwise_scores_text = print_global_pairwise_voting(self.gmask, self.data, self.reconstructed_words,
                                                        self.cand_entities, self.printPredictions,
                                                        self.entity_embeddings, span_num, winner_pos,
                                                        gt_cand_position)

        if acc == []:
            acc.append("| no overlap with gm")

        return ' '.join(acc), pairwise_scores_text


class FNWeakMatcherLogging(object):
    """ This is used to produce text for the FN.
    From the filtered spans i.e. the spans that we keep that do not overlap with each other
    filtered_spans: [(best_cand_score, begin_idx, end_idx, best_cand_id,
                          scores_text, best_cand_position, span_num),(),...]"""
    def __init__(self, printPredictions, filtered_spans, cand_entities, cand_entities_len,
                 final_scores, scores_l, scores_names_l, reconstructed_words, gm_bucketing=None,
                 gmask=None, entity_embeddings=None, span_num_b_e_gt=None):
        self.printPredictions = printPredictions
        self.data = filtered_spans
        self.cand_entities = cand_entities
        self.cand_entities_len = cand_entities_len
        self.scores_l = scores_l
        self.scores_names_l = scores_names_l
        self.final_scores = final_scores
        self.reconstructed_words = reconstructed_words
        self.gm_bucketing = gm_bucketing
        self.gmask = gmask
        self.entity_embeddings = entity_embeddings
        self.span_num_b_e_gt = span_num_b_e_gt

    def check(self, gm_num, s, e, gt):
        # now I will compare each possible span of filtered_spans and for each of them if they overlap
        # with the fn mention I will print what was their winner entity (even if it was below the
        # threshold) and what was the assigned score to the gt (if it was a candidate)
        acc = []
        for (best_cand_score, s2, e2, best_cand_id, scores_text, best_cand_position, span_num) in self.data:
            overlap = False  # overlap with this specific filtered span
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

            # add to the text accumulator the info for this filtered span
            # print winner of this span info plus gt info: find gt_score, gt_cand_position
            # check all the candidate entities of this span and find where is the gt
            # of course we may not find it at all (recall miss)
            gt_cand_position = -1
            for j in range(self.cand_entities_len[span_num]):
                if self.cand_entities[span_num][j] == gt:
                    gt_cand_position = j
                    break

            assert(abs(best_cand_score - self.final_scores[span_num][best_cand_position]) < 0.001)
            acc.append("[span: {} winner: {}, score={}, {}, pem_pos={}".format(
                ' '.join(self.reconstructed_words[s2:e2]),
                self.printPredictions.map_entity(best_cand_id),
                best_cand_score,
                self.printPredictions.scores_text(self.scores_l, self.scores_names_l, span_num, best_cand_position),
                best_cand_position))
            if gt_cand_position >= 0:
                acc.append(" | gt: {}, score={}, {}, pem_pos={} ]".format(
                    self.printPredictions.map_entity(gt),
                    self.final_scores[span_num][gt_cand_position],
                    self.printPredictions.scores_text(self.scores_l, self.scores_names_l, span_num, gt_cand_position),
                    gt_cand_position))
                if self.gm_bucketing:
                    self.gm_bucketing.process_fn(gt_cand_position, best_cand_id == gt, self.cand_entities_len[span_num])
            else:
                acc.append(" | {}, recall miss".format(self.printPredictions.map_entity(gt)))
            if False and self.printPredictions.print_global_pairwise_scores:
                acc.append(print_global_pairwise_voting(self.gmask, self.span_num_b_e_gt, self.reconstructed_words,
                                                        self.cand_entities, self.printPredictions,
                                                        self.entity_embeddings, span_num, best_cand_position,
                                                        gt_cand_position))

        if acc == []:
            acc.append(" | no overlap with any filtered span")

        return ' '.join(acc)



# TODO for ed it works well. for EL we have more spans than just the gold mentions that vote.
# pass as parameters the begin_spans, end_spans from metrics.py
def print_global_pairwise_voting(gmask, span_num_b_e_gt, reconstructed_words, cand_entities, printPredictions,
                                 entity_embeddings, span_num, winner_pos, gt_pos):
    i = 0
    return_acc = ["'winner & gt' score given by each global voter"]
    winner_score_sum = 0
    gt_score_sum = 0
    voters_cnt = 0
    for other_span, b, e, _ in span_num_b_e_gt:
        assert(i == other_span)
        if other_span == span_num:
            i += 1
            continue   #only the other spans vote
        mention_acc = ["mention {} {}: ".format(other_span, ' '.join(reconstructed_words[b:e]))]
        for cand_ent_pos in range(gmask.shape[1]):
            mask_value = gmask[other_span][cand_ent_pos]
            assert(mask_value >= 0)
            if mask_value > 0:
                winner_score = np.dot(entity_embeddings[other_span][cand_ent_pos],
                                      entity_embeddings[span_num][winner_pos]) * mask_value
                gt_score = np.dot(entity_embeddings[other_span][cand_ent_pos],
                                      entity_embeddings[span_num][gt_pos]) * mask_value
                winner_score_sum += winner_score
                gt_score_sum += gt_score
                voters_cnt += 1
                mention_acc.append("{} {:.2f} & {:.2f} |".format(
                    printPredictions.map_entity(cand_entities[other_span][cand_ent_pos], onlyname=True),
                                                      winner_score, gt_score))
        i += 1
        return_acc.append(' '.join(mention_acc))
    return_acc.append("global winner_score_avg = {:.2f}   gt_score_avg = {:.2f}".format(
        winner_score_sum/voters_cnt, gt_score_sum/voters_cnt))
    return '\n'.join(return_acc)
