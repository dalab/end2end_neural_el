import numpy as np
from collections import defaultdict
from operator import itemgetter
import tensorflow as tf


class Evaluator(object):
    def __init__(self, threshold, name):
        self.threshold = threshold
        self.name = name
        self.TP = defaultdict(int)    # docid -> counter
        self.FP = defaultdict(int)    # docid -> counter
        self.FN = defaultdict(int)    # docid -> counter
        self.docs = set()             # set with all the docid encountered
        self.gm_num = 0

    def gm_add(self, gm_in_batch):
        self.gm_num += gm_in_batch

    def check_tp(self, score, docid):
        if score >= self.threshold:
            self.docs.add(docid)
            self.TP[docid] += 1
            return True
        return False

    def check_fp(self, score, docid):
        if score >= self.threshold:
            self.docs.add(docid)
            self.FP[docid] += 1
            return True
        return False

    def check_fn(self, score, docid):
        if score < self.threshold:
            self.docs.add(docid)
            self.FN[docid] += 1
            return True
        return False

    def _score_computation(self, el_mode):
        micro_tp, micro_fp, micro_fn = 0, 0, 0
        macro_pr, macro_re = 0, 0

        for docid in self.docs:
            tp, fp, fn = self.TP[docid], self.FP[docid], self.FN[docid]
            micro_tp += tp
            micro_fp += fp
            micro_fn += fn

            doc_precision = tp / (tp + fp + 1e-6)
            macro_pr += doc_precision

            doc_recall = tp / (tp + fn + 1e-6)
            macro_re += doc_recall

        if el_mode is False:
            assert(self.gm_num == micro_tp + micro_fn)

        micro_pr = 100 * micro_tp / (micro_tp + micro_fp + 1e-6)
        micro_re = 100 * micro_tp / (micro_tp + micro_fn + 1e-6)
        micro_f1 = 2*micro_pr*micro_re / (micro_pr + micro_re + 1e-6)

        macro_pr = 100 * macro_pr / len(self.docs)
        macro_re = 100 * macro_re / len(self.docs)
        macro_f1 = 2*macro_pr*macro_re / (macro_pr + macro_re + 1e-6)

        return micro_pr, micro_re, micro_f1, macro_pr, macro_re, macro_f1

    def print_log_results(self, tf_writer, eval_cnt, el_mode):
        micro_pr, micro_re, micro_f1, macro_pr, macro_re, macro_f1 = self._score_computation(el_mode)

        print("micro", "P: %.1f" % micro_pr, "\tR: %.1f" % micro_re, "\tF1: %.1f" % micro_f1)
        print("macro", "P: %.1f" % macro_pr, "\tR: %.1f" % macro_re, "\tF1: %.1f" % macro_f1)

        if tf_writer is None:
            return micro_f1, macro_f1


        name = self.name+" macro"
        writer_name = "el_" if el_mode else "ed_"
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=macro_f1)])
        tf_writer[writer_name+"f1"].add_summary(summary, eval_cnt)
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=macro_pr)])
        tf_writer[writer_name+"pr"].add_summary(summary, eval_cnt)
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=macro_re)])
        tf_writer[writer_name+"re"].add_summary(summary, eval_cnt)

        name = self.name+" micro"
        writer_name = "el_" if el_mode else "ed_"
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=micro_f1)])
        tf_writer[writer_name+"f1"].add_summary(summary, eval_cnt)
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=micro_pr)])
        tf_writer[writer_name+"pr"].add_summary(summary, eval_cnt)
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=micro_re)])
        tf_writer[writer_name+"re"].add_summary(summary, eval_cnt)

        return micro_f1, macro_f1

    def print_log_results_old(self, tf_writer, eval_cnt, el_mode):
        micro_tp, micro_fp, micro_fn = 0, 0, 0
        macro_pr, macro_re = 0, 0

        try:
            valid_macro_prec_cnt = 0
            valid_macro_recall_cnt = 0
            for docid in self.docs:
                tp, fp, fn = self.TP[docid], self.FP[docid], self.FN[docid]
                micro_tp += tp
                micro_fp += fp
                micro_fn += fn

                if tp + fp > 0:
                    doc_precision = tp / (tp + fp)
                    macro_pr += doc_precision
                    valid_macro_prec_cnt += 1
                if tp + fn > 0:
                    doc_recall = tp / (tp + fn)
                    macro_re += doc_recall
                    valid_macro_recall_cnt += 1

            if el_mode is False:
                assert(self.gm_num == micro_tp + micro_fn)

            micro_pr = 100 * micro_tp / (micro_tp + micro_fp)     if (micro_tp + micro_fp) > 0 else 0
            micro_re = 100 * micro_tp / (micro_tp + micro_fn)     if (micro_tp + micro_fn) > 0 else 0
            micro_f1 = 2*micro_pr*micro_re / (micro_pr + micro_re)   if (micro_pr + micro_re) > 0 else 0

            macro_pr = 100 * macro_pr / valid_macro_prec_cnt      if valid_macro_prec_cnt > 0 else 0
            macro_re = 100 * macro_re / valid_macro_recall_cnt    if valid_macro_recall_cnt > 0 else 0
            macro_f1 = 2*macro_pr*macro_re / (macro_pr + macro_re)   if (macro_pr + macro_re) > 0 else 0
        except ZeroDivisionError:
            print("Exception! ZeroDivisionError in print results!\nmicro_tp, micro_fp, micro_fn = ", micro_tp,
                  micro_fp, micro_fn)

        print("micro", "P: %.1f" % micro_pr, "\tR: %.1f" % micro_re, "\tF1: %.1f" % micro_f1)
        print("macro", "P: %.1f" % macro_pr, "\tR: %.1f" % macro_re, "\tF1: %.1f" % macro_f1)

        if tf_writer is None:
            print("len(self.docs)={}\tvalid_macro_prec_cnt={}\tvalid_macro_recall_cnt={}".format(
                len(self.docs), valid_macro_prec_cnt, valid_macro_recall_cnt))
            return micro_f1, macro_f1


        name = self.name+" macro"
        writer_name = "el_" if el_mode else "ed_"
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=macro_f1)])
        tf_writer[writer_name+"f1"].add_summary(summary, eval_cnt)
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=macro_pr)])
        tf_writer[writer_name+"pr"].add_summary(summary, eval_cnt)
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=macro_re)])
        tf_writer[writer_name+"re"].add_summary(summary, eval_cnt)

        name = self.name+" micro"
        writer_name = "el_" if el_mode else "ed_"
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=micro_f1)])
        tf_writer[writer_name+"f1"].add_summary(summary, eval_cnt)
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=micro_pr)])
        tf_writer[writer_name+"pr"].add_summary(summary, eval_cnt)
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=micro_re)])
        tf_writer[writer_name+"re"].add_summary(summary, eval_cnt)

        return micro_f1, macro_f1

class StrongMatcher(object):
    """is initialized with the gm_gt_list i.e. a list of tuples
    (begin_idx, end_idx, gt) and from the list of tuples it builds a set of tuples
    that will help us answer if our prediction matches with a tuple from the
    ground truth"""
    def __init__(self, b_e_gt_iterator):
        self.data = set()   # of tuples (begin_idx, end_idx, gt)
        for t in b_e_gt_iterator:
            self.data.add(t)

    def check(self, t):
        """returns True if tuple matches with ground truth else False"""
        return True if t in self.data else False


class WeakMatcher(object):
    """is initialized with the gm_gt_list i.e. a list of tuples
    (begin_idx, end_idx, gt) and from the list of tuples it builds a data structure
    that will help us answer if our prediction matches with a tuple from the
    ground truth.
    structure used: a dict with key the gt and value a list of tuples
    (begin_idx, end_idx). So i compare the predicted triplet (b,e,ent_id)
    with all the ground truth triplets and check
    if they overlap (weak matching)  and return True or False.
    e.g.  4 -> [(5,7), (13,14)] """
    def __init__(self, b_e_gt_iterator):
        self.data = defaultdict(list)
        for b, e, gt in b_e_gt_iterator:
            self.data[gt].append((b, e))

    def check(self, t):
        # here the t comes from filtereds_spans[1:] so begin_idx, end_idxm best_cand_id but name it gt in the code
        s, e, gt = t
        if gt in self.data:
            for s2, e2 in self.data[gt]:
                if s<=s2 and e<=e2 and s2<e:
                    return True
                elif s>=s2 and e>=e2 and s<e2:
                    return True
                elif s<=s2 and e>=e2:
                    return True
                elif s>=s2 and e<=e2:
                    return True
        return False


class FNStrongMatcher(object):
    """when initialized it takes our algorithms predictions
    (score, begin_idx, end_idx, ent_id) list and builds a dictionary.
    later we use it to check what score we have given to the ground truth i.e.
    gold mention plus the correct entity.
    structure used: a dict with key (begin_idx, end_idx, ent_id) --> given_score
    by my algorithm"""
    def __init__(self, filtered_spans):
        self.data = dict()
        for score, b, e, ent_id in filtered_spans:
            self.data[(b, e, ent_id)] = score

    def check(self, t):
        """t are tuples (begin_idx, end_idx, gt) from gm_gt_list. I check
        if the ground truth is in my predictions and return the given score."""
        return self.data[t] if t in self.data else -10000


class FNWeakMatcher(object):
    """when initialized it takes our algorithms predictions
    (score, begin_idx, end_idx, ent_id) list and builds a data structure.
    later we use it to check what score we have given to the ground truth i.e.
    gold mention plus the correct entity.
    structure used: # a dict with key the gt and value a list of tuples
    (begin_idx, end_idx, given_score). So i compare the ground truth triplet (s,e,gt)
    with all the spans that my algorithm has linked to the same entity (gt) and check
    if they overlap (data matching) and return the highest score.
    e.g.  4 -> [(5,7, 0.2), (13,14, 0.3)] """
    def __init__(self, filtered_spans):
        self.data = defaultdict(list)
        for score, b, e, ent_id in filtered_spans:
            self.data[ent_id].append((b, e, score))

    def check(self, t):
        """t are tuples (begin_idx, end_idx, gt) from gm_gt_list. I check
        if the ground truth has overlap with some of my predictions and return
        the highest given score."""
        s, e, gt = t
        best_score = -10000
        if gt in self.data:
            for s2, e2, score in self.data[gt]:
                if s<=s2 and e<=e2 and s2<e:
                    best_score = max(best_score, score)
                elif s>=s2 and e>=e2 and s<e2:
                    best_score = max(best_score, score)
                elif s<=s2 and e>=e2:
                    best_score = max(best_score, score)
                elif s>=s2 and e<=e2:
                    best_score = max(best_score, score)
        return best_score


def _filtered_spans_and_gm_gt_list(b, final_scores, cand_entities_len, cand_entities,
                                   begin_span, end_span, spans_len,
                                   begin_gm, end_gm, ground_truth,
                                   ground_truth_len, words_len):
    spans = []
    for i in range(spans_len[b]):  # candidate span
        begin_idx = begin_span[b][i]
        end_idx = end_span[b][i]

        best_cand_id = -1
        best_cand_score = -10000
        for j in range(cand_entities_len[b][i]):  # how many candidate entities we have for this span
            score = final_scores[b][i][j]
            if score > best_cand_score:
                best_cand_score = score
                best_cand_id = cand_entities[b][i][j]

        spans.append((best_cand_score, begin_idx, end_idx, best_cand_id))

    # now filter this list of spans based on score. from the overlapping ones keep the one
    # with the highest score.
    spans = sorted(spans, reverse=True)  # highest score          lowest score
    filtered_spans = []
    claimed = np.full(words_len[b], False, dtype=bool)  # initially all words are free to select
    for span in spans:
        best_cand_score, begin_idx, end_idx, best_cand_id = span
        if not np.any(claimed[begin_idx:end_idx]) and best_cand_id > 0:
            # nothing is claimed so take it   TODO this > 0 condition is it correct???
            claimed[begin_idx:end_idx] = True
            filtered_spans.append(span)

    gm_gt_list = [(begin_gm[b][i], end_gm[b][i], ground_truth[b][i]) for i in range(ground_truth_len[b])]

    return filtered_spans, gm_gt_list


def threshold_calculation(final_scores, cand_entities_len, cand_entities,
                          begin_span, end_span, spans_len, begin_gm, end_gm, ground_truth,
                          ground_truth_len, words_len, chunk_id, el_mode):
    tp_fp_batch_scores = []
    fn_batch_scores = []
    if el_mode is False:
        begin_gm = begin_span
        end_gm = end_span
    for b in range(final_scores.shape[0]):  # batch
        filtered_spans, gm_gt_list = _filtered_spans_and_gm_gt_list(b, final_scores, cand_entities_len, cand_entities,
                                                                    begin_span, end_span, spans_len, begin_gm, end_gm, ground_truth, ground_truth_len, words_len)
        matcher = WeakMatcher(gm_gt_list) if el_mode else StrongMatcher(gm_gt_list)
        for t in filtered_spans:
            if matcher.check(t[1:]):
                tp_fp_batch_scores.append((t[0], 1))   # (score, TP)
            else:
                tp_fp_batch_scores.append((t[0], 0))   # (score, FP)

        # now check for the fn
        matcher = FNWeakMatcher(filtered_spans) if el_mode else FNStrongMatcher(filtered_spans)
        for t in gm_gt_list:
            score = matcher.check(t)
            fn_batch_scores.append(score)

    return tp_fp_batch_scores, fn_batch_scores


def metrics_calculation(evaluator, final_scores, cand_entities_len, cand_entities,
                        begin_span, end_span, spans_len,
                        begin_gm, end_gm, ground_truth,
                        ground_truth_len, words_len, chunk_id, el_mode):
    if el_mode is False:
        begin_gm = begin_span
        end_gm = end_span
    # for each candidate span find which is the cand entity with the highest score
    for b in range(final_scores.shape[0]):  # batch
        filtered_spans, gm_gt_list = _filtered_spans_and_gm_gt_list(b, final_scores, cand_entities_len, cand_entities,
                                                                    begin_span, end_span, spans_len, begin_gm, end_gm, ground_truth, ground_truth_len, words_len)
        matcher = WeakMatcher(gm_gt_list) if el_mode else StrongMatcher(gm_gt_list)
        docid = chunk_id[b].split(b"&*", 1)[0]  # b'947testa_CRICKET&*0&*0'     to     b'947testa_CRICKET'
        # TODO remove this and the assertion
        evaluator.gm_add(len(gm_gt_list))
        for t in filtered_spans:
            if matcher.check(t[1:]):
                evaluator.check_tp(t[0], docid)
            else:
                evaluator.check_fp(t[0], docid)

        # now check for the fn
        matcher = FNWeakMatcher(filtered_spans) if el_mode else FNStrongMatcher(filtered_spans)
        for t in gm_gt_list:
            score = matcher.check(t)
            evaluator.check_fn(score, docid)


def metrics_calculation_and_prediction_printing(evaluator, final_scores,
                                                cand_entities_len, cand_entities,
                                                begin_span, end_span, spans_len,
                                                begin_gm, end_gm, ground_truth,
                                                ground_truth_len, words_len, chunk_id,
                                                words, chars, chars_len,
                                                scores_l, global_pairwise_scores, scores_names_l,
                                                el_mode, printPredictions=None):
    if el_mode is False:
        begin_gm = begin_span
        end_gm = end_span
    # for each candidate span find which is the cand entity with the highest score
    for b in range(final_scores.shape[0]):  # batch
        spans = []
        for i in range(spans_len[b]):  # candidate span
            begin_idx = begin_span[b][i]
            end_idx = end_span[b][i]

            best_cand_id = -1
            best_cand_score = -10000
            best_cand_similarity_score = -10000
            best_cand_position = -1
            scores_text = "invalid"
            for j in range(cand_entities_len[b][i]):  # how many candidate entities we have for this span
                score = final_scores[b][i][j]
                if score > best_cand_score:
                    best_cand_score = score
                    best_cand_id = cand_entities[b][i][j]
                    scores_text = ' '.join([scores_name + "=" + str(score[b][i][j]) for scores_name, score in zip(scores_names_l, scores_l)])
                    # best_cand_similarity_score = similarity_scores[b][i][j]
                    best_cand_position = j

            span_num = i
            spans.append((best_cand_score, begin_idx, end_idx, best_cand_id,
                          scores_text, best_cand_position, span_num))

        # now filter this list of spans based on score. from the overlapping ones keep the one
        # with the highest score.
        spans = sorted(spans, reverse=True)  # highest score          lowest score
        filtered_spans = []
        claimed = np.full(words_len[b], False, dtype=bool)  # initially all words are free to select
        for span in spans:
            best_cand_score, begin_idx, end_idx, best_cand_id = span[:4]
            if not np.any(claimed[begin_idx:end_idx]) and best_cand_id > 0:
                # nothing is claimed so take it   TODO this > 0 condition is it correct???
                claimed[begin_idx:end_idx] = True
                filtered_spans.append(span)

        # now traverse all the filtered spans and compare them with the gold mentions
        # for each tuple of filtered_spans check for tp or fp
        gm_gt_list = [(begin_gm[b][i], end_gm[b][i], ground_truth[b][i]) for i in range(ground_truth_len[b])]
        matcher = WeakMatcher(gm_gt_list) if el_mode else StrongMatcher(gm_gt_list)

        docid = chunk_id[b].split(b"&*", 1)[0]
        evaluator.gm_add(len(gm_gt_list))

        tp_pred = []
        fp_pred = []
        fn_pred = []
        gt_minus_fn_pred = []  # gt_minus_fn_pred + fn_pred create the gm_gt_list
        for t in filtered_spans:
            if matcher.check(t[1:4]):
                if evaluator.check_tp(t[0], docid):
                    tp_pred.append(t)
            else:
                if evaluator.check_fp(t[0], docid):
                    fp_pred.append(t)

        # now check for the fn
        temp = [t[:4] for t in filtered_spans]
        matcher = FNWeakMatcher(temp) if el_mode else FNStrongMatcher(temp)
        for gm_num, t in enumerate(gm_gt_list):
            score = matcher.check(t)
            if evaluator.check_fn(score, docid):
                fn_pred.append((gm_num, *t))
            else:
                gt_minus_fn_pred.append((gm_num, *t))

        if printPredictions is not None:
            gmask = global_pairwise_scores[0][b] if global_pairwise_scores else None
            entity_embeddings = global_pairwise_scores[1][b] if global_pairwise_scores else None
            printPredictions.process_sample(str(chunk_id[b]),
                                            tp_pred, fp_pred, fn_pred, gt_minus_fn_pred,
                                            words[b], words_len[b],
                                            chars[b], chars_len[b],
                                            cand_entities[b], cand_entities_len[b],
                                            final_scores[b], filtered_spans,
                                            [score[b] for score in scores_l], scores_names_l,
                                            gmask, entity_embeddings)


