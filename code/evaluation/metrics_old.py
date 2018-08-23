import numpy as np
from collections import defaultdict
from operator import itemgetter
import tensorflow as tf


class Evaluator_aux(object):
    def __init__(self, threshold, name):
        self.threshold = threshold
        self.name = name
        self.TP = defaultdict(int)    # docid -> counter
        self.FP = defaultdict(int)    # docid -> counter
        self.FN = defaultdict(int)    # docid -> counter
        self.docs = set()             # set with all the docid encountered

    def check_tp(self, score, docid):
        if score >= self.threshold:
            self.docs.add(docid)
            self.TP[docid] += 1

    def check_fp(self, score, docid):
        if score >= self.threshold:
            self.docs.add(docid)
            self.FP[docid] += 1

    def check_fn(self, score, docid):
        if score < self.threshold:
            self.docs.add(docid)
            self.FN[docid] += 1

    def print_results(self):
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


            micro_pr = micro_tp / (micro_tp + micro_fp)     if (micro_tp + micro_fp) > 0 else 0
            micro_re = micro_tp / (micro_tp + micro_fn)     if (micro_tp + micro_fn) > 0 else 0
            micro_f1 = 2*micro_pr*micro_re / (micro_pr + micro_re)   if (micro_pr + micro_re) > 0 else 0

            macro_pr = macro_pr / valid_macro_prec_cnt      if valid_macro_prec_cnt > 0 else 0
            macro_re = macro_re / valid_macro_recall_cnt    if valid_macro_recall_cnt > 0 else 0
            macro_f1 = 2*macro_pr*macro_re / (macro_pr + macro_re)   if (macro_pr + macro_re) > 0 else 0
        except ZeroDivisionError:
            print("Exception! ZeroDivisionError in print results!\nmicro_tp, micro_fp, micro_fn = ", micro_tp,
                  micro_fp, micro_fn)

        print(self.name, "thr", self.threshold)
        print("micro", "P:", micro_pr, "\tR:", micro_re, "\tF1:", micro_f1)
        print("macro", "P:", macro_pr, "\tR:", macro_re, "\tF1:", macro_f1)

        return micro_f1, macro_f1, self.threshold

class Evaluator(object):
    def __init__(self, weak_thr=None, strong_thr=None, name=""):
        self.weak_evaluators = []
        self.strong_evaluators = []
        self.name = name
        for thr in weak_thr:
            self.weak_evaluators.append(Evaluator_aux(thr, "data"))
        for thr in strong_thr:
            self.strong_evaluators.append(Evaluator_aux(thr, "strong"))

    def weak_check_tp(self, score, docid):
        #list(map(lambda x: x.check_tp(score, docid), self.weak_evaluators))
        for x in self.weak_evaluators:
            x.check_tp(score, docid)

    def weak_check_fp(self, score, docid):
        #map(lambda x: x.check_fp(score, docid), self.weak_evaluators)
        for x in self.weak_evaluators:
            x.check_fp(score, docid)

    def weak_check_fn(self, score, docid):
        #map(lambda x: x.check_fn(score, docid), self.weak_evaluators)
        for x in self.weak_evaluators:
            x.check_fn(score, docid)

    def strong_check_tp(self, score, docid):
        #map(lambda x: x.check_tp(score, docid), self.strong_evaluators)
        for x in self.strong_evaluators:
            x.check_tp(score, docid)

    def strong_check_fp(self, score, docid):
        #map(lambda x: x.check_fp(score, docid), self.strong_evaluators)
        for x in self.strong_evaluators:
            x.check_fp(score, docid)

    def strong_check_fn(self, score, docid):
        #map(lambda x: x.check_fn(score, docid), self.strong_evaluators)
        for x in self.strong_evaluators:
            x.check_fn(score, docid)

    def print_log_results(self, writer, eval_cnt):
        weak_scores = [x.print_results() for x in self.weak_evaluators]
        strong_scores = [x.print_results() for x in self.strong_evaluators]
        if writer is not None:
            for micro_f1, macro_f1, threshold in weak_scores:
                name = self.name+" data " + str(threshold)
                summary = tf.Summary(value=[tf.Summary.Value(tag=name+" micro_f1",
                                            simple_value=micro_f1)])
                writer.add_summary(summary, eval_cnt)
                summary = tf.Summary(value=[tf.Summary.Value(tag=name+" macro_f1",
                                                             simple_value=macro_f1)])
                writer.add_summary(summary, eval_cnt)

            for micro_f1, macro_f1, threshold in strong_scores:
                name = self.name+" strong " + str(threshold)
                summary = tf.Summary(value=[tf.Summary.Value(tag=name+" micro_f1",
                                                             simple_value=micro_f1)])
                writer.add_summary(summary, eval_cnt)
                summary = tf.Summary(value=[tf.Summary.Value(tag=name+" macro_f1",
                                                             simple_value=macro_f1)])
                writer.add_summary(summary, eval_cnt)

        result_list = weak_scores if self.weak_evaluators != [] else strong_scores
        return max(result_list, key=itemgetter(0))[0]


class WeakStrongMatching(object):
    def __init__(self, b_e_gt_iterator):
        self.exact = set()   # of tuples (begin_idx, end_idx, gt)
        self.weak = defaultdict(list)   # a map with key the gt and value a list of tuples
        # e.g.  4 -> [(5,7), (13,14)]
        # so in order to check for data match a search my gt if in the
        # data dictionary and if yes then check one by one overlap with
        # the span

        for b, e, gt in b_e_gt_iterator:
            self.exact.add((b, e, gt))
            self.weak[gt].append((b, e))

    def strong_check(self, t):
        return True if t in self.exact else False

    def weak_check(self, t):
        s, e, gt = t
        if gt in self.weak:
            for s2, e2 in self.weak[gt]:
                if s<=s2 and e<=e2 and s2<e:
                    return True
                elif s>=s2 and e>=e2 and s<e2:
                    return True
                elif s<=s2 and e>=e2:
                    return True
                elif s>=s2 and e<=e2:
                    return True

        return False


class FNWeakStrongMatching(object):
    def __init__(self, filtered_spans):
        self.weak = defaultdict(list)   # a map with key the gt and value a list of tuples
        # e.g.  4 -> [(5,7, 0.2), (13,14, 0.3)]
        # so in order to check for data match a search my gt if in the
        # data dictionary and if yes then check one by one overlap with
        # the span

        for score, b, e, ent_id in filtered_spans:
            self.weak[ent_id].append((b, e, score))


    def strong_check(self, t):
        s, e, gt = t
        best_score = -10000
        if gt in self.weak:
            for s2, e2, score in self.weak[gt]:
                if s==s2 and e==e2:
                    best_score = max(best_score, score)
        return best_score


    def weak_check(self, t):
        s, e, gt = t
        best_score = -10000
        if gt in self.weak:
            for s2, e2, score in self.weak[gt]:
                if s<=s2 and e<=e2 and s2<e:
                    best_score = max(best_score, score)
                elif s>=s2 and e>=e2 and s<e2:
                    best_score = max(best_score, score)
                elif s<=s2 and e>=e2:
                    best_score = max(best_score, score)
                elif s>=s2 and e<=e2:
                    best_score = max(best_score, score)
        return best_score


def validation_scores_calculation(evaluator, final_scores, cand_entities_len, cand_entities,
                                  begin_span, end_span, spans_len,
                                  begin_gm, end_gm, ground_truth,
                                  ground_truth_len, words_len, chunk_id, test_mode):
    if test_mode is False:
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


        # now traverse all the filtered spans and compare them with the gold mentions
        # for each tuple of filtered_spans check for tp or fp
        gm_gt_list = [(begin_gm[b][i], end_gm[b][i], ground_truth[b][i]) for i in range(ground_truth_len[b])]
        matcher = WeakStrongMatching(gm_gt_list)

        # b'947testa_CRICKET&*0&*0'     to     b'947testa_CRICKET'
        docid = chunk_id[b].split(b"&*", 1)[0]
        for t in filtered_spans:
            if matcher.strong_check(t[1:]):
                evaluator.strong_check_tp(t[0], docid)
            else:
                evaluator.strong_check_fp(t[0], docid)

            if matcher.weak_check(t[1:]):
                evaluator.weak_check_tp(t[0], docid)
            else:
                evaluator.weak_check_fp(t[0], docid)

        # now check for the fn
        matcher = FNWeakStrongMatching(filtered_spans)
        for t in gm_gt_list:
            score = matcher.strong_check(t)
            evaluator.strong_check_fn(score, docid)

            score = matcher.weak_check(t)
            evaluator.weak_check_fn(score, docid)


def evaluation_scores_calculation(evaluator, final_scores, cand_entities_len, cand_entities,
                                  begin_span, end_span, spans_len,
                                  begin_gm, end_gm, ground_truth,
                                  ground_truth_len, words_len, chunk_id, similarity_scores,
                                  words, chars, chars_len, cand_entities_scores,
                                  test_mode, printPredictions=None):
    if test_mode is False:
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
            for j in range(cand_entities_len[b][i]):  # how many candidate entities we have for this span
                score = final_scores[b][i][j]
                if score > best_cand_score:
                    best_cand_score = score
                    best_cand_id = cand_entities[b][i][j]
                    best_cand_similarity_score = similarity_scores[b][i][j]
                    best_cand_position = j

            spans.append((best_cand_score, begin_idx, end_idx, best_cand_id,
                          best_cand_similarity_score, best_cand_position))

        # now filter this list of spans based on score. from the overlapping ones keep the one
        # with the highest score.
        spans = sorted(spans, reverse=True)  # highest score          lowest score
        filtered_spans = []
        claimed = np.full(words_len[b], False, dtype=bool)  # initially all words are free to select
        for span in spans:
            best_cand_score, begin_idx, end_idx, best_cand_id, _, _ = span
            if not np.any(claimed[begin_idx:end_idx]) and best_cand_id > 0:
                # nothing is claimed so take it   TODO this > 0 condition is it correct???
                claimed[begin_idx:end_idx] = True
                filtered_spans.append(span)

        # now traverse all the filtered spans and compare them with the gold mentions
        # for each tuple of filtered_spans check for tp or fp
        gm_gt_list = [(begin_gm[b][i], end_gm[b][i], ground_truth[b][i]) for i in range(ground_truth_len[b])]
        matcher = WeakStrongMatching(gm_gt_list)

        # b'947testa_CRICKET&*0&*0'     to     b'947testa_CRICKET'
        docid = chunk_id[b].split(b"&*", 1)[0]

        tp_pred = []
        fp_pred = []
        fn_pred = []
        thr = printPredictions.thr if printPredictions is not None else 0.2
        for t in filtered_spans:
            if matcher.strong_check(t[1:-2]):
                evaluator.strong_check_tp(t[0], docid)
            else:
                evaluator.strong_check_fp(t[0], docid)

            if matcher.weak_check(t[1:-2]):
                evaluator.weak_check_tp(t[0], docid)
                if t[0] >= thr:
                    tp_pred.append(t)
            else:
                evaluator.weak_check_fp(t[0], docid)
                if t[0] >= thr:
                    fp_pred.append(t)

        # now check for the fn
        matcher = FNWeakStrongMatching(
                    [t[:-2] for t in filtered_spans])
        for t in gm_gt_list:
            score = matcher.strong_check(t)
            evaluator.strong_check_fn(score, docid)

            score = matcher.weak_check(t)
            evaluator.weak_check_fn(score, docid)
            if score < thr:
                fn_pred.append(t)

        if printPredictions is not None:
            printPredictions.process_sample(chunk_id[b], gm_gt_list,
                        tp_pred, fp_pred, fn_pred,
                        words[b], words_len[b],
                        chars[b], chars_len[b],
                        cand_entities[b], cand_entities_scores[b])
