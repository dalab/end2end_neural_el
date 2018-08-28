import argparse
import pickle
import model.config as config
import os
import tensorflow as tf
from model.model_ablations import Model
from evaluation.metrics import Evaluator, metrics_calculation_and_prediction_printing
import model.train as train
from model.util import load_train_args


def validation_loss_calculation(model, iterator, dataset_handle, opt_thr, el_mode, name=""):
    if args.print_predictions:
        printPredictions.process_file(el_mode, name, opt_thr)
    model.sess.run(iterator.initializer)
    evaluator = Evaluator(opt_thr, name=name)

    while True:
        try:
            retrieve_l = [model.final_scores,
                          model.cand_entities_len, model.cand_entities,
                          model.begin_span, model.end_span, model.spans_len,
                          model.begin_gm, model.end_gm,
                          model.ground_truth, model.ground_truth_len,
                          model.words_len, model.chunk_id,
                          model.words, model.chars, model.chars_len]
            scores_retrieve_l, scores_names_l = [], []
            if model.args.nn_components.find("lstm") != -1:
                scores_retrieve_l.append(model.similarity_scores)
                scores_names_l.append("lstm")
            if model.args.nn_components.find("pem") != -1:
                scores_retrieve_l.append(model.log_cand_entities_scores)
                scores_names_l.append("logpem")
            if model.args.nn_components.find("attention") != -1:
                scores_retrieve_l.append(model.attention_scores)
                scores_names_l.append("attention")
            if model.args.nn_components.find("global") != -1:
                scores_retrieve_l.append(model.final_scores_before_global)
                scores_names_l.append("before_global")
                scores_retrieve_l.append(model.global_voting_scores)
                scores_names_l.append("global_voting")
            global_pairwise_scores = []
            if args.print_global_voters:
                global_pairwise_scores.append(model.gmask)
                global_pairwise_scores.append(model.pure_entity_embeddings)

            retrieve_l.append(scores_retrieve_l)
            retrieve_l.append(global_pairwise_scores)
            result_l = model.sess.run(
                retrieve_l, feed_dict={model.input_handle_ph: dataset_handle, model.dropout: 1})
            metrics_calculation_and_prediction_printing(evaluator, *result_l, scores_names_l, el_mode,
                                          printPredictions=printPredictions)

        except tf.errors.OutOfRangeError:
            if args.print_predictions:
                printPredictions.file_ended()
            print(name)
            micro_f1, macro_f1 = evaluator.print_log_results(None, -1, el_mode)
            break
    return macro_f1

# identical with the train.compute_ed_el_scores
def compute_ed_el_scores(model, handles, names, iterators, el_mode):
    if args.hardcoded_thr:
        opt_thr = args.hardcoded_thr
        print("hardcoded threshold used:", opt_thr)
    else:
        # first compute the optimal threshold based on validation datasets.
        opt_thr, _ = train.optimal_thr_calc(model, handles, iterators, el_mode)
    # give the opt_thr and the projection variables to the PrintPredictions for insight
    if printPredictions:
        printPredictions.extra_info = print_thr_and_ffnn_values(model, opt_thr)

    results = []
    for test_handle, test_name, test_it in zip(handles, names, iterators):
        f1_score = validation_loss_calculation(model, test_it, test_handle, opt_thr,
                                               el_mode=el_mode, name=test_name)
        results.append(f1_score)
    return results


def evaluate():

    ed_datasets, ed_names = train.create_el_ed_pipelines(gmonly_flag=True, filenames=args.ed_datasets, args=args)
    el_datasets, el_names = train.create_el_ed_pipelines(gmonly_flag=False, filenames=args.el_datasets, args=args)

    input_handle_ph = tf.placeholder(tf.string, shape=[], name="input_handle_ph")
    sample_dataset = ed_datasets[0] if ed_datasets != [] else el_datasets[0]
    iterator = tf.data.Iterator.from_string_handle(
        input_handle_ph, sample_dataset.output_types, sample_dataset.output_shapes)
    next_element = iterator.get_next()

    model = Model(train_args, next_element)
    model.build()
    model.input_handle_ph = input_handle_ph    # just for convenience so i can access it from everywhere
    print(tf.global_variables())
    if args.p_e_m_algorithm:
        model.final_scores = model.cand_entities_scores


    def ed_el_dataset_handles(sess, datasets):
        test_iterators = []
        test_handles = []
        for dataset in datasets:
            test_iterator = dataset.make_initializable_iterator()
            test_iterators.append(test_iterator)
            test_handles.append(sess.run(test_iterator.string_handle()))
        return test_iterators, test_handles


    for el_mode, datasets, names in zip([False, True], [ed_datasets, el_datasets], [ed_names, el_names]):
        if names == []:
            continue
        model.restore_session("el" if el_mode else "ed")
        #print_variables_values(model)

        with model.sess as sess:
            print("Evaluating {} datasets".format("EL" if el_mode else "ED"))
            iterators, handles = ed_el_dataset_handles(sess, datasets)
            compute_ed_el_scores(model, handles, names, iterators, el_mode=el_mode)


# TODO delete this function
def print_variables_values(model):
    var_names = ['similarity_and_prior_ffnn/output_weights:0',
                 'similarity_and_prior_ffnn/output_bias:0']
    for var_name in var_names:
        var_handle = [v for v in tf.global_variables() if v.name == var_name][0]
        print(var_name)
        print(model.sess.run(var_handle))


def print_thr_and_ffnn_values(model, opt_thr):
    result = "opt_thr={}, nn_components={}\n".format(opt_thr, model.args.nn_components)
    var_names, var_print_names = [], []
    if model.args.final_score_ffnn[0] == 0:
        var_names.extend(['similarity_and_prior_ffnn/output_weights:0',
                          'similarity_and_prior_ffnn/output_bias:0'])
        var_print_names.extend(['lstm_pem_attention_weights', 'lstm_pem_attention_bias'])
    if model.args.nn_components.find("global") != -1 and model.args.global_score_ffnn[0] == 0:
        # print only if simple projection, otherwise it could be too many variables
        var_names.extend(['global_voting/psi_and_global_ffnn/output_weights:0',
                          'global_voting/psi_and_global_ffnn/output_bias:0'])
        var_print_names.extend(['psi_globalscore_weights', 'psi_globalscore_bias'])
    for var_name, print_name in zip(var_names, var_print_names):
        var_handle = [v for v in tf.global_variables() if v.name == var_name][0]
        result += print_name + "=" + str(model.sess.run(var_handle))
    return result


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default="alldatasets_perparagr", #"standard",
                        help="under folder data/tfrecords/")
    parser.add_argument("--training_name", help="under folder data/tfrecords/")
    parser.add_argument("--checkpoint_model_num", default=None,
                        help="e.g. give '7' or '4' if you want checkpoints/model-7 or model-4 "
                             "to be restored")

    parser.add_argument("--print_predictions", dest='print_predictions', action='store_true',
                        help="prints for each dataset the predictions to a file and compares with ground"
                             "truth and simple baselines.")
    parser.add_argument("--no_print_predictions", dest='print_predictions', action='store_false')
    parser.set_defaults(print_predictions=True)
    parser.add_argument("--print_global_voters", type=bool, default=False)
    parser.add_argument("--print_global_pairwise_scores", type=bool, default=False)

    parser.add_argument("--ed_datasets", default="aida_train.txt_z_aida_dev.txt_z_aida_test.txt_z_"
                                                 "ace2004.txt_z_aquaint.txt_z_clueweb.txt_z_msnbc.txt_z_wikipedia.txt")
    parser.add_argument("--el_datasets", default="aida_train.txt_z_aida_dev.txt_z_aida_test.txt_z_"
                                                 "ace2004.txt_z_aquaint.txt_z_clueweb.txt_z_msnbc.txt_z_wikipedia.txt")

    parser.add_argument("--ed_val_datasets", default="1")
    parser.add_argument("--el_val_datasets", default="1")

    parser.add_argument("--p_e_m_algorithm", type=bool, default=False,
                        help="Baseline. Doesn't use the NN but only the p_e_m dictionary for"
                             "its predictions.")
    parser.add_argument("--all_spans_training", type=bool, default=False)
    parser.add_argument("--entity_extension", default=None, help="extension_entities or extension_entities_all etc")

    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--gm_bucketing_pempos", default=None, help="0_1_2_7  will create bins 0, 1, 2, [3,7], [8,inf)")
    parser.add_argument("--hardcoded_thr", type=float, default=None)
    args = parser.parse_args()

    temp = "all_spans_" if args.all_spans_training else ""
    args.output_folder = config.base_folder+"data/tfrecords/" + \
                         args.experiment_name+"/{}training_folder/".format(temp)+ \
                         args.training_name+"/"
    args.checkpoints_folder = args.output_folder + "checkpoints/"
    args.predictions_folder = args.output_folder + "predictions/"

    if args.p_e_m_algorithm:
        args.predictions_folder = args.output_folder + "p_e_m_predictions/"

    if args.print_predictions and not os.path.exists(args.predictions_folder):
        os.makedirs(args.predictions_folder)
    if args.print_predictions and not os.path.exists(args.predictions_folder+"ed/"):
        os.makedirs(args.predictions_folder+"ed/")
    if args.print_predictions and not os.path.exists(args.predictions_folder+"el/"):
        os.makedirs(args.predictions_folder+"el/")

    train_args = load_train_args(args.output_folder, "evaluate")

    args.ed_datasets = args.ed_datasets.split('_z_') if args.ed_datasets != "" else None
    args.el_datasets = args.el_datasets.split('_z_') if args.el_datasets != "" else None
    args.ed_val_datasets = [int(x) for x in args.ed_val_datasets.split('_')]
    args.el_val_datasets = [int(x) for x in args.el_val_datasets.split('_')]
    args.gm_bucketing_pempos = [int(x) for x in args.gm_bucketing_pempos.split('_')] if args.gm_bucketing_pempos else []

    print(args)
    return args, train_args


if __name__ == "__main__":
    args, train_args = _parse_args()
    print("train_args:\n", train_args)
    # TODO do this argument transfering in the load_train_args instead of train.py, evaluate.py, ensemble_evaluate.py
    train_args.checkpoint_model_num = args.checkpoint_model_num
    train_args.entity_extension = args.entity_extension
    train.args = args
    args.batch_size = train_args.batch_size
    printPredictions = None
    if args.print_predictions:
        from evaluation.print_predictions import PrintPredictions
        printPredictions = PrintPredictions(config.base_folder+"data/tfrecords/"+
                         args.experiment_name+"/", args.predictions_folder, args.entity_extension,
                                            args.gm_bucketing_pempos, args.print_global_voters,
                                            args.print_global_pairwise_scores)
    evaluate()

