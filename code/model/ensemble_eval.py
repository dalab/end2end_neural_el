import argparse
import pickle
import model.config as config
import os
import tensorflow as tf
from model.model_ablations import Model
import model.train as train
from evaluation.metrics import Evaluator, metrics_calculation_and_prediction_printing, threshold_calculation
import model.reader as reader
from model.util import load_train_args

def validation_loss_calculation(filename, opt_thr, el_mode):
    if args.predictions_folder is not None:
        printPredictions.process_file(el_mode, filename, opt_thr)

    ensemble_fixed = []
    ensemble_acc = []  # final_scores and similarity_scores. all the rest are fixed
    for model_num, model_folder in enumerate(args.output_folder):  # for all ensemble models
        model, handles = create_input_pipeline(el_mode, model_folder,
                                               [filename])
        retrieve_l = [model.final_scores, model.similarity_scores,
                      model.cand_entities_len, model.cand_entities,
                      model.begin_span, model.end_span, model.spans_len,
                      model.begin_gm, model.end_gm,
                      model.ground_truth, model.ground_truth_len,
                      model.words_len, model.chunk_id,
                      # model.similarity_scores,
                      model.words, model.chars, model.chars_len,
                      model.log_cand_entities_scores]
        elem_idx = 0
        while True:
            try:
                result_l = model.sess.run(
                    retrieve_l, feed_dict={model.input_handle_ph: handles[0], model.dropout: 1})
                if model_num == 0:
                    ensemble_fixed.append(result_l[2:])
                    ensemble_acc.append(result_l[:2])
                else:
                    ensemble_acc[elem_idx][0] += result_l[0]
                    ensemble_acc[elem_idx][1] += result_l[1]

                elem_idx += 1
            except tf.errors.OutOfRangeError:
                break
        model.close_session()

    evaluator = Evaluator(opt_thr, name=filename)
    number_of_models = len(args.output_folder)
    for (final_scores, similarity_scores), fixed in zip(ensemble_acc, ensemble_fixed):
        final_scores /= number_of_models
        similarity_scores /= number_of_models

        metrics_calculation_and_prediction_printing(evaluator,
                        final_scores, similarity_scores, *fixed, el_mode,
                        printPredictions=printPredictions)

    if printPredictions:
        printPredictions.file_ended()
    print(filename)
    micro_f1, macro_f1 = evaluator.print_log_results(None, -1, el_mode)
    return macro_f1


def optimal_thr_calc(el_mode):
    filenames = args.el_datasets if el_mode else args.ed_datasets
    val_datasets = args.el_val_datasets if el_mode else args.ed_val_datasets

    ensemble_fixed = []
    ensemble_acc = []  # final_scores and similarity_scores. all the rest are fixed
    for model_num, model_folder in enumerate(args.output_folder):  # for all ensemble models
        model, handles = create_input_pipeline(el_mode, model_folder,
                                        [filenames[i] for i in val_datasets])

        retrieve_l = (model.final_scores, model.cand_entities_len, model.cand_entities,
                      model.begin_span, model.end_span, model.spans_len,
                      model.begin_gm, model.end_gm,
                      model.ground_truth, model.ground_truth_len,
                      model.words_len, model.chunk_id)
        elem_idx = 0
        for dataset_handle in handles:  # 1, 4  for each validation dataset
            while True:
                try:
                    result_l = model.sess.run(
                        retrieve_l, feed_dict={model.input_handle_ph: dataset_handle, model.dropout: 1})
                    if model_num == 0:
                        ensemble_fixed.append(result_l[1:])
                        ensemble_acc.append(result_l[0])
                    else:
                        ensemble_acc[elem_idx] += result_l[0]

                    elem_idx += 1
                except tf.errors.OutOfRangeError:
                    break
        model.close_session()

    number_of_models = len(args.output_folder)
    tp_fp_scores_labels = []
    fn_scores = []
    for final_scores, fixed in zip(ensemble_acc, ensemble_fixed):
        final_scores /= number_of_models

        tp_fp_batch, fn_batch = threshold_calculation(final_scores, *fixed, el_mode)
        tp_fp_scores_labels.extend(tp_fp_batch)
        fn_scores.extend(fn_batch)

    return train.optimal_thr_calc_aux(tp_fp_scores_labels, fn_scores)


def create_input_pipeline(el_mode, model_folder, filenames):
    tf.reset_default_graph()
    folder = config.base_folder+"data/tfrecords/" + args.experiment_name + ("/test/" if el_mode else "/train/")
    datasets = []
    for file in filenames:
        datasets.append(reader.test_input_pipeline([folder+file], args))

    input_handle_ph = tf.placeholder(tf.string, shape=[], name="input_handle_ph")
    iterator = tf.contrib.data.Iterator.from_string_handle(
        input_handle_ph, datasets[0].output_types, datasets[0].output_shapes)
    next_element = iterator.get_next()

    train_args = load_train_args(args.output_folder, "ensemble_eval")

    print("loading Model:", model_folder)
    #train_args.evaluation_script = True
    train_args.entity_extension = args.entity_extension
    model = Model(train_args, next_element)
    model.build()
    #print("model train_args:", model.args)
    #print("model checkpoint_folder:", model.args.checkpoints_folder)
    model.input_handle_ph = input_handle_ph
    model.restore_session("el" if el_mode else "ed")

    #iterators, handles = from_datasets_to_iterators_and_handles(model.sess, datasets)
    iterators = []
    handles = []
    for dataset in datasets:
        #iterator = dataset.make_initializable_iterator() # one shot iterators fits better here
        iterator = dataset.make_one_shot_iterator()
        iterators.append(iterator)
        handles.append(model.sess.run(iterator.string_handle()))
    return model, handles


def evaluate():
    for el_mode in [False, True]:
        filenames = args.el_datasets if el_mode else args.ed_datasets
        if filenames:
            print("Evaluating {} datasets".format("EL" if el_mode else "ED"))
            opt_thr, _ = optimal_thr_calc(el_mode)
            # TODO check the following lines
            results = []
            #for test_handle, test_name, test_it in zip(datasets, names):
            for filename in filenames:
                f1_score = validation_loss_calculation(filename, opt_thr, el_mode=el_mode)
            results.append(f1_score)

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default="alldatasets_perparagr", #"standard",
                        help="under folder data/tfrecords/")
    parser.add_argument("--training_name", help="under folder data/tfrecords/")
    parser.add_argument("--predictions_folder", default=None, help="full or relative path. where to print"
                                                                   "the result files.")

    parser.add_argument("--ed_datasets", default="aida_train.txt_z_aida_dev.txt_z_aida_test.txt_z_"
                                                 "ace2004.txt_z_aquaint.txt_z_clueweb.txt_z_msnbc.txt_z_wikipedia.txt")
    parser.add_argument("--el_datasets", default="aida_train.txt_z_aida_dev.txt_z_aida_test.txt_z_"
                                                 "ace2004.txt_z_aquaint.txt_z_clueweb.txt_z_msnbc.txt_z_wikipedia.txt")

    parser.add_argument("--ed_val_datasets", default="1")
    parser.add_argument("--el_val_datasets", default="1")

    parser.add_argument("--all_spans_training", help="y_y_n_y_n")
    parser.add_argument("--entity_extension", default=None, help="extension_entities or extension_entities_all etc")
    args = parser.parse_args()

    args.ed_datasets = args.ed_datasets.split('_z_') if args.ed_datasets != "" else None
    args.el_datasets = args.el_datasets.split('_z_') if args.el_datasets != "" else None
    args.ed_val_datasets = [int(x) for x in args.ed_val_datasets.split('_')]
    args.el_val_datasets = [int(x) for x in args.el_val_datasets.split('_')]

    args.training_name = args.training_name.split('_z_')
    args.all_spans_training = ["all_spans_" if x == 'y' else "" for x in args.all_spans_training.split('_')]
    assert(len(args.training_name) == len(args.all_spans_training))

    if args.predictions_folder is not None and not os.path.exists(args.predictions_folder):
        os.makedirs(args.predictions_folder)
    if args.predictions_folder is not None and not os.path.exists(args.predictions_folder+"ed/"):
        os.makedirs(args.predictions_folder+"ed/")
    if args.predictions_folder is not None and not os.path.exists(args.predictions_folder+"el/"):
        os.makedirs(args.predictions_folder+"el/")

    args.output_folder = []
    for training_name, prefix in zip(args.training_name, args.all_spans_training):
        args.output_folder.append(config.base_folder+"data/tfrecords/" + \
                             args.experiment_name+"/{}training_folder/".format(prefix) + \
                             training_name+"/")

    args.batch_size = 1
    print(args)
    return args


if __name__ == "__main__":
    args = _parse_args()
    printPredictions = None
    if args.predictions_folder is not None:
        from evaluation.print_predictions import PrintPredictions
        printPredictions = PrintPredictions(config.base_folder+"data/tfrecords/"+
                         args.experiment_name+"/", args.predictions_folder)
    evaluate()

