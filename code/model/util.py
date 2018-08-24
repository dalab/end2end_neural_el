import tensorflow as tf
import numpy as np
import model.config as config

def projection(inputs, output_size, initializer=None, model=None):
    return ffnn(inputs, 0, -1, output_size, dropout=None, output_weights_initializer=initializer, model=model)

def shape(x, dim):
    return x.get_shape()[dim].value or tf.shape(x)[dim]

def ffnn(inputs, num_hidden_layers, hidden_size, output_size, dropout, output_weights_initializer=None, model=None):
    l2maxnorm = model.args.ffnn_l2maxnorm if model else None
    if l2maxnorm:
        dropout = None  # either dropout or l2normalization not both

    if len(inputs.get_shape()) > 2:
        # from [batch, max_sentence_length, emb i.e.500]    to   [batch*max_sentence_length, emb]
        current_inputs = tf.reshape(inputs, [-1, shape(inputs, -1)])
    else:
        current_inputs = inputs

    for i in range(num_hidden_layers):
        hidden_weights = tf.get_variable("hidden_weights_{}".format(i), [shape(current_inputs, 1), hidden_size])
        hidden_bias = tf.get_variable("hidden_bias_{}".format(i), [hidden_size])
        variable_summaries(hidden_weights)
        variable_summaries(hidden_bias)
        if l2maxnorm:
            temp = hidden_weights / tf.maximum(tf.norm(hidden_weights)/l2maxnorm, 1)
            model.ffnn_l2normalization_op_list.append(hidden_weights.assign(temp))
            temp = hidden_bias / tf.maximum(tf.norm(hidden_bias)/l2maxnorm, 1)
            model.ffnn_l2normalization_op_list.append(hidden_bias.assign(temp))

        current_outputs = tf.nn.relu(tf.matmul(current_inputs, hidden_weights) + hidden_bias)

        if dropout is not None:
            current_outputs = tf.nn.dropout(current_outputs, dropout)
        current_inputs = current_outputs

                              #    500                  1
    output_weights = tf.get_variable("output_weights", [shape(current_inputs, 1), output_size], initializer=output_weights_initializer)
    output_bias = tf.get_variable("output_bias", [output_size])
    variable_summaries(output_weights)
    variable_summaries(output_bias)
    if l2maxnorm and not model.args.ffnn_l2maxnorm_onlyhiddenlayers:
        temp = output_weights / tf.maximum(tf.norm(output_weights)/l2maxnorm, 1)
        model.ffnn_l2normalization_op_list.append(output_weights.assign(temp))
        temp = output_bias / tf.maximum(tf.norm(output_bias)/l2maxnorm, 1)
        model.ffnn_l2normalization_op_list.append(output_bias.assign(temp))

    outputs = tf.matmul(current_inputs, output_weights) + output_bias
    #print("model/util variable name = ", output_weights.name, output_bias.name)

    if len(inputs.get_shape()) == 3:
        outputs = tf.reshape(outputs, [shape(inputs, 0), shape(inputs, 1), output_size])
    elif len(inputs.get_shape()) == 4:
        outputs = tf.reshape(outputs, [shape(inputs, 0), shape(inputs, 1), shape(inputs, 2), output_size])
    elif len(inputs.get_shape()) > 4:
        raise ValueError("FFNN with rank {} not supported".format(len(inputs.get_shape())))
    return outputs



def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    name = "_" + var.name.split("/")[-1].split(":")[0]
    #print("name of summary", name)
    with tf.name_scope('summaries'+name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))


import sys
# https://stackoverflow.com/questions/616645/how-do-i-duplicate-sys-stdout-to-a-log-file-in-python
# http://web.archive.org/web/20141016185743/https://mail.python.org/pipermail/python-list/2007-May/460639.html
class Tee(object):

    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def close(self):
        if self.stdout is not None:
            sys.stdout = self.stdout
            self.stdout = None
        if self.file is not None:
            self.file.close()
            self.file = None

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def __del__(self):
        self.close()


def _correct_train_args_leohnard_dalab(train_args, model_folder):
    """if the experiment is executed on leohnard then the path starts with
    /cluster/ whereas on dalabgpu startswith /local/ also folders may have
    been moved to a different position so I should correct the paths."""
    if train_args.output_folder != model_folder:
        train_args.output_folder = model_folder
        train_args.checkpoints_folder = model_folder + "checkpoints/"
        train_args.summaries_folder = model_folder + "summaries/"
        train_args.inconsistent_model_folder = True

import pickle

def load_train_args(output_folder, running_mode):
    """running_mode: train, train_continue, evaluate, ensemble_eval, gerbil"""
    with open(output_folder+"train_args.pickle", 'rb') as handle:
        train_args = pickle.load(handle)
    _correct_train_args_leohnard_dalab(train_args, output_folder)
    # update checkpoint train_args to be compatible with new arguments added in code
    train_args.running_mode = running_mode
    if not hasattr(train_args, 'nn_components') and hasattr(train_args, 'pem_lstm_attention'):
        train_args.nn_components = train_args.pem_lstm_attention
    if not hasattr(train_args, 'nn_components'):
        train_args.nn_components = "pem_lstm"
    if not hasattr(train_args, 'model_heads_from_bilstm'):
        train_args.model_heads_from_bilstm = False
    if not hasattr(train_args, 'zero'):
        train_args.zero = 1e-3  # for compatibility with old experiments
    if not hasattr(train_args, 'attention_use_AB'):
        train_args.attention_use_AB = False
    if not hasattr(train_args, 'attention_on_lstm'):
        train_args.attention_on_lstm = False
    if not hasattr(train_args, 'span_emb'):
        train_args.span_emb = "boundaries"
        if train_args.model_heads:
            train_args.span_emb += "_head"
    if not hasattr(train_args, 'span_boundaries_from_wordemb'):
        train_args.span_boundaries_from_wordemb = False
    if not hasattr(train_args, 'attention_ent_vecs_no_regularization'):
        train_args.attention_ent_vecs_no_regularization = False
    if not hasattr(train_args, 'global_topkthr'):
        train_args.global_topkthr = None
    if not hasattr(train_args, 'global_topkfromallspans'):
        train_args.global_topkfromallspans = None
    if not hasattr(train_args, 'hardcoded_thr'):
        train_args.hardcoded_thr = None
    if not hasattr(train_args, 'global_one_loss'):
        train_args.global_one_loss = False
    if not hasattr(train_args, 'global_norm_or_mean'):
        train_args.global_norm_or_mean = "norm"
    if not hasattr(train_args, 'ffnn_dropout'):
        train_args.ffnn_dropout = True
    if not hasattr(train_args, 'cand_ent_num_restriction'):
        train_args.cand_ent_num_restriction = None
    if not hasattr(train_args, 'ffnn_l2maxnorm'):
        train_args.ffnn_l2maxnorm = None
    if not hasattr(train_args, 'ffnn_l2maxnorm_onlyhiddenlayers'):
        train_args.ffnn_l2maxnorm_onlyhiddenlayers = False
    if not hasattr(train_args, 'pem_without_log'):
        train_args.pem_without_log = False
    if not hasattr(train_args, 'global_mask_scale_each_mention_voters_to_one'):
        train_args.global_mask_scale_each_mention_voters_to_one = False
    if not hasattr(train_args, 'pem_buckets_boundaries'):
        train_args.pem_buckets_boundaries = None
    if not hasattr(train_args, 'global_gmask_based_on_localscore'):
        train_args.global_gmask_based_on_localscore = False
    if not hasattr(train_args, 'attention_retricted_num_of_entities'):
        train_args.attention_retricted_num_of_entities = None

    if not hasattr(train_args, 'stage2_nn_components'):
        train_args.stage2_nn_components = "local_global"
    if not hasattr(train_args, 'global_gmask_unambigious'):
        train_args.global_gmask_unambigious = False
    return train_args


def load_ent_vecs(args):
    entity_embeddings_nparray = np.load(config.base_folder + "data/entities/ent_vecs/ent_vecs.npy")
    entity_embeddings_nparray[0] = 0
    if hasattr(args, 'entity_extension') and args.entity_extension is not None:
        entity_extension = np.load(config.base_folder +"data/entities/"+args.entity_extension+
                                           "/ent_vecs/ent_vecs.npy")
        entity_embeddings_nparray = np.vstack((entity_embeddings_nparray, entity_extension))
    return entity_embeddings_nparray




if __name__ == "__main__":
    path = "/home/master_thesis_share/data/tfrecords/per_document_no_wikidump/all_spans_training_folder/group_global/c50h50_lstm150_nohead_attR10K100_fffnn0_0_glthrm005_glffnn0_0v1/train_args.pickle"
    with open(path, 'rb') as handle:
        train_args = pickle.load(handle)
    train_args.global_topk = None
    with open(path, 'wb') as handle:
        pickle.dump(train_args, handle)





