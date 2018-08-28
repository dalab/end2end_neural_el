import tensorflow as tf
import argparse
import model.config as config
import pickle

def parse_sequence_example(serialized):
    sequence_features={
            "words": tf.FixedLenSequenceFeature([], dtype=tf.int64),   # in order to have a vector. if i put [1] it will probably
            # be a matrix with just one column
            "chars": tf.VarLenFeature(tf.int64),
            "chars_len": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "begin_span": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "end_span": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "cand_entities": tf.VarLenFeature(tf.int64),
            "cand_entities_scores": tf.VarLenFeature(tf.float32),
            "cand_entities_labels": tf.VarLenFeature(tf.int64),
            "cand_entities_len": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "ground_truth": tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }
    if True:
        sequence_features["begin_gm"] = tf.FixedLenSequenceFeature([], dtype=tf.int64)
        sequence_features["end_gm"] = tf.FixedLenSequenceFeature([], dtype=tf.int64)

    context, sequence = tf.parse_single_sequence_example(
        serialized,
        context_features={
            "chunk_id": tf.FixedLenFeature([], dtype=tf.string),
            "words_len": tf.FixedLenFeature([], dtype=tf.int64),
            "spans_len": tf.FixedLenFeature([], dtype=tf.int64),
            "ground_truth_len": tf.FixedLenFeature([], dtype=tf.int64)
        },
        sequence_features=sequence_features)

    return context["chunk_id"], sequence["words"], context["words_len"],\
           tf.sparse_tensor_to_dense(sequence["chars"]), sequence["chars_len"],\
           sequence["begin_span"], sequence["end_span"], context["spans_len"],\
           tf.sparse_tensor_to_dense(sequence["cand_entities"]),\
           tf.sparse_tensor_to_dense(sequence["cand_entities_scores"]),\
           tf.sparse_tensor_to_dense(sequence["cand_entities_labels"]),\
           sequence["cand_entities_len"],\
           sequence["ground_truth"], context["ground_truth_len"],\
           sequence["begin_gm"], sequence["end_gm"]
    #return context, sequence


def count_records_of_one_epoch(trainfiles):
    filename_queue = tf.train.string_input_producer(trainfiles, num_epochs=1)
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)
    with tf.Session() as sess:
        sess.run(
            tf.variables_initializer(
                tf.global_variables() + tf.local_variables()
            )
        )

        # Start queue runners
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        counter = 0
        try:
            while not coord.should_stop():
                fetch_vals = sess.run((key))
                #print(fetch_vals)
                counter += 1
        except tf.errors.OutOfRangeError:
            pass
        except KeyboardInterrupt:
            print("Training stopped by Ctrl+C.")
        finally:
            coord.request_stop()
        coord.join(threads)
    print("number of tfrecords in trainfiles = ", counter)
    return counter


def train_input_pipeline(filenames, args):
    dataset = tf.data.TFRecordDataset(filenames)
    #dataset = tf.contrib.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_sequence_example)
    #dataset = dataset.map(parse_sequence_example, num_parallel_calls=3)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=args.shuffle_capacity)
    dataset = dataset.padded_batch(args.batch_size, dataset.output_shapes)
    return dataset


def test_input_pipeline(filenames, args):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_sequence_example)
    dataset = dataset.padded_batch(args.batch_size, dataset.output_shapes)
    return dataset

if __name__ == "__main__":
    count_records_of_one_epoch(["/home/master_thesis_share/data/tfrecords/"
                               "wikiRLTD_perparagr_wthr_6_cthr_201/"
                               "train/wikidumpRLTD.txt"])


