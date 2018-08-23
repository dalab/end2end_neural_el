import os
import tensorflow as tf


class BaseModel(object):

    def __init__(self, args):
        self.args = args
        self.sess = None
        self.ed_saver = None
        self.el_saver = None

    def reinitialize_weights(self, scope_name):
        """Reinitializes the weights of a given layer"""
        variables = tf.contrib.framework.get_variables(scope_name)
        init = tf.variables_initializer(variables)
        self.sess.run(init)

    def add_train_op(self, lr_method, lr, loss, clip=-1):
        """Defines self.train_op that performs an update on a batch
        Args:
            lr_method: (string) sgd method, for example "adam"
            lr: (tf.placeholder) tf.float32, learning rate
            loss: (tensor) tf.float32 loss to minimize
            clip: (python float) clipping of gradient. If < 0, no clipping
        """
        _lr_m = lr_method.lower() # lower to make sure

        with tf.variable_scope("train_step"):
            if _lr_m == 'adam': # sgd method
                optimizer = tf.train.AdamOptimizer(lr)
            elif _lr_m == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(lr)
            elif _lr_m == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr)
            elif _lr_m == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(lr)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))

            if clip > 0: # gradient clipping if clip is positive
                grads, vs     = zip(*optimizer.compute_gradients(loss))
                grads, gnorm  = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(loss)

    def initialize_session(self):
        """Defines self.sess and initialize the variables"""
        print("Initializing tf session")
        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        #self.sess = tf.Session(config=config)
        self.sess = tf.Session()
        #from tensorflow.python import debug as tf_debug
        #self.sess = tf_debug.LocalCLIDebugWrapperSession(tf.Session())
        self.sess.run(tf.global_variables_initializer())
        self.ed_saver = tf.train.Saver(var_list=self.checkpoint_variables(), max_to_keep=self.args.checkpoints_num)
        self.el_saver = tf.train.Saver(var_list=self.checkpoint_variables(), max_to_keep=self.args.checkpoints_num)

    def restore_session(self, option="latest"):
        """option: 'latest', 'ed', 'el'  so it chooses the latest checkpoint for ed for el or from
        both of them if it is 'latest' (this is used in continue_training'"""
        """restores from the latest checkpoint of this folder"""
        assert(option in ["latest", "ed", "el"])
        if hasattr(self.args, 'checkpoint_model_num') and self.args.checkpoint_model_num is not None:
            assert(option != "latest")  # it is either ed or el
            checkpoint_path = self.args.checkpoints_folder + option + "/model-{}".format(self.args.checkpoint_model_num)
        else:
            if option == "ed":
                checkpoint_path = self.my_latest_checkpoint(self.args.checkpoints_folder+"ed/")
            elif option == "el":
                checkpoint_path = self.my_latest_checkpoint(self.args.checkpoints_folder+"el/")
            elif option == "latest":
                print("Reloading the latest trained model...(either ed or el)")
                ed = self.my_latest_checkpoint(self.args.checkpoints_folder+"ed/")
                el = self.my_latest_checkpoint(self.args.checkpoints_folder+"el/")
                ed_eval_cnt = int(ed[ed.rfind('-') + 1:])
                el_eval_cnt = int(el[el.rfind('-') + 1:])
                if ed_eval_cnt >= el_eval_cnt:
                    checkpoint_path = self.my_latest_checkpoint(self.args.checkpoints_folder+"ed/")
                    option = "ed"
                else:
                    checkpoint_path = self.my_latest_checkpoint(self.args.checkpoints_folder+"el/")
                    option = "el"
        print("Using checkpoint: {}".format(checkpoint_path))
        self.sess = tf.Session()
        self.ed_saver = tf.train.Saver(var_list=self.checkpoint_variables(), max_to_keep=self.args.checkpoints_num)
        self.el_saver = tf.train.Saver(var_list=self.checkpoint_variables(), max_to_keep=self.args.checkpoints_num)

        saver = self.ed_saver if option == "ed" else self.el_saver
        saver.restore(self.sess, checkpoint_path)
        self.init_embeddings()
        print("Finished loading checkpoint.")
        return checkpoint_path

    def my_latest_checkpoint(self, folder_path):   # model-9.meta
        files = [name for name in os.listdir(folder_path) if name.startswith("model") and name.endswith("meta")]
        max_epoch = max([int(name[len("model-"):-len(".meta")]) for name in files])
        return folder_path + "model-" + str(max_epoch)

    def save_session(self, eval_cnt, save_ed_flag, save_el_flag):
        """Saves session = weights"""
        for save_flag, category in zip([save_ed_flag, save_el_flag], ["ed", "el"]):
            if save_flag is False:
                continue
            checkpoints_folder = self.args.checkpoints_folder + category + "/"
            if not os.path.exists(checkpoints_folder):
                os.makedirs(checkpoints_folder)
            print("saving session checkpoint for {}...".format(category))
            checkpoint_prefix = os.path.join(checkpoints_folder, "model")
            saver = self.ed_saver if category == "ed" else self.el_saver
            save_path = saver.save(self.sess, checkpoint_prefix, global_step=eval_cnt)
            print("Checkpoint saved in file: %s" % save_path)

    def close_session(self):
        """Closes the session"""
        self.sess.close()

    def _restore_list(self):
        return [n for n in tf.global_variables()
                 if n.name != 'entities/_entity_embeddings:0']

    def checkpoint_variables(self):
        """omit word embeddings and entity embeddings from being stored in checkpoint when they are fixed
        in order to save disk space. word emb are always fixed, entity emb are fixed when
        args.train_ent_vecs == False"""
        omit_variables = ['words/_word_embeddings:0']
        if not self.args.train_ent_vecs:
            omit_variables.append('entities/_entity_embeddings:0')
        variables = [n for n in tf.global_variables() if n.name not in omit_variables]
        print("checkpoint variables to restore:", variables)
        return variables

    def find_variable_handler_by_name(self, var_name):
        for n in tf.global_variables():
            if n.name == var_name:
                return n