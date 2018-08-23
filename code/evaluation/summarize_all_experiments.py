import argparse
import os


def process_experiment(ed_acc, el_acc, training_name):
    if not os.path.exists(os.path.join(training_name, "log.txt")):
        print("File doesn't exists: ", os.path.join(training_name, "log.txt"))
        return
    if file_is_used(os.path.join(training_name, "log.txt")):
        print("File is being used by another process. Skip it.", os.path.join(training_name, "log.txt"))
        return
    with open(os.path.join(training_name, "log.txt"), "r") as fin:
        print("file: ", training_name+"/log.txt")
        best = dict()
        best["ed_dev_f1"] = 0
        best["el_dev_f1"] = 0
        best["ed_test_f1"] = 0
        best["el_test_f1"] = 0

        mode = ""
        for line in fin:
            line = line.rstrip()
            if line.startswith("args.eval_cnt"):
                eval_cnt = line[line.rfind(' ')+1:]
            elif line.startswith("Evaluating ED datasets"):
                mode = "ed"
            elif line.startswith("Evaluating EL datasets"):
                mode = "el"
            elif line.startswith(args.dev_set): #("aida_dev.txt"):
                try:
                    micro_line = next(fin)
                    macro_line = next(fin)
                    line = macro_line if args.macro_or_micro == "macro" else micro_line
                    dev_f1 = float(line.split()[-1])
                    dev_pr = float(line.split()[2])
                    dev_re = float(line.split()[4])
                    if dev_f1 > best[mode+"_dev_f1"]:
                        best[mode+"_dev_f1"] = dev_f1
                        best[mode+"_dev_pr"] = dev_pr
                        best[mode+"_dev_re"] = dev_re
                        best[mode+"_eval_cnt"] = eval_cnt

                        # now read forward the test results
                        #assert(next(fin).startswith(args.test_set))  #("aida_test.txt"))
                        next_line = next(fin)
                        while not next_line.startswith(args.test_set):
                            next_line = next(fin)
                        assert(next_line.startswith(args.test_set))  #("aida_test.txt"))
                        micro_line = next(fin)
                        macro_line = next(fin)
                        line = macro_line if args.macro_or_micro == "macro" else micro_line
                        best[mode+"_test_f1"] = float(line.split()[-1])
                        best[mode+"_test_pr"] = float(line.split()[2])
                        best[mode+"_test_re"] = float(line.split()[4])

                except StopIteration:
                    break

        path = training_name[training_name.find(base_folder)+len(base_folder):]
        # print the scores for this log file
        #fixed_no_wikidump_entvecsl2/checkpoints/model-7    model-30.meta
        if "ed_eval_cnt" in best:
            checkpoint_text = "checkpoint_yes" if os.path.exists(training_name + "/checkpoints/ed/model-{}.meta".format(best["ed_eval_cnt"])) else "checkpoint_no"
            ed_acc.append((best["ed_dev_f1"],  best["ed_test_f1"], path,
                           best["ed_test_pr"], best["ed_test_re"], best["ed_eval_cnt"], checkpoint_text, training_name))
        if "el_eval_cnt" in best:
            checkpoint_text = "checkpoint_yes" if os.path.exists(training_name + "/checkpoints/el/model-{}.meta".format(best["el_eval_cnt"])) else "checkpoint_no"
            el_acc.append((best["el_dev_f1"], best["el_test_f1"], path,
                           best["el_test_pr"], best["el_test_re"], best["el_eval_cnt"], checkpoint_text, training_name))


def process_folder(ed_acc, el_acc, training_name):
    """training_name may be a folder with one experiment or a group folder containing many experiment. In
    the second case do recursion on all the subfolders."""
    training_name_suffix = os.path.basename(os.path.normpath(training_name))
    if training_name_suffix.startswith("group_") or training_name_suffix.startswith("reduced") or\
        training_name_suffix.startswith("ensemble_"):
        # then it is a group folder so do recursion on all your subfolders
        d = training_name
        subfolders = [os.path.join(d, o) for o in os.listdir(d)
                        if os.path.isdir(os.path.join(d, o))]
        for subfolder in subfolders:
            process_folder(ed_acc, el_acc, subfolder)
    else:
        process_experiment(ed_acc, el_acc, training_name)


def file_is_used(filepath):
    from subprocess import check_output, Popen, PIPE, DEVNULL, STDOUT
    try:
        lsout = Popen(['lsof', filepath], stdout=PIPE, shell=False, stderr=DEVNULL)
        check_output(["grep", filepath], stdin=lsout.stdout, shell=False)
        return True
    except:
        #check_output will throw an exception here if it won't find any process using that file
        return False


def main():
    ed_acc = []
    el_acc = []
    if args.group_folder_path:
        process_folder(ed_acc, el_acc, args.group_folder_path)
    else:
        d = base_folder
        # if the base folder is itself an experiment i.e. contains a training_folder and an all_spans_training_folder
        print(os.listdir(d))
        if len([o for o in os.listdir(d) if o in ["all_spans_training_folder", "training_folder"]]) > 0:
            experiment_names = [d]
        else:
            experiment_names = [os.path.join(d, o) for o in os.listdir(d)
                            if os.path.isdir(os.path.join(d, o))]
        print("experiment_names =", experiment_names)
        for experiment_name in experiment_names:
            training_names = []
            for temp in ["training_folder", "all_spans_training_folder"]:
                d = os.path.join(experiment_name, temp)
                if not os.path.exists(d):
                    continue
                training_names.extend([os.path.join(d, o) for o in os.listdir(d)
                                        if os.path.isdir(os.path.join(d, o))])
            #print(training_names)
            for training_name in training_names:
                process_folder(ed_acc, el_acc, training_name)

    ed_acc = sorted(ed_acc, reverse=True)
    el_acc = sorted(el_acc, reverse=True)
    print("Dev_score, Test_score, path, test_precision, test_recall, eval_cnt, checkp_existence")
    print("ED Best Scores:")
    for t in ed_acc:
        print('\t'.join(map(str, t[:-1])))
    print("\n\n\nEL Best Scores:")
    for t in el_acc:
        print('\t'.join(map(str, t[:-1])))


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_folder", default="../../data/tfrecords/")
    parser.add_argument("--macro_or_micro", default="macro")
    parser.add_argument("--dev_set", default="aida_dev.txt")
    parser.add_argument("--test_set", default="aida_test.txt")
    parser.add_argument("--group_folder_path", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    base_folder = os.path.abspath(args.base_folder)
    print("base_folder =", base_folder)
    print("group_folder =", args.group_folder_path)
    main()
