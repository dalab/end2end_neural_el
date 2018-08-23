import argparse
import os
import preprocessing.util as util
from subprocess import call

def process_aida(in_filepath, out_filepath):

    # _, wiki_id_name_map = util.load_wiki_name_id_map(lowercase=False)
    #_, wiki_id_name_map = util.entity_name_id_map_from_dump()
    entityNameIdMap = util.EntityNameIdMap()
    entityNameIdMap.init_compatible_ent_id()
    unknown_gt_ids = 0   # counter of ground truth entity ids that are not in the wiki_name_id.txt
    ent_id_changes = 0
    text_acc = []
    with open(in_filepath) as fin, open(args.output_folder+"tokenize_"+out_filepath, "w") as fout:
        in_mention = False   # am i inside a mention span or not
        first_document = True
        for line in fin:
            l = line.split('\t')
            if in_mention and not (len(l) == 7 and l[1]=='I'):
                # if I am in mention but the current line does not continue the previous mention
                # then print MMEND and be in state in_mention=FALSE
                #fout.write("MMEND\n")
                text_acc.append("MMEND")
                in_mention = False

            if line.startswith("-DOCSTART-"):
                if not first_document:
                    #fout.write("DOCEND\n")
                    text_acc.append("DOCEND")
                # line = "-DOCSTART- (967testa ATHLETICS)\n"
                doc_title = line[len("-DOCSTART- ("): -2]
                #fout.write("DOCSTART_"+doc_title.replace(' ', '_')+"\n")
                text_acc.append("DOCSTART_"+doc_title.replace(' ', '_'))
                first_document = False
            elif line == "\n":
                #fout.write("*NL*\n")
                text_acc.append("\n")
            elif len(l) == 7 and l[1] == 'B':  # this is a new mention
                wiki_title = l[4]
                wiki_title = wiki_title[len("http://en.wikipedia.org/wiki/"):].replace('_', ' ')
                new_ent_id = entityNameIdMap.compatible_ent_id(wiki_title, l[5])
                if new_ent_id is not None:
                    if new_ent_id != l[5]:
                        ent_id_changes += 1
                        #print(line, "old ent_id: " + l[5], " new_ent_id: ", new_ent_id)
                    #fout.write("MMSTART_"+new_ent_id+"\n")   # TODO check here if entity id is inside my wikidump
                    text_acc.append("MMSTART_"+new_ent_id)
                                                   # if not then omit this mention
                    #fout.write(l[0]+"\n")  # write the word
                    text_acc.append(l[0])  # write the word
                    in_mention = True
                else:
                    unknown_gt_ids += 1
                    #fout.write(l[0]+"\n")  # write the word
                    text_acc.append(l[0])  # write the word
                    print(line)
            else:
                # words that continue a mention len(l) == 7: and l[1]=='I'
                # or normal word outside of mention, or in mention without disambiguation (len(l) == 4)
                #fout.write(l[0].rstrip()+"\n")
                text_acc.append(l[0].rstrip())
        #fout.write("DOCEND\n")  # for the last document
        text_acc.append("DOCEND")  # for the last document
        fout.write(' '.join(text_acc))
    print("process_aida     unknown_gt_ids: ", unknown_gt_ids)
    print("process_aida     ent_id_changes: ", ent_id_changes)
    print("now tokenize with stanford tokenizer")
    tokenize_command = 'cd {}; java -cp "*" ' \
                       'edu.stanford.nlp.process.PTBTokenizer -options "tokenizeNLs=True" < {} > {}'.format(
        args.stanford_tokenizer_folder, args.output_folder+"tokenize_"+out_filepath, args.output_folder+out_filepath)
    print(tokenize_command)
    call(tokenize_command, shell=True)


def split_dev_test(in_filepath):
    with open(in_filepath) as fin, open(args.output_folder+"temp_aida_dev", "w") as fdev,\
            open(args.output_folder+"temp_aida_test", "w") as ftest:
        fout = fdev
        for line in fin:
            if line.startswith("-DOCSTART-") and line.find("testb") != -1:
                fout = ftest
            fout.write(line)


def create_necessary_folders():
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aida_folder", default="/home/master_thesis_share/data/basic_data/test_datasets/AIDA/")
    parser.add_argument("--output_folder", default="/home/master_thesis_share/data/new_datasets/")
    parser.add_argument("--stanford_tokenizer_folder",
                        default="/home/programs/stanford_core_nlp/stanford-corenlp-full-2017-06-09")
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    create_necessary_folders()
    process_aida(args.aida_folder+"aida_train.txt", "aida_train.txt")

    split_dev_test(args.aida_folder+"testa_testb_aggregate_original")
    process_aida(args.output_folder+"temp_aida_dev", "aida_dev.txt")
    process_aida(args.output_folder+"temp_aida_test", "aida_test.txt")

    os.remove(args.output_folder + "temp_aida_dev")
    os.remove(args.output_folder + "temp_aida_test")
    os.remove(args.output_folder + "tokenize_aida_train.txt")
    os.remove(args.output_folder + "tokenize_aida_dev.txt")
    os.remove(args.output_folder + "tokenize_aida_test.txt")
