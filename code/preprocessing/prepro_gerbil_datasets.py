import argparse
import os
import preprocessing.util as util
from nltk.tokenize import word_tokenize


class ProcessDataset(object):
    def __init__(self):
        self.entityNameIdMap = util.EntityNameIdMap()
        self.entityNameIdMap.init_gerbil_compatible_ent_id()
        self.unknown_ent_name = dict()
        self.no_english_uri = dict()
        self.all_gm_cnt = dict()
        self.englishuri_gm_cnt = dict()
        self.valid_gms = dict()

    def process(self, filepath):
        #the name of the dataset. just extract the last part of path
        dataset = os.path.basename(os.path.normpath(filepath))
        unknown_ent_name = 0
        no_english_uri = 0
        all_gm_cnt = 0
        englishuri_gm_cnt = 0  # has an english uri
        valid_gms = 0  # the ones that we use in the end
        with open(filepath, "r") as fin, open(args.output_folder+dataset+".txt", "w") as fout:
            for doc_name_line in fin:
                doc_name_line = "DOCSTART_" + "".join(filter(str.isalnum, doc_name_line.rstrip()[len("DOCSTART_"):]))
                assert(doc_name_line.startswith("DOCSTART_"))
                # accumulates the text of one document until we process all mentions then outputs to fout
                text = next(fin)
                text = text.rstrip()[len("text: "):]  # clear text
                if text == "":
                    # dataset error. try to read next line. assert that next line is not annotations or docstart
                    text = next(fin)
                gm_num = int(next(fin).rsplit(" ", 1)[1])
                all_gm_cnt += gm_num
                annotations = []
                for _ in range(gm_num):
                    # http://en.wikipedia.org/wiki/Fundraising
                    #m_start, length, uri_list = next(fin).split(", ", 2)
                    #m_start = int(m_start[1:])
                    m_start, length, uri_list = next(fin).split("_z_", 2)
                    m_start = int(m_start)
                    m_end = m_start + int(length)
                    #for uri in uri_list.rstrip()[1:-2].split(", "):
                    for uri in uri_list.rstrip()[1:-1].split(", "):
                        if uri.startswith("http://en.wikipedia.org"):
                            break
                    if not uri.startswith("http://en.wikipedia.org"):
                        # skip this annotation since it doesn't have a link from english wikipedia
                        no_english_uri += 1
                        print(uri_list.rstrip())
                    else:
                        annotations.append((m_start, m_end, uri))
                annotations = sorted(annotations)  # sort gm on start point
                englishuri_gm_cnt += len(annotations)

                text_acc = [doc_name_line.rstrip()]
                m_start_pr = 0  # previous mention start position
                m_end_pr = 0    # previous mention end position
                for (m_start, m_end, uri) in annotations:
                    text_acc.append(text[m_end_pr:m_start])
                    ent_id = self.entityNameIdMap.gerbil_compatible_ent_id(uri)
                    if ent_id is not None:
                        text_acc.append("MMSTART_"+ent_id)
                        text_acc.append(text[m_start:m_end])
                        text_acc.append("MMEND")
                        valid_gms += 1
                    else:  # if not then omit this mention
                        unknown_ent_name += 1
                        text_acc.append(text[m_start:m_end])
                    m_start_pr = m_start
                    m_end_pr = m_end
                # add text after the last mention
                text_acc.append(text[m_end_pr:])
                text_acc.append("DOCEND")
                doc_text = ' '.join(text_acc)
                doc_text = word_tokenize(doc_text)
                #doc_text = fix_tokenization(doc_text)
                doc_text = [word if word not in ["``", "''"] else '"' for word in doc_text]
                fout.write('\n'.join(doc_text) + "\n")
        self.unknown_ent_name[dataset] = unknown_ent_name
        self.no_english_uri[dataset] = no_english_uri
        self.all_gm_cnt[dataset] = all_gm_cnt
        self.englishuri_gm_cnt[dataset] = englishuri_gm_cnt
        self.valid_gms[dataset] = valid_gms

    def process_readable(self, filepath):
        dataset = os.path.basename(os.path.normpath(filepath))
        with open(filepath, "r") as fin, open(args.output_folder+dataset+".txt", "w") as fout:
            for doc_name_line in fin:
                doc_name_line = "DOCSTART_" + "".join(filter(str.isalnum, doc_name_line.rstrip()[len("DOCSTART_"):]))
                # accumulates the text of one document until we process all mentions then outputs to fout
                text = next(fin)
                text = text.rstrip()[len("text: "):]  # clear text
                if text == "":
                    # dataset error. try to read next line. assert that next line is not annotations or docstart
                    text = next(fin)
                gm_num = int(next(fin).rsplit(" ", 1)[1])
                annotations = []
                for _ in range(gm_num):
                    # http://en.wikipedia.org/wiki/Fundraising
                    m_start, length, uri_list = next(fin).split("_z_", 2)
                    m_start = int(m_start)
                    m_end = m_start + int(length)
                    english_url_l = []
                    for uri in uri_list.rstrip()[1:-1].split(", "):
                        if uri.startswith("http://en.wikipedia.org"):
                            temp = "http://en.wikipedia.org/wiki/"
                            assert(uri.startswith(temp))
                            english_url_l.append(uri[len(temp):])
                    if not english_url_l:
                        # this annotation doesn't have a link from english wikipedia
                        english_url_l.append("no wiki url")
                    annotations.append((m_start, m_end, english_url_l))
                annotations = sorted(annotations)  # sort gm on start point

                text_acc = [doc_name_line.rstrip()]
                m_start_pr = 0  # previous mention start position
                m_end_pr = 0    # previous mention end position
                for cnt, (m_start, m_end, uri_l) in enumerate(annotations, 1):
                    text_acc.append(text[m_end_pr:m_start])
                    text_acc.append("["+str(cnt))
                    text_acc.append(text[m_start:m_end])
                    text_acc.append("]")
                    m_start_pr = m_start
                    m_end_pr = m_end
                # add text after the last mention
                text_acc.append(text[m_end_pr:])
                doc_text = ' '.join(text_acc)
                fout.write(doc_text + "\n")
                for cnt, (_, _, uri_l) in enumerate(annotations, 1):
                    fout.write(str(cnt) + ": " + str(uri_l) + "\n")

def fix_tokenizatVion(doc_text):
    result = []
    for word in doc_text:
        if word in ["``", "''"]:
            result.append('"')
        else:
            result.append(word)
    return result


def create_necessary_folders():
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--other_datasets_folder",
                        default="/home/master_thesis_share/data/gerbil/gerbil_datasets/raw/")
    parser.add_argument("--output_folder", default="/home/master_thesis_share/data/gerbil/gerbil_datasets/readable_datasets/")
    parser.add_argument("--human_readable_output", type=bool, default=True)
    return parser.parse_args()


def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isfile(os.path.join(a_dir, name))]


def main():
    processDataset = ProcessDataset()
    for dataset in get_immediate_files(args.other_datasets_folder):
        if not args.human_readable_output:
            processDataset.process(os.path.join(args.other_datasets_folder, dataset))
        else:
            processDataset.process_readable(os.path.join(args.other_datasets_folder, dataset))
        print("Dataset", dataset, "done.")
    print("processDataset.all_gm_cnt =", processDataset.all_gm_cnt)
    print("processDataset.englishuri_gm_cnt =", processDataset.englishuri_gm_cnt)
    print("processDataset.valid_gms =", processDataset.valid_gms)  # valid gold mentions used
    print("processDataset.no_english_uri =", processDataset.no_english_uri)
    print("processDataset.unknown_ent_name =", processDataset.unknown_ent_name)


if __name__ == "__main__":
    args = _parse_args()
    create_necessary_folders()
    main()

