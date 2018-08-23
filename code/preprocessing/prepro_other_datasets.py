import argparse
import os
import preprocessing.util as util
import xml.etree.ElementTree as ET
from subprocess import call


class ProcessDataset(object):
    def __init__(self):
        self.entityNameIdMap = util.EntityNameIdMap()
        self.entityNameIdMap.init_compatible_ent_id()

    def process(self, dataset_folder):
        #the name of the dataset. just extract the last part of path
        dataset = os.path.basename(os.path.normpath(dataset_folder))
        print("processing dataset:", dataset)
        xml_filepath = os.path.join(dataset_folder, "{}.xml".format(dataset))
        rawtext_folder = os.path.join(dataset_folder, "RawText")
        if not (os.path.exists(rawtext_folder) and os.path.exists(xml_filepath)):
            print("Dataset ", dataset, "is not in proper format.")
            return
        tree = ET.parse(xml_filepath)
        root = tree.getroot()

        unknown_ent_name = 0
        with open(args.output_folder+"temp_"+dataset, "w") as fout:
            for document in root.findall('document'):
                docName = document.get('docName')
                with open(os.path.join(rawtext_folder, docName)) as fin:
                   text = fin.read()  # the whole raw text
                # accumulates the text of one document until we process all mentions then outputs to fout
                text_acc = []
                text_acc.append("DOCSTART_"+docName.replace(' ', '_').replace('.', '_'))
                m_start_pr = 0  # previous mention start position
                m_end_pr = 0    # previous mention end position
                for annotation in document.findall('annotation'):
                    m_start = int(annotation.find('offset').text)  # inclusive
                    m_end = m_start + int(annotation.find('length').text)  # exclusive

                    text_acc.append(text[m_end_pr:m_start])
                    ent_id = self.entityNameIdMap.compatible_ent_id(
                        name=annotation.find('wikiName').text
                    )
                    if ent_id is not None:
                        text_acc.append("MMSTART_"+ent_id)
                        text_acc.append(text[m_start:m_end])
                        text_acc.append("MMEND")
                    else:  # if not then omit this mention
                        unknown_ent_name += 1
                        text_acc.append(text[m_start:m_end])
                        print("unknown entity name: ", annotation.find('wikiName').text)
                    m_start_pr = m_start
                    m_end_pr = m_end
                # add text after the last mention
                text_acc.append(text[m_end_pr:])
                text_acc.append("DOCEND\n")
                fout.write(' '.join(text_acc))

        print("unknown_ent_name =", unknown_ent_name)
        tokenize_command = 'cd {}; java -cp "*" '\
            'edu.stanford.nlp.process.PTBTokenizer -options "tokenizeNLs=True" < {} > {}'.format(
            args.stanford_tokenizer_folder, os.path.abspath(args.output_folder+"temp_"+dataset),
            os.path.abspath(args.output_folder+dataset+".txt"))
        print(tokenize_command)
        call(tokenize_command, shell=True)
        os.remove(args.output_folder+"temp_"+dataset)



def create_necessary_folders():
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--other_datasets_folder",
                        default="../data/basic_data/test_datasets/wned-datasets/")
    parser.add_argument("--output_folder", default="../data/new_datasets/")
    parser.add_argument("--stanford_tokenizer_folder",
                        default="../data/stanford_core_nlp/stanford-corenlp-full-2017-06-09")
    return parser.parse_args()

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def main():
    processDataset = ProcessDataset()
    for dataset in get_immediate_subdirectories(args.other_datasets_folder):
        processDataset.process(os.path.join(args.other_datasets_folder, dataset))
        print("Dataset ", dataset, "done.")

if __name__ == "__main__":
    args = _parse_args()
    create_necessary_folders()
    main()


