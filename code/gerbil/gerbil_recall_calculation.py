
import argparse
import os
import preprocessing.util as util
import rdflib


class ProcessDataset(object):
    def __init__(self):
        self.entityNameIdMap = util.EntityNameIdMap()
        self.entityNameIdMap.init_gerbil_compatible_ent_id()
        self.unknown_ent_name = dict()
        self.no_english_uri = dict()
        self.all_gm_cnt = dict()
        self.englishuri_gm_cnt = dict()
        self.valid_gms = dict()

    def process(self, filepath, filename):
        #the name of the dataset. just extract the last part of path
        unknown_ent_name = 0
        print("Dataset", filename, filepath)
        g = rdflib.Graph()
        dataset = g.parse(filepath)
        print("graph has %s statements." % len(g))
        print(g.serialize(format='n3'))












def main():
    if not os.path.exists("/home/master_thesis_share/data/gerbil/nif_original_datasets/"):
        os.makedirs("/home/master_thesis_share/data/gerbil/nif_original_datasets/")
    paths_names = [("/home/gerbil/gerbil_data/datasets/N3/Reuters-128.ttl", "n3_reuters_128")]
    #paths_names = [("/home/gerbil/gerbil_data/datasets/oke-challenge2016/evaluation-data/task1/evaluation-dataset-task1.ttl", "oke2016eval")]
    processDataset = ProcessDataset()
    for filepath, filename in paths_names:
        processDataset.process(filepath, filename)


if __name__ == "__main__":
    main()

