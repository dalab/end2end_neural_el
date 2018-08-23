
import pickle
from nltk.tokenize import word_tokenize
import preprocessing.prepro_util as prepro_util
from preprocessing.util import load_wikiid2nnid, reverse_dict, load_wiki_name_id_map, FetchCandidateEntities
import os
import model.config as config


class BuildEntityUniverse(object):
    def __init__(self):
        self.entities_universe = set()
        self.fetchCandidateEntities = FetchCandidateEntities(Struct())
        prepro_util.args = Struct()

    def process(self, text, given_spans):
        # if we wanted to find entities for ed only then restrict it to given_spans instead of all spans
        chunk_words = word_tokenize(text)
        myspans = prepro_util.SamplesGenerator.all_spans(chunk_words)
        for left, right in myspans:
            cand_ent, _ = self.fetchCandidateEntities.process(chunk_words[left:right])
            # cand_ent is a list of strings (i.e. wikiids are still strings) not nums
            if cand_ent:
                self.entities_universe.update(cand_ent)

    def flush_entity_universe(self):
        print("len(self.entities_universe) =", len(self.entities_universe))
        entities_folder = config.base_folder+"data/entities/extension_entities/"
        _, wiki_id_name_map = load_wiki_name_id_map()
        if not os.path.exists(entities_folder):
            os.makedirs(entities_folder)

        def dump_entities(entity_set, name):
            with open(entities_folder + name+".pickle", 'wb') as handle:
                pickle.dump(entity_set, handle)
            with open(entities_folder + name+".txt", "w") as fout:
                for ent_id in entity_set:
                    fout.write(ent_id + "\t" + wiki_id_name_map[ent_id].replace(' ', '_') + "\n")

        dump_entities(self.entities_universe, "entities_universe")
        # now calculate the expansion i.e. from this universe omit the ones that we have already trained
        extension_entity_set = set()
        wikiid2nnid = load_wikiid2nnid()
        for wikiid in self.entities_universe:
            if wikiid not in wikiid2nnid:
                extension_entity_set.add(wikiid)

        print("len(extension_entity_set) =", len(extension_entity_set))
        dump_entities(extension_entity_set, "extension_entities")


class Struct(object):
    def __init__(self):
        self.p_e_m_choice = "yago"
        self.cand_ent_num = 30
        self.lowercase_p_e_m = False
        self.lowercase_spans = False
        self.max_mention_width = 10
        self.spans_separators = ["."]


if __name__ == "__main__":
    pass

