import argparse
import os
import sys
import preprocessing.util as util
import preprocessing.config as config
import traceback


def wikidump_to_new_format():
    doc_cnt = 0
    hyperlink2EntityId = util.EntityNameIdMap()
    hyperlink2EntityId.init_hyperlink2id()
    if args.debug:
        infilepath = config.base_folder + "data/mydata/tokenized_toy_wiki_dump2.txt"
        outfilepath = args.out_folder+"toy_wikidump.txt"
    else:
        infilepath = config.base_folder+"data/basic_data/tokenizedWiki.txt"
        outfilepath = args.out_folder+"wikidump.txt"
    with open(infilepath) as fin,\
         open(outfilepath, "w") as fout:
        in_mention = False
        for line in fin:
            line = line.rstrip()       # omit the '\n' character
            if line.startswith('<doc\xa0id="'):
                docid = line[9:line.find('"', 9)]
                doctitle = line[line.rfind('="') + 2:-2]
                fout.write("DOCSTART_" + docid + "_" + doctitle.replace(' ', '_') + "\n")
            elif line.startswith('<a\xa0href="'):
                ent_id = hyperlink2EntityId.hyperlink2id(line)
                if ent_id != config.unk_ent_id:
                    in_mention = True
                    fout.write("MMSTART_"+ent_id+"\n")
            elif line == '</doc>':
                fout.write("DOCEND\n")
                doc_cnt += 1
                if doc_cnt % 5000 == 0:
                    print("document counter: ", doc_cnt)
            elif line == '</a>':
                if in_mention:
                    fout.write("MMEND\n")
                    in_mention = False
            else:
                fout.write(line+"\n")


def subset_wikidump_only_relevant_mentions():
    # consider only the RLTD entities (484048). take them from the files
    entities_universe = set()
    with open("/home/other_projects/deep_ed/data/generated/nick/"
              "wikiid2nnid.txt") as fin:
        for line in fin:
            ent_id = line.split('\t')[0]
            entities_universe.add(ent_id)

    # filter wikidump
    doc_cnt = 0
    mention_errors = 0
    if args.debug:
        infilepath = args.out_folder+"toy_wikidump.txt"
        outfilepath = args.out_folder+"toy_wikidumpRLTD.txt"
    else:
        infilepath = args.out_folder+"wikidump.txt"
        outfilepath = args.out_folder+"wikidumpRLTD.txt"
    with open(infilepath) as fin, open(outfilepath, "w") as fout:
        in_mention_acc = []
        for line in fin:
            if line.startswith('DOCSTART_'):
                document_acc = [line]
                paragraph_acc = []
                paragraph_relevant = False
                in_mention_acc = []
            elif line == '*NL*\n':  # the or not necessary.
                # there is always a *NL* before DOCEND
                # end of paragraph so check if relevant
                if in_mention_acc:
                    in_mention_acc.append(line)
                else:
                    paragraph_acc.append(line)   # normal word
                if in_mention_acc:                          # we have a parsing error resulting to enter a mention but
                    #print("in_mention_acc", in_mention_acc)
                    mention_errors += 1                 # never detecting the end of it so still in_mention
                    paragraph_acc.extend(in_mention_acc[1:])
                    #print("paragraph_acc", paragraph_acc)
                    in_mention_acc = []
                if paragraph_relevant:
                    try:
                        assert(len(paragraph_acc) >= 4)  # MMSTART, word, MMEND  *NL*orDOCEND
                        assert(len(document_acc) >= 1)
                        document_acc.extend(paragraph_acc)
                    except AssertionError:
                        _, _, tb = sys.exc_info()
                        traceback.print_tb(tb)  # Fixed format
                        tb_info = traceback.extract_tb(tb)
                        filename, line, func, text = tb_info[-1]
                        print('An error occurred on line {} in statement {}'.format(line, text))
                        print("in_mention_acc", in_mention_acc)
                        print("paragraph_acc", paragraph_acc)


                paragraph_acc = []
                paragraph_relevant = False
            elif line == "DOCEND\n":
                assert(in_mention_acc == [])  # because there is always an *NL* before DOCEND
                if len(document_acc) > 1:
                    document_acc.append(line)
                    fout.write(''.join(document_acc))
                document_acc = []     # those 3 commands are not necessary
                paragraph_acc = []
                paragraph_relevant = False

                doc_cnt += 1
                if doc_cnt % 5000 == 0:
                    print("document counter: ", doc_cnt)
            elif line.startswith('MMSTART_'):
                if in_mention_acc:                          # not a parsing error resulting to enter a mention but
                    #print("in_mention_acc", in_mention_acc)
                    mention_errors += 1                 # never detecting the end of it so still in_mention
                    paragraph_acc.extend(in_mention_acc[1:])
                    #print("paragraph_acc", paragraph_acc)
                    in_mention_acc = []
                ent_id = line.rstrip()[8:]   # assert that ent_id in wiki_name_id_map
                if ent_id in entities_universe:
                    paragraph_relevant = True
                    in_mention_acc.append(line)
            elif line == 'MMEND\n':
                if in_mention_acc:
                    in_mention_acc.append(line)
                    paragraph_acc.extend(in_mention_acc)
                    in_mention_acc = []
                # else this mention is not in our universe so we don't accumulate it.
            else:
                if in_mention_acc:
                    in_mention_acc.append(line)
                else:
                    paragraph_acc.append(line)   # normal word

    print("mention_errors =", mention_errors)




def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities_universe_file",
                        default="/home/master_thesis_share/data/entities/entities_universe.txt")
    parser.add_argument("--out_folder", default="/home/master_thesis_share/data/new_datasets/wikidump/")
    parser.add_argument("--debug", type=bool, default=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()

    #if args.debug:
    #    wikidump_to_new_format()

    subset_wikidump_only_relevant_mentions()
