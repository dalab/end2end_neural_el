import model.config as config


def process_file(filename):
    entities = set()
    mentions = set()
    with open(config.base_folder+"data/new_datasets/"+filename) as fin:
        inmention = False
        mention_acc = []
        for line in fin:
            line = line.rstrip()     # omit the '\n' character
            if line.startswith('MMSTART_'):
                ent_id = line[8:]   # assert that ent_id in wiki_name_id_map
                entities.add(ent_id)
                inmention = True
                mention_acc = []
            elif line == 'MMEND':
                inmention = False
                mentions.add(' '.join(mention_acc))
            elif inmention:
                mention_acc.append(line)
    return entities, mentions


def main():
    train_entities, train_mentions = process_file("aida_train.txt")
    dev_entities, dev_mentions = process_file("aida_dev.txt")
    test_entities, test_mentions = process_file("aida_test.txt")

    print("len(train_entities) =", len(train_entities))
    print("len(dev_entities) =", len(dev_entities))
    print("len(test_entities) =", len(test_entities))

    print("train_entities.intersection(dev_entities) =", len(train_entities.intersection(dev_entities)))
    print("train_entities.intersection(test_entities) =", len(train_entities.intersection(test_entities)))
    print("dev_entities.intersection(test_entities) =", len(dev_entities.intersection(test_entities)))


    print("len(train_mentions) =", len(train_mentions))
    print("len(dev_mentions) =", len(dev_mentions))
    print("len(test_mentions) =", len(test_mentions))

    print("train_mentions.intersection(dev_mentions) =", len(train_mentions.intersection(dev_mentions)))
    print("train_mentions.intersection(test_mentions) =", len(train_mentions.intersection(test_mentions)))
    print("dev_mentions.intersection(test_mentions) =", len(dev_mentions.intersection(test_mentions)))

if __name__ == "__main__":
    main()





























