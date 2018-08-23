


def p_e_m_disamb_redirect_wikinameid_maps():

    wall_start = time.time()
    redirections = dict()
    with open(config.base_folder + "data/basic_data/wiki_redirects.txt") as fin:
        redirections_errors = 0
        for line in fin:
            # line = line[:-1]
            line = line.rstrip()
            try:
                old_title, new_title = line.split("\t")
                redirections[old_title] = new_title
            except ValueError:
                redirections_errors += 1

    print("load redirections. wall time:", (time.time() - wall_start)/60, " minutes")
    print("redirections_errors: ", redirections_errors)

    wall_start = time.time()
    disambiguations_ids = set()
    disambiguations_titles = set()
    disambiguations_errors = 0
    with open(config.base_folder + "data/basic_data/wiki_disambiguation_pages.txt") as fin:
        for line in fin:
            line = line.rstrip()
            try:
                article_id, title = line.split("\t")
                disambiguations_ids.add(int(article_id))
                disambiguations_titles.add(title)
            except ValueError:
                disambiguations_errors += 1
    print("load disambiguations. wall time:", (time.time() - wall_start)/60, " minutes")
    print("disambiguations_errors: ", disambiguations_errors)

    wall_start = time.time()
    wiki_name_id_map = dict()
    wiki_name_id_map_lower = dict()   # i lowercase the names
    wiki_id_name_map = dict()
    wiki_name_id_map_errors = 0
    with open(config.base_folder + "data/basic_data/wiki_name_id_map.txt") as fin:
        for line in fin:
            line = line.rstrip()
            try:
                wiki_title, wiki_id = line.split("\t")
                wiki_name_id_map[wiki_title] = int(wiki_id)
                wiki_name_id_map_lower[wiki_title.lower()] = int(wiki_id)
                wiki_id_name_map[int(wiki_id)] = wiki_title
            except ValueError:
                wiki_name_id_map_errors += 1
    print("load wiki_name_id_map. wall time:", (time.time() - wall_start)/60, " minutes")
    print("wiki_name_id_map_errors: ", wiki_name_id_map_errors)

    wall_start = time.time()
    p_e_m = dict()  # for each mention we have a list of tuples (ent_id, score)
    p_e_m_errors = 0
    line_cnt = 0
    with open(config.base_folder + "data/p_e_m/prob_yago_crosswikis_wikipedia_p_e_m.txt") as fin:
        for line in fin:
            line = line.rstrip()
            try:
                temp = line.split("\t")
                mention, entities = temp[0],  temp[2:2+config.cand_ent_num]
                res = []
                for e in entities:
                    ent_id, score, _ = e.split(',', 2)
                    #print(ent_id, score)
                    res.append((int(ent_id), float(score)))
                p_e_m[mention] = res    # for each mention we have a list of tuples (ent_id, score)
                #print(repr(line))
                #print(mention, p_e_m[mention])
            #except ValueError:
            except Exception as esd:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                p_e_m_errors += 1
                print("error in line: ", repr(line))
                #line_cnt += 1
                #if line_cnt > 100:
                #    break
    print("end of p_e_m reading. wall time:", (time.time() - wall_start)/60, " minutes")
    print("p_e_m_errors: ", p_e_m_errors)

    with open(config.base_folder+"data/serializations/p_e_m_disamb_redirect_wikinameid_maps.pickle", 'wb') as handle:
        pickle.dump((p_e_m, config.cand_ent_num, disambiguations_ids, disambiguations_titles, redirections, \
                     wiki_name_id_map, wiki_id_name_map, wiki_name_id_map_lower), handle)
    return p_e_m, disambiguations_ids, redirections, wiki_name_id_map
