-- Entity embeddings utilities

assert(opt.num_words_per_ent)

-- Word lookup:
geom_w2v_M = w2vutils.M:float()

-- Stats:
local num_invalid_ent_wikiids = 0
local total_ent_wiki_vec_requests = 0
local last_wrote = 0
local function invalid_ent_wikiids_stats(ent_thid)
  total_ent_wiki_vec_requests = total_ent_wiki_vec_requests + 1
  if ent_thid == unk_ent_thid then
    num_invalid_ent_wikiids = num_invalid_ent_wikiids + 1
  end
  if (num_invalid_ent_wikiids % 15000 == 0 and num_invalid_ent_wikiids ~= last_wrote) then
    last_wrote = num_invalid_ent_wikiids
    local perc = 100.0 * num_invalid_ent_wikiids / total_ent_wiki_vec_requests
    print(red('*** Perc invalid ent wikiids = ' .. perc .. ' . Absolute num = ' .. num_invalid_ent_wikiids))
  end
end

-- ent id -> vec
function geom_entwikiid2vec(ent_wikiid)
  local ent_thid = get_thid(ent_wikiid)
  assert(ent_thid)
  invalid_ent_wikiids_stats(ent_thid)
  local ent_vec = nn.Normalize(2):forward(lookup_ent_vecs.weight[ent_thid]:float())
  return ent_vec
end

-- ent name -> vec
local function geom_entname2vec(ent_name)
  assert(ent_name)
  return geom_entwikiid2vec(get_ent_wikiid_from_name(ent_name))
end

function entity_similarity(e1_wikiid, e2_wikiid)
  local e1_vec = geom_entwikiid2vec(e1_wikiid)
  local e2_vec = geom_entwikiid2vec(e2_wikiid)
  return e1_vec * e2_vec
end


local function geom_top_k_closest_words(ent_name, ent_vec, k)
  local tf_map = ent_wiki_words_4EX[ent_name]
  local w_not_found = {}
  for w,_ in pairs(tf_map) do
    if tf_map[w] >= 10 then
      w_not_found[w] = tf_map[w]
    end
  end
  
  distances = geom_w2v_M * ent_vec

  local best_scores, best_word_ids = topk(distances, k)
  local returnwords = {}
  local returndistances = {}
  for i = 1,k do
    local w = get_word_from_id(best_word_ids[i])
    if is_stop_word_or_number(w) then
      table.insert(returnwords, red(w))
    elseif tf_map[w] then
      if tf_map[w] >= 15 then
        table.insert(returnwords, yellow(w .. '{' .. tf_map[w] .. '}'))
      else
        table.insert(returnwords, skyblue(w .. '{' .. tf_map[w] .. '}'))
      end
      w_not_found[w] = nil
    else
      table.insert(returnwords, w)
    end
    assert(best_scores[i] == distances[best_word_ids[i]], best_scores[i] .. '  ' .. distances[best_word_ids[i]])
    table.insert(returndistances, distances[best_word_ids[i]])
  end
  return returnwords, returndistances, w_not_found
end


local function geom_most_similar_words_to_ent(ent_name, k)
  local ent_wikiid = get_ent_wikiid_from_name(ent_name)
  local k = k or 1
  local ent_vec = geom_entname2vec(ent_name)
  assert(math.abs(1 - ent_vec:norm()) < 0.01 or ent_vec:norm() == 0, ':::: ' .. ent_vec:norm())
  
  print('\nTo entity: ' .. blue(ent_name) .. '; vec norm = ' .. ent_vec:norm() .. ':')
  neighbors, scores, w_not_found = geom_top_k_closest_words(ent_name, ent_vec, k)
  print(green('WORDS MODEL: ') .. list_with_scores_to_str(neighbors, scores))
  
  local str = yellow('WORDS NOT FOUND: ')
  for w,tf in pairs(w_not_found) do
    if tf >= 20 then
      str = str .. yellow(w .. '{' .. tf .. '}; ')
    else
      str = str .. w .. '{' .. tf .. '}; '
    end
  end
  print('\n' .. str)
  print('============================================================================')
end


-- Unit tests :
function geom_unit_tests()
  print('\n' .. yellow('Words to Entity Similarity test:'))  
  for i=1,table_len(ent_names_4EX) do
    geom_most_similar_words_to_ent(ent_names_4EX[i], 200)
  end
end
