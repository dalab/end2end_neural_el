-- The code in this file does two things:
--   a) extracts and puts the entity relatedness dataset in two maps (reltd_validate and
--      reltd_test). Provides functions to evaluate entity embeddings on this dataset
--      (Table 1 in our paper).
--   b) extracts all entities that appear in any of the ED (as mention candidates) or
--      entity relatedness datasets. These are placed in an object called rewtr that will 
--      be used to restrict the set of entities for which we want to train entity embeddings 
--      (done with the file entities/learn_e2v/learn_a.lua).

if not opt then
  cmd = torch.CmdLine()
  cmd:option('-root_data_dir', '', 'Root path of the data, $DATA_PATH.')
  cmd:text()
  opt = cmd:parse(arg or {})
  assert(opt.root_data_dir ~= '', 'Specify a valid root_data_dir path argument.')  
end

dofile 'utils/utils.lua'
tds = tds or require 'tds'

if not ent_lines_4EX then
  -- Load a few pre-selected entities. For debug and unit tests.
  dofile 'entities/learn_e2v/4EX_wiki_words.lua'
end

----------------------- Some function definitions ------------

-- Loads the entity relatedness dataset (validation and test parts) in a map called reltd.
-- Format: reltd = {query_id q -> (query_entity e1, entity_candidates cand) }
--         cand = {e2 -> label}, where label is binary, if the candidate entity is related to e1
function load_reltd_set(rel_t7filename, rel_txtfilename, set_type)
  print('==> Loading relatedness ' .. set_type)
  if not paths.filep(rel_t7filename) then
    print('  ---> t7 file NOT found. Loading relatedness ' .. set_type .. ' from txt file instead (slower).')
    local reltd = tds.Hash()
    for line in io.lines(rel_txtfilename) do
      local parts = split(line, ' ')
      local label = tonumber(parts[1])
      assert(label == 0 or label == 1)
      
      local t = split(parts[2], ':')
      local q = tonumber(t[2])
      
      local i = 2
      while parts[i] ~= '#' do
        i = i + 1
      end
      i = i + 1

      ents = split(parts[i] , '-')
      e1 = tonumber(ents[1])
      e2 = tonumber(ents[2])

      if not reltd[q] then
        reltd[q] = tds.Hash()
        reltd[q].e1 = e1
        reltd[q].cand = tds.Hash()
      end
      reltd[q].cand[e2] = label
    end

    print('    Done loading relatedness ' .. set_type .. '. Num queries = ' .. table_len(reltd) .. '\n')
    print('Writing t7 File for future usage. Next time relatedness dataset will load faster!')
    torch.save(rel_t7filename, reltd)
    print('    Done saving.')
    return reltd
  else
    print('  ---> from t7 file.')
    return torch.load(rel_t7filename)
  end
end

-- Extracts all entities in the relatedness set, either candidates or :
local function extract_reltd_ents(reltd)
  local reltd_ents_direct = tds.Hash()
  for _,v in pairs(reltd) do
    reltd_ents_direct[v.e1] = 1
    for e2,_ in pairs(v.cand) do 
      reltd_ents_direct[e2] = 1
    end
  end
  return reltd_ents_direct
end

-- computes rltd scores based on a given entity_sim function
local function compute_e2v_rltd_scores(reltd, entity_sim)
  local scores = {}
  for q,_ in pairs(reltd) do
    scores[q] = {}
    for e2,_ in pairs(reltd[q].cand) do
      local aux = {}
      aux.e2 = e2
      aux.score = entity_sim(reltd[q].e1, e2)
      table.insert(scores[q], aux)
    end
    table.sort(scores[q], function(a,b) return a.score > b.score end)
  end
  return scores
end

-- computes rltd scores based on ground truth labels
local function compute_ideal_rltd_scores(reltd)
  local scores = {}
  for q,_ in pairs(reltd) do
    scores[q] = {}
    for e2,label in pairs(reltd[q].cand) do
      local aux = {}
      aux.e2 = e2
      aux.score = label
      table.insert(scores[q], aux)
    end
    table.sort(scores[q], function(a,b) return a.score > b.score end)
  end
  return scores
end

-- Mean Average Precision: 
-- https://en.wikipedia.org/wiki/Information_retrieval#Mean_average_precision
local function compute_MAP(scores, reltd)
  local sum_avgp = 0.0
  local num_queries = 0
  for q,_ in pairs(scores) do
    local avgp = 0.0
    local num_rel_ents_so_far = 0
    local num_ents_so_far = 0.0
    for _,c in pairs(scores[q]) do
      local e2 = c.e2
      local label = reltd[q].cand[e2]
      num_ents_so_far = num_ents_so_far + 1.0
      if (label == 1) then
        num_rel_ents_so_far = num_rel_ents_so_far + 1
        local precision = num_rel_ents_so_far / num_ents_so_far
        avgp = avgp + precision
      end
    end
    avgp = avgp / num_rel_ents_so_far
    sum_avgp = sum_avgp + avgp
    num_queries = num_queries + 1
  end

  assert(num_queries == table_len(reltd))
  return sum_avgp / num_queries
end

-- NDCG: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG
local function compute_DCG(k, q, scores_q, reltd)
  local dcg = 0.0
  local i = 0
  for _,c in pairs(scores_q) do
    local label = reltd[q].cand[c.e2]
    i = i + 1
    if (label == 1) and i <= k then
      dcg = dcg + (1.0 / math.log(math.max(2,i) + 0.0, 2))
    end
  end
  return dcg
end

local function compute_NDCG(k, all_table)
  local sum_ndcg = 0.0
  local num_queries = 0
  for q,_ in pairs(all_table.scores) do
    local dcg = compute_DCG(k, q, all_table.scores[q], all_table.reltd)
    local idcg = compute_DCG(k, q, all_table.ideals_rltd_scores[q], all_table.reltd)
    assert(dcg <= idcg, dcg .. ' ' .. idcg)
    sum_ndcg = sum_ndcg + (dcg / idcg)
    num_queries = num_queries + 1
  end

  assert(num_queries == table_len(all_table.reltd))
  return sum_ndcg / num_queries
end

local function compute_relatedness_metrics_from_maps(entity_sim, validate_set, test_set)
  print(yellow('Entity Relatedness quality measure:'))
  
  collectgarbage(); collectgarbage();
  local ideals_rltd_validate_scores = compute_ideal_rltd_scores(validate_set)
  collectgarbage(); collectgarbage();
  local ideals_rltd_test_scores = compute_ideal_rltd_scores(test_set)
  collectgarbage(); collectgarbage();
  
  assert(math.abs(-1 + compute_MAP(ideals_rltd_validate_scores, validate_set)) < 0.001) 
  collectgarbage(); collectgarbage();  
  assert(math.abs(-1 + compute_MAP(ideals_rltd_test_scores, test_set)) < 0.001) 

  collectgarbage(); collectgarbage();
  local scores_validate = compute_e2v_rltd_scores(validate_set, entity_sim)
  collectgarbage(); collectgarbage();
  local scores_test = compute_e2v_rltd_scores(test_set, entity_sim)
  collectgarbage(); collectgarbage();

  local validate_table = {}
  validate_table.scores = scores_validate
  validate_table.ideals_rltd_scores = ideals_rltd_validate_scores
  validate_table.reltd = validate_set
  local test_table = {}
  test_table.scores = scores_test
  test_table.ideals_rltd_scores = ideals_rltd_test_scores
  test_table.reltd = test_set

  local map_validate = compute_MAP(scores_validate, validate_set)
  collectgarbage(); collectgarbage();
  local ndcg_1_validate = compute_NDCG(1, validate_table)
  collectgarbage(); collectgarbage();
  local ndcg_5_validate = compute_NDCG(5, validate_table)
  collectgarbage(); collectgarbage();
  local ndcg_10_validate = compute_NDCG(10, validate_table)
  collectgarbage(); collectgarbage();
  local total = map_validate + ndcg_1_validate + ndcg_5_validate + ndcg_10_validate

  local map_validate_str = blue_num_str(map_validate)
  local ndcg_1_validate_str = blue_num_str(ndcg_1_validate)
  local ndcg_5_validate_str = blue_num_str(ndcg_5_validate)
  local ndcg_10_validate_str = blue_num_str(ndcg_10_validate)  
  local total_str = blue_num_str(total)

  local map_test = red(string.format("%.3f", compute_MAP(scores_test, test_set)))
  collectgarbage(); collectgarbage();
  local ndcg_1_test = red(string.format("%.3f", compute_NDCG(1, test_table)))
  collectgarbage(); collectgarbage();
  local ndcg_5_test = red(string.format("%.3f", compute_NDCG(5, test_table)))
  collectgarbage(); collectgarbage();
  local ndcg_10_test = red(string.format("%.3f", compute_NDCG(10, test_table)))
  collectgarbage(); collectgarbage();

  print(yellow('measure    ='), 'NDCG1' , 'NDCG5', 'NDCG10', 'MAP', 'TOTAL VALIDATION')
  print(yellow('our (vald) ='), ndcg_1_validate_str, ndcg_5_validate_str, ndcg_10_validate_str, map_validate_str, total_str)  
  print(yellow('our (test) ='), ndcg_1_test, ndcg_5_test, ndcg_10_test, map_test)  
  print(yellow('Yamada\'16  ='), 0.59, 0.56, 0.59, 0.52)  
  print(yellow('WikiMW     ='), 0.54, 0.52, 0.55, 0.48)  
end


--------------------------------------------------------------
------------------------ Main code ---------------------------
--------------------------------------------------------------
rel_test_txtfilename = opt.root_data_dir .. 'basic_data/relatedness/test.svm'
rel_test_t7filename = opt.root_data_dir .. 'generated/relatedness_test.t7'

rel_validate_txtfilename = opt.root_data_dir .. 'basic_data/relatedness/validate.svm'
rel_validate_t7filename = opt.root_data_dir .. 'generated/relatedness_validate.t7'

local reltd_validate = load_reltd_set(rel_validate_t7filename, rel_validate_txtfilename, 'validate')
local reltd_test = load_reltd_set(rel_test_t7filename, rel_test_txtfilename, 'test')

local reltd_ents_direct_validate = extract_reltd_ents(reltd_validate)
local reltd_ents_direct_test = extract_reltd_ents(reltd_test)


local rewtr_t7filename = opt.root_data_dir .. 'generated/all_candidate_ents_ed_rltd_datasets_RLTD.t7'
print('==> Loading relatedness thid tensor')
if not paths.filep(rewtr_t7filename) then
  print('  ---> t7 file NOT found. Loading reltd_ents_wikiid_to_rltdid from txt file instead (slower).')

  -- Gather the restricted set of entities for which we train entity embeddings:
  local rltd_all_ent_wikiids = tds.Hash()
  
  -- 1) From the relatedness dataset
  for ent_wikiid,_ in pairs(reltd_ents_direct_validate) do
    rltd_all_ent_wikiids[ent_wikiid] = 1
  end
  for ent_wikiid,_ in pairs(reltd_ents_direct_test) do
    rltd_all_ent_wikiids[ent_wikiid] = 1
  end
  
  -- 1.1) From a small dataset (used for debugging / unit testing).
  for _,line in pairs(ent_lines_4EX) do
    local parts = split(line, '\t')
    assert(table_len(parts) == 3)
    ent_wikiid = tonumber(parts[1])
    assert(ent_wikiid)
    rltd_all_ent_wikiids[ent_wikiid] = 1
  end

  -- end2end_neural_el start           insert my entity universe
  entities_file_path = '../../data/entities/entities_universe.txt'
  wikiid2nnid_folder_path = '../../data/entities/wikiid2nnid/'
  
  --entities_file_path = '../../data/entities/extension_entities/extension_entities.txt'
  --wikiid2nnid_folder_path = '../../data/entities/extension_entities/wikiid2nnid/'

  it, _ = io.open(entities_file_path)
  local line = it:read()
  while line do
    local parts = split(line, '\t')
    local ent_wikiid = tonumber(parts[1])
    assert(ent_wikiid)
    rltd_all_ent_wikiids[ent_wikiid] = 1
    line = it:read()
  end
  -- end2end_neural_el end

--[=====[ 
  -- 2) From all ED datasets:
  local files = {'aida_train.csv', 'aida_testA.csv', 'aida_testB.csv', 
    'wned-aquaint.csv', 'wned-msnbc.csv', 'wned-ace2004.csv',
    'wned-clueweb.csv', 'wned-wikipedia.csv'}
  for _,f in pairs(files) do
    it, _ = io.open(opt.root_data_dir .. 'generated/test_train_data/' .. f)
    local line = it:read()
    while line do
      local parts = split(line, '\t')
      assert(parts[6] == 'CANDIDATES')
      assert(parts[table_len(parts) - 1] == 'GT:')
      if parts[7] ~= 'EMPTYCAND' then
        for i = 7, table_len(parts) - 2 do
          local p = split(parts[i], ',')
          local ent_wikiid = tonumber(p[1])
          assert(ent_wikiid)
          rltd_all_ent_wikiids[ent_wikiid] = 1
        end

        local p = split(parts[table_len(parts)], ',')
        if table_len(p) >= 2 then
          local ent_wikiid = tonumber(p[2])
          assert(ent_wikiid)
        end
      end
      line = it:read()
    end
  end
--]=====]

  -- Insert unk_ent_wikiid
  local unk_ent_wikiid = 1
  rltd_all_ent_wikiids[unk_ent_wikiid] = 1
  
  -- Sort all wikiids
  local sorted_rltd_all_ent_wikiids = tds.Vec()
  for ent_wikiid,_ in pairs(rltd_all_ent_wikiids) do
    sorted_rltd_all_ent_wikiids:insert(ent_wikiid)           -- push back to the vector
  end
  sorted_rltd_all_ent_wikiids:sort(function(a,b) return a < b end)  -- sort the vector
  
  local reltd_ents_wikiid_to_rltdid = tds.Hash()
  for rltd_id,wikiid in pairs(sorted_rltd_all_ent_wikiids) do
    reltd_ents_wikiid_to_rltdid[wikiid] = rltd_id
  end
  
  rewtr = tds.Hash()
  rewtr.reltd_ents_wikiid_to_rltdid = reltd_ents_wikiid_to_rltdid
  rewtr.reltd_ents_rltdid_to_wikiid = sorted_rltd_all_ent_wikiids
  rewtr.num_rltd_ents = #sorted_rltd_all_ent_wikiids
  
  -- end2end_neural_el       print these to file as well
  print('Now printing to file the wikiid to torch ids mappings...')
  out_file = wikiid2nnid_folder_path .. 'wikiid2nnid.txt'
  --out_file = opt.root_data_dir .. 'generated/wikiid2nnid/wikiid2nnid.txt' -- 'generated/my_wikiid_nnid_map/reltd_ents_wikiid_to_rltdid.txt'
  ouf = assert(io.open(out_file, "w"))
  for k, v in pairs(reltd_ents_wikiid_to_rltdid) do
    -- print(k, v)
    ouf:write(k .. '\t' .. v .. '\n')
  end
  ouf:flush()
  io.close(ouf) 
  out_file = wikiid2nnid_folder_path .. 'nnid2wikiid.txt'
  --out_file = opt.root_data_dir .. 'generated/wikiid2nnid/nnid2wikiid.txt'
  ouf = assert(io.open(out_file, "w"))
  for k, v in pairs(sorted_rltd_all_ent_wikiids) do
    -- print(k, v)
    ouf:write(k .. '\t' .. v .. '\n')
  end
  ouf:flush()
  io.close(ouf) 
  -- end2end_neural_el


  print('Writing reltd_ents_wikiid_to_rltdid to t7 File for future usage.')
  torch.save(rewtr_t7filename, rewtr)
  print('    Done saving.')
else
  print('  ---> from t7 file.')
  rewtr = torch.load(rewtr_t7filename)
end

print('    Done loading relatedness sets. Num queries test = ' .. table_len(reltd_test) .. 
  '. Num queries valid = ' .. table_len(reltd_validate) ..
  '. Total num ents restricted set = ' .. rewtr.num_rltd_ents)

-- Main function that computes results for the entity relatedness dataset (Table 1 of 
-- our paper) given any entity similarity function as input. 
function compute_relatedness_metrics(entity_sim)
  compute_relatedness_metrics_from_maps(entity_sim, reltd_validate, reltd_test)
end
