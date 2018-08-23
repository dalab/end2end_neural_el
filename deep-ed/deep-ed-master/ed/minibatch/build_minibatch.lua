-- Builds and fills a minibatch. In our case, minibatches are variable sized because they
-- contain all mentions in opt.batch_size documents.

assert(unk_ent_wikiid)

-- For train & test.
function empty_minibatch_with_ids(num_mentions)
  local inputs = {}
  -- ctxt_w_ids
  inputs[1] = {}
  inputs[1][1] = torch.ones(num_mentions, opt.ctxt_window):int():mul(unk_w_id)
  -- TO BE FILLED : inputs[1][2] =  CTXT WORD VECTORS : num_mentions x opt.ctxt_window x ent_vecs_size
  
  -- e_wikiids
  inputs[2] = {}
  inputs[2][1] = torch.ones(num_mentions, opt.num_cand_before_rerank):int():mul(unk_ent_thid)
  -- TO BE FILLED : inputs[2][2] = ENT VECTORS: num_mentions x opt.num_cand_before_rerank x ent_vecs_size
  
  -- p(e|m)
  inputs[3] = torch.zeros(num_mentions, opt.num_cand_before_rerank) 
  
  return inputs
end


-- Parse context words:
local function parse_context(parts) 
  local ctxt_word_ids = torch.ones(opt.ctxt_window):int():mul(unk_w_id)
  
  local lc = parts[4]
  local lc_words = split(lc, ' ')
  local lc_words_size = table_len(lc_words)
  local j = opt.ctxt_window / 2
  local i = lc_words_size
  while (j >= 1 and i >= 1) do
    while (i >= 2 and get_id_from_word(lc_words[i]) == unk_w_id) do
      i = i - 1
    end
    ctxt_word_ids[j] = get_id_from_word(lc_words[i])
    j = j - 1
    i = i - 1
  end
  
  local rc = parts[5]
  local rc_words = split(rc, ' ')
  local rc_words_size = table_len(rc_words)
  j = (opt.ctxt_window / 2) + 1
  i = 1
  while (j <= opt.ctxt_window and i <= rc_words_size) do
    while (i < rc_words_size and get_id_from_word(rc_words[i]) == unk_w_id) do
      i = i + 1
    end
    ctxt_word_ids[j] = get_id_from_word(rc_words[i])
    j = j + 1
    i = i + 1
  end
  
  return ctxt_word_ids
end


---------------------- Entity candidates: ----------------
local function parse_num_cand_and_grd_trth(parts)
  assert(parts[6] == 'CANDIDATES')  
  if parts[7] == 'EMPTYCAND' then
    return 0
  else
    local num_cand = 1
    while parts[7 + num_cand] ~= 'GT:' do
      num_cand = num_cand + 1
    end
    return num_cand
  end
end

--- Collect the grd truth label:
-- @return grd_trth_idx, grd_trth_ent_wikiid, grd_trth_prob
local function get_grd_trth(parts, num_cand, for_training)
  assert(parts[7 + math.max(1, num_cand)] == 'GT:')
  local grd_trth_str = parts[8 + math.max(1, num_cand)]
  local grd_trth_parts = split(grd_trth_str, ',')
  local grd_trth_idx = tonumber(grd_trth_parts[1])
  if grd_trth_idx ~= -1 then
    assert(grd_trth_idx >= 1 and grd_trth_idx <= num_cand)
    assert(grd_trth_str == grd_trth_idx .. ',' .. parts[6 + grd_trth_idx])
  else
    assert(not for_training)
  end  
  local grd_trth_prob = 0
  if grd_trth_idx > 0 then
    grd_trth_prob = math.min(1.0, math.max(1e-3, tonumber(grd_trth_parts[3])))
  end
  local grd_trth_ent_wikiid = unk_ent_wikiid
  if table_len(grd_trth_parts) >= 2 then
    grd_trth_ent_wikiid = tonumber(grd_trth_parts[2])
  end
  assert(grd_trth_ent_wikiid and grd_trth_prob)
  return grd_trth_idx, grd_trth_ent_wikiid, grd_trth_prob
end


-- @return grd_trth_idx, grd_trth_ent_wikiid, ent_cand_wikiids, log_p_e_m, log_p_e
function parse_candidate_entities(parts, for_training, orig_max_num_cand)
  -- Num of entity candidates
  local num_cand = parse_num_cand_and_grd_trth(parts)
  
  -- Ground truth index in the set of entity candidates, wikiid, p(e|m)
  local grd_trth_idx, grd_trth_ent_wikiid, grd_trth_prob = get_grd_trth(parts, num_cand, for_training)
  
  -- When including coreferent mentions this might happen
  if grd_trth_idx > orig_max_num_cand then 
    grd_trth_idx = -1
  end
  
  -- P(e|m) prior:
  local log_p_e_m = torch.ones(orig_max_num_cand):mul(-1e8)
  
  -- Vector of entity candidates ids
  local ent_cand_wikiids = torch.ones(orig_max_num_cand):int():mul(unk_ent_wikiid)
  
  -- Parse all candidates
  for cand_index = 1,math.min(num_cand, orig_max_num_cand) do
    local cand_parts = split(parts[6 + cand_index], ',')
    local cand_ent_wikiid = tonumber(cand_parts[1])
    assert(cand_ent_wikiid)
    ent_cand_wikiids[cand_index] = cand_ent_wikiid
    
    assert(for_training or get_thid(cand_ent_wikiid) ~= unk_ent_thid) -- RLTD entities have valid id    
    
    local cand_p_e_m = math.min(1.0, math.max(1e-3, tonumber(cand_parts[2])))
    log_p_e_m[cand_index] = torch.log(cand_p_e_m)
    local cand_p_e = get_ent_freq(cand_ent_wikiid)
  end
  
  -- Reinsert grd truth for training only
  if grd_trth_idx == -1 and for_training then
    assert(num_cand >= orig_max_num_cand)
    grd_trth_idx = orig_max_num_cand
    ent_cand_wikiids[grd_trth_idx] = grd_trth_ent_wikiid
    log_p_e_m[grd_trth_idx] = torch.log(grd_trth_prob)
  end
  
  -- Sanity checks:
  assert(log_p_e_m[1] == torch.max(log_p_e_m), line)
  assert(log_p_e_m[2] == torch.max(log_p_e_m:narrow(1, 2, orig_max_num_cand - 1)), line)
  if (grd_trth_idx ~= -1) then
    assert(grd_trth_ent_wikiid ~= unk_ent_wikiid)
    assert(ent_cand_wikiids[grd_trth_idx] == grd_trth_ent_wikiid)
    assert(log_p_e_m[grd_trth_idx] == torch.log(grd_trth_prob))
  else 
    assert(not for_training)
  end
  
  return grd_trth_idx, grd_trth_ent_wikiid, ent_cand_wikiids, log_p_e_m
end

local function get_cand_ent_thids(minibatch_tensor)
  return minibatch_tensor[2][1]
end



-- Fills in the minibatch and returns the idx of the grd truth entity:
function process_one_line(line, minibatch_tensor, mb_index, for_training)
  local parts = split(line, '\t')  
  
  -- Ctxt word ids:
  local ctxt_word_ids = parse_context(parts)
  minibatch_tensor[1][1][mb_index] = ctxt_word_ids
  
  -- Entity candidates:
  local grd_trth_idx, grd_trth_ent_wikiid, ent_cand_wikiids, log_p_e_m = 
        parse_candidate_entities(parts, for_training, opt.num_cand_before_rerank)
  minibatch_tensor[2][1][mb_index] = get_ent_thids(ent_cand_wikiids)
  assert(grd_trth_idx == -1 or (get_wikiid_from_thid(minibatch_tensor[2][1][mb_index][grd_trth_idx]) == grd_trth_ent_wikiid))
  
  -- log p(e|m):
  minibatch_tensor[3][mb_index] = log_p_e_m
  
  return grd_trth_idx
end


-- Reranking of entity candidates:
---- Keeps only 4 candidates, top 2 candidates from p(e|m) and top 2 from <ctxt_vec,e_vec>
max_num_cand = opt.keep_p_e_m + opt.keep_e_ctxt 

local function rerank(minibatch_tensor, targets, for_training)
  local num_mentions = get_cand_ent_thids(minibatch_tensor):size(1)
  local new_targets = torch.ones(num_mentions):mul(-1)
  
  -- Average of word vectors in a window of (at most) size 50 around the mention.
  local ctxt_vecs = minibatch_tensor[1][2]
  if opt.ctxt_window > 50 then 
    ctxt_vecs = ctxt_vecs:narrow(2, math.floor(opt.ctxt_window / 2) - 25 + 1, 50)
  end
  ctxt_vecs = ctxt_vecs:sum(2):view(num_mentions, ent_vecs_size)
  
  local new_log_p_e_m = correct_type(torch.ones(num_mentions, max_num_cand):mul(-1e8))
  local new_ent_cand_wikiids = torch.ones(num_mentions, max_num_cand):int():mul(unk_ent_wikiid)  
  local new_ent_cand_vecs = correct_type(torch.zeros(num_mentions, max_num_cand, ent_vecs_size))
  
  for k = 1,num_mentions do
    local ent_vecs = minibatch_tensor[2][2][k]
    local scores = ent_vecs * ctxt_vecs[k]
    assert(scores:size(1) == opt.num_cand_before_rerank)
    local _,ctxt_indices = torch.sort(scores, true)
    
    local added_indices = {}   
    for j = 1,opt.keep_e_ctxt do
      added_indices[ctxt_indices[j]] = 1
    end
    local j = 1
    while table_len(added_indices) < max_num_cand do
      added_indices[j] = 1
      j = j + 1
    end
    
    local new_grd_trth_idx = -1
    local i = 1
    for idx,_ in pairs(added_indices) do
      new_ent_cand_wikiids[k][i] = minibatch_tensor[2][1][k][idx]
      new_ent_cand_vecs[k][i] = minibatch_tensor[2][2][k][idx]
      new_log_p_e_m[k][i] = minibatch_tensor[3][k][idx]
      
      if idx == targets[k] then
        assert(minibatch_tensor[2][1][k][idx] ~= unk_ent_wikiid)
        new_grd_trth_idx = i
      end
      i = i + 1
    end

    -- Grd trth:
    if targets[k] > 0 and (new_grd_trth_idx == -1) then
      assert(targets[k] > opt.keep_p_e_m)
--      if not for_training then
--        print('Grd trth idx became -1') ------------> 79 times on AIDA test B for max_num_cand = 2 + 2
--      end
      if for_training then -- Reinsert the grd truth entity for training only
        new_ent_cand_wikiids[k][1] = minibatch_tensor[2][1][k][targets[k]]
        new_ent_cand_vecs[k][1] = minibatch_tensor[2][2][k][targets[k]]
        new_log_p_e_m[k][1] = minibatch_tensor[3][k][targets[k]]
        new_grd_trth_idx = 1
      end
    end
    
    new_targets[k] = new_grd_trth_idx
  end
  
  minibatch_tensor[2][1] = new_ent_cand_wikiids
  minibatch_tensor[2][2] = new_ent_cand_vecs
  minibatch_tensor[3] = new_log_p_e_m
  
  return minibatch_tensor, new_targets
end


-- Convert mini batch to correct type (e.g. move data to GPU):
-- ATT: Since lookup_table:forward() cannot be called twice without changing the output,
-- we have to keep the order here: 
function minibatch_to_correct_type(minibatch_tensor, targets, for_training)
  -- ctxt_w_vecs : num_mentions x opt.ctxt_window x ent_vecs_size
  minibatch_tensor[1][2] = word_lookup_table:forward(minibatch_tensor[1][1])
    
  -- ent_vecs : num_mentions x max_num_cand x ent_vecs_size
  minibatch_tensor[2][2] = ent_lookup_table:forward(minibatch_tensor[2][1])
  
  -- log p(e|m) : num_mentions x max_num_cand
  minibatch_tensor[3] = correct_type(minibatch_tensor[3])
  
  return rerank(minibatch_tensor, targets, for_training)
end


--- Used in data_loader.lua
function minibatch_table2tds(inputs_table)
  local inputs = tds.Hash()
  inputs[1] = tds.Hash()
  inputs[1][1] = inputs_table[1][1]
  inputs[2] = tds.Hash()
  inputs[2][1] = inputs_table[2][1]
  inputs[3] = inputs_table[3]
  return inputs
end

function minibatch_tds2table(inputs_tds)
  local inputs = {}
  inputs[1] = {}
  inputs[1][1] = inputs_tds[1][1]
  inputs[2] = {}
  inputs[2][1] = inputs_tds[2][1]
  inputs[3] = inputs_tds[3]
  return inputs
end


-- Utility functions used in test file for debug/vizualization
function get_cand_ent_wikiids(minibatch_tensor)
  local num_mentions = get_cand_ent_thids(minibatch_tensor):size(1)
  assert(get_cand_ent_thids(minibatch_tensor):size(2) == max_num_cand)
  local t = torch.zeros(num_mentions, max_num_cand):int()
  for i = 1, num_mentions do
    for j = 1, max_num_cand do
      t[i][j] = get_wikiid_from_thid(get_cand_ent_thids(minibatch_tensor)[i][j])
    end
  end
  return t
end

function get_log_p_e_m(minibatch_tensor)
  return minibatch_tensor[3]
end

function get_ctxt_word_ids(minibatch_tensor)
  return minibatch_tensor[1][1]
end
