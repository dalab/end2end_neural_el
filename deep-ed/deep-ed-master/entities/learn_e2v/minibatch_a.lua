assert(opt.entities == '4EX' or opt.entities == 'ALL' or opt.entities == 'RLTD', opt.entities)

function empty_minibatch()
  local ctxt_word_ids = torch.ones(opt.batch_size, opt.num_words_per_ent, opt.num_neg_words):mul(unk_w_id)
  local ent_component_words = torch.ones(opt.batch_size, opt.num_words_per_ent):int()
  local ent_wikiids = torch.ones(opt.batch_size):int()
  local ent_thids = torch.ones(opt.batch_size):int()
  return {{ctxt_word_ids}, {ent_component_words}, {ent_thids, ent_wikiids}}
end

-- Get functions:
function get_pos_and_neg_w_ids(minibatch) 
  return minibatch[1][1]
end
function get_pos_and_neg_w_vecs(minibatch) 
  return minibatch[1][2]
end
function get_pos_and_neg_w_unig_at_power(minibatch) 
  return minibatch[1][3]
end
function get_ent_wiki_w_ids(minibatch) 
  return minibatch[2][1]
end
function get_ent_wiki_w_vecs(minibatch) 
  return minibatch[2][2]
end
function get_ent_thids_batch(minibatch) 
  return minibatch[3][1]
end
function get_ent_wikiids(minibatch) 
  return minibatch[3][2]
end


-- Fills in the minibatch and returns the grd truth word index per each example.
-- An example in our case is an entity, a positive word sampled from \hat{p}(e|m) 
-- and several negative words sampled from \hat{p}(w)^\alpha.
function process_one_line(line, minibatch, mb_index)
  if opt.entities == '4EX' then
    line = ent_lines_4EX[ent_names_4EX[math.random(1, table_len(ent_names_4EX))]]
  end
  
  local parts = split(line, '\t')
  local num_parts = table_len(parts)
  
  if num_parts == 3 then  ---------> Words from the Wikipedia canonical page
    assert(table_len(parts) == 3, line)
    ent_wikiid = tonumber(parts[1])
    words_plus_stop_words = split(parts[3], ' ')
    
  else --------> Words from Wikipedia hyperlinks
    assert(num_parts >= 9, line .. ' --> ' .. num_parts)
    assert(parts[6] == 'CANDIDATES', line)

    local last_part = parts[num_parts]
    local ent_str = split(last_part, ',')
    ent_wikiid = tonumber(ent_str[2])
    
    words_plus_stop_words = {}
    local left_ctxt_w = split(parts[4], ' ')
    local left_ctxt_w_num = table_len(left_ctxt_w)
    for i = math.max(1, left_ctxt_w_num - opt.hyp_ctxt_len + 1), left_ctxt_w_num do 
      table.insert(words_plus_stop_words, left_ctxt_w[i])
    end
    local right_ctxt_w = split(parts[5], ' ')
    local right_ctxt_w_num = table_len(right_ctxt_w)
    for i = 1, math.min(right_ctxt_w_num, opt.hyp_ctxt_len) do 
      table.insert(words_plus_stop_words, right_ctxt_w[i])
    end         
  end
  
  assert(ent_wikiid)
  local ent_thid = get_thid(ent_wikiid)
  assert(get_wikiid_from_thid(ent_thid) == ent_wikiid)
  get_ent_thids_batch(minibatch)[mb_index] = ent_thid
  assert(get_ent_thids_batch(minibatch)[mb_index] == ent_thid)
  
  get_ent_wikiids(minibatch)[mb_index] = ent_wikiid
  

  -- Remove stop words from entity wiki words representations.
  local positive_words_in_this_iter =  {}
  local num_positive_words_this_iter = 0
  for _,w in pairs(words_plus_stop_words) do
    if contains_w(w) then
      table.insert(positive_words_in_this_iter, w)
      num_positive_words_this_iter = num_positive_words_this_iter + 1
    end
  end  
  
  -- Try getting some words from the entity title if the canonical page is empty.
  if num_positive_words_this_iter == 0 then 
    local ent_name = parts[2]
    words_plus_stop_words = split_in_words(ent_name)
    for _,w in pairs(words_plus_stop_words) do
      if contains_w(w) then
        table.insert(positive_words_in_this_iter, w)
        num_positive_words_this_iter = num_positive_words_this_iter + 1
      end
    end  
    
    -- Still empty ? Get some random words then.
    if num_positive_words_this_iter == 0 then
      table.insert(positive_words_in_this_iter, get_word_from_id(random_unigram_at_unig_power_w_id()))
    end
  end
  
  local targets = torch.zeros(opt.num_words_per_ent):int()
  
  -- Sample some negative words:
  get_pos_and_neg_w_ids(minibatch)[mb_index]:apply(
    function(x) 
      -- Random negative words sampled sampled from \hat{p}(w)^\alpha.
      return random_unigram_at_unig_power_w_id() 
    end
  )

  -- Sample some positive words:
  for i = 1,opt.num_words_per_ent do
    local positive_w = positive_words_in_this_iter[math.random(1, num_positive_words_this_iter)]
    local positive_w_id = get_id_from_word(positive_w)
        
    -- Set the positive word in a random position. Remember that index (used in training).
    local grd_trth = math.random(1, opt.num_neg_words)
    get_ent_wiki_w_ids(minibatch)[mb_index][i] = positive_w_id
    assert(get_ent_wiki_w_ids(minibatch)[mb_index][i] == positive_w_id)
    targets[i] = grd_trth
    get_pos_and_neg_w_ids(minibatch)[mb_index][i][grd_trth] = positive_w_id
  end

  return targets
end


-- Fill minibatch with word and entity vectors:
function postprocess_minibatch(minibatch, targets)
  
  minibatch[1][1] = get_pos_and_neg_w_ids(minibatch):view(opt.batch_size * opt.num_words_per_ent * opt.num_neg_words)
  minibatch[2][1] = get_ent_wiki_w_ids(minibatch):view(opt.batch_size * opt.num_words_per_ent)
  
  -- ctxt word vecs
  minibatch[1][2] = w2vutils:lookup_w_vecs(get_pos_and_neg_w_ids(minibatch))
  
  minibatch[1][3] = torch.zeros(opt.batch_size * opt.num_words_per_ent * opt.num_neg_words)
  minibatch[1][3]:map(minibatch[1][1], function(_,w_id) return get_w_unnorm_unigram_at_power(w_id) end)
end


-- Convert mini batch to correct type (e.g. move data to GPU):
function minibatch_to_correct_type(minibatch)
  minibatch[1][1] = correct_type(minibatch[1][1])
  minibatch[2][1] = correct_type(minibatch[2][1])
  minibatch[1][2] = correct_type(minibatch[1][2])
  minibatch[1][3] = correct_type(minibatch[1][3])
  minibatch[3][1] = correct_type(minibatch[3][1])
end