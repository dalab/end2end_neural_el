dofile 'ed/test/coref_persons.lua'
dofile 'ed/test/ent_freq_stats_test.lua'
dofile 'ed/test/ent_p_e_m_stats_test.lua'

testAccLogger = Logger(opt.root_data_dir .. 'generated/ed_models/training_plots/f1-' .. banner)

---------------------------- Define all datasets -----------------------------

datasets = {}

datasets['aida-A'] = opt.root_data_dir .. 'generated/test_train_data/aida_testA.csv' -- Validation set
datasets['aida-B'] = opt.root_data_dir .. 'generated/test_train_data/aida_testB.csv'
datasets['MSNBC'] = opt.root_data_dir .. 'generated/test_train_data/wned-msnbc.csv'
datasets['AQUAINT'] = opt.root_data_dir .. 'generated/test_train_data/wned-aquaint.csv'
datasets['ACE04'] = opt.root_data_dir .. 'generated/test_train_data/wned-ace2004.csv'

------- Uncomment the following lines if you want to test on more datasets during training (will be slower).
--datasets['train-aida'] = opt.root_data_dir .. 'generated/test_train_data/aida_train.csv'
--datasets['CLUEWEB'] = opt.root_data_dir .. 'generated/test_train_data/wned-clueweb.csv'
--datasets['WNED-WIKI'] = opt.root_data_dir .. 'generated/test_train_data/wned-wikipedia.csv'


local classes = {}
for i = 1,max_num_cand do
  table.insert(classes, i)
end

--------------------------- Functions ------------------------------------------
local function get_dataset_lines(banner)
  it, _ = io.open(datasets[banner])
  local all_doc_lines = tds.Hash()
  local line = it:read()
  while line do
    local parts = split(line, '\t')
    local doc_name = parts[1]
    if not all_doc_lines[doc_name] then
      all_doc_lines[doc_name] = tds.Hash()
    end
    all_doc_lines[doc_name][1 + #all_doc_lines[doc_name]] = line
    line = it:read()
  end
  -- Gather coreferent mentions to increase accuracy.
  return build_coreference_dataset(all_doc_lines, banner)
end


local function get_dataset_num_non_empty_candidates(dataset_lines)
  local num_predicted = 0
  for doc_id, lines_map in pairs(dataset_lines) do 
    for _,sample_line in pairs(lines_map)  do
      local parts = split(sample_line, '\t')
      if parts[7] ~= 'EMPTYCAND' then
        num_predicted = num_predicted + 1
      end
    end
  end
  return num_predicted
end

local function test_one(banner, f1_scores, epoch)
  collectgarbage(); collectgarbage();
  -- Load dataset lines
  local dataset_lines = get_dataset_lines(banner)
  
  local dataset_num_mentions = 0
  for doc_id, lines_map in pairs(dataset_lines) do 
    dataset_num_mentions = dataset_num_mentions + #lines_map
  end  
  print('\n===> ' .. banner .. '; num mentions = ' .. dataset_num_mentions)

  local time = sys.clock()

  confusion = optim.ConfusionMatrix(classes)  
  confusion:zero()
    
  xlua.progress(0, dataset_num_mentions)
  
  local num_true_positives = 0.0
  local grd_ent_freq_map = new_ent_freq_map()
  local correct_classified_ent_freq_map = new_ent_freq_map()
  local grd_ent_prior_map = new_ent_prior_map()
  local correct_classified_ent_prior_map = new_ent_prior_map()
  
  local num_mentions_without_gold_ent_in_candidates = 0
  local both_pem_ours = 0 -- num mentions solved both by argmax p(e|m) and our global model
  local only_pem_not_ours = 0 -- num mentions solved by argmax p(e|m), but not by our global model
  local only_ours_not_pem = 0 -- num mentions solved by our global model, but not by argmax p(e|m)
  local not_ours_not_pem = 0 -- num mentions not solved neither by our model nor by argmax p(e|m)
  
  local processed_docs = 0
  local processed_mentions = 0
  
  for doc_id, doc_lines in pairs(dataset_lines) do
    processed_docs = processed_docs + 1
    local num_mentions = #doc_lines
    processed_mentions = processed_mentions  + num_mentions
    local inputs = empty_minibatch_with_ids(num_mentions)
    local targets = torch.zeros(num_mentions)
    local mentions = {}
    for k = 1, num_mentions do
      local sample_line = doc_lines[k]
      local parts = split(sample_line, '\t')
      mentions[k] = parts[3]
      local target = process_one_line(sample_line, inputs, k, false)
      targets[k] = target      
    end
    inputs, targets = minibatch_to_correct_type(inputs, targets, false)
    
    -- NN forward pass:
    model, additional_local_submodels = get_model(num_mentions)
    model:evaluate()
    local preds = model:forward(inputs):float()      
    
    
    --------- Subnetworks used to print debug weights and scores :
    -- num_mentions x num_ctxt_vecs
    debug_softmax_word_weights = additional_local_submodels.model_debug_softmax_word_weights:forward(inputs):float() 
    -- num_mentions, max_num_cand:
    final_local_scores = additional_local_submodels.model_final_local:forward(inputs):float()      

    -- Process results:
    local all_ent_wikiids = get_cand_ent_wikiids(inputs)
    
    for k = 1, num_mentions do
      local pred = preds[k]  -- vector of size : max_num_cand
      assert (torch.norm(pred) ~= math.nan)

      -- Ignore unk entities (padding entities):
      local ent_wikiids = all_ent_wikiids[k] -- vector of size : max_num_cand
      for i = 1,max_num_cand do
        if ent_wikiids[i] == unk_ent_wikiid then
          pred[i] = -1e8 --> -infinity
        end
      end
      
      -- PRINT DEBUG SCORES: Show network weights and scores for entities with a valid gold label.
      if (targets[k] > 0) then
        local log_p_e_m = get_log_p_e_m(inputs)

        if (k == 1) then
          print('\n')
          print(blue('============================================'))
          print(blue('============ DOC : ' .. doc_id .. ' ================'))
          print(blue('============================================'))
        end

        -- Winning entity
        local _, argmax_idx = torch.max(pred, 1)
        local win_idx = argmax_idx[1] -- the actual number
        local ent_win = ent_wikiids[win_idx]
        local ent_win_name = get_ent_name_from_wikiid(ent_win)
        local ent_win_log_p_e_m = log_p_e_m[k][win_idx]
        local ent_win_local = final_local_scores[k][win_idx]
               
        -- Just some sanity check
        local best_pred, best_pred_idxs = topk(pred, 1)
        if (pred[best_pred_idxs[1]] ~= best_pred[1]) then
          print(pred)
        end
        
        assert(pred[best_pred_idxs[1]] == best_pred[1])
        assert(pred[best_pred_idxs[1]] == pred[win_idx])
        
        -- Grd trth entity
        local grd_idx = targets[k]
        local ent_grd = ent_wikiids[grd_idx]
        local ent_grd_name = get_ent_name_from_wikiid(ent_grd)
        local ent_grd_log_p_e_m = log_p_e_m[k][grd_idx]
        local ent_grd_local = final_local_scores[k][grd_idx]
                
        if win_idx ~= grd_idx then
          assert(ent_win ~= ent_grd)
          print('\n====> ' .. red('INCORRECT ANNOTATION') .. 
            ' : mention = ' .. skyblue(mentions[k]) ..
            ' ==> ENTITIES (OURS/GOLD): ' .. red(ent_win_name) .. ' <---> ' .. green(ent_grd_name))
        else
          assert(ent_win == ent_grd)
          local mention_str = 'mention = ' .. mentions[k]
          if math.exp(ent_grd_log_p_e_m) >= 0.99 then
            mention_str = yellow(mention_str)
          end          
          print('\n====> ' .. green('CORRECT ANNOTATION') .. 
            ' : ' .. mention_str ..
            ' ==> ENTITY: ' .. green(ent_grd_name))
        end

        print(
          'SCORES: global= ' .. nice_print_red_green(pred[win_idx], pred[grd_idx]) ..
          '; local(<e,ctxt>)= ' .. nice_print_red_green(ent_win_local, ent_grd_local) ..
          '; log p(e|m)= ' .. nice_print_red_green(ent_win_log_p_e_m, ent_grd_log_p_e_m)
        )
          
        -- Print top attended ctxt words and their attention weights:
        local str_words = '\nTop context words (sorted by attention weight, only non-zero weights - top R words): \n'
        local ctxt_word_ids = get_ctxt_word_ids(inputs) -- num_mentions x opt.ctxt_window
        local best_scores, best_word_idxs = topk(debug_softmax_word_weights[k], opt.ctxt_window)
        local seen_unk_w_id = false
        for wk = 1,opt.ctxt_window do 
          local w_idx_ctxt = best_word_idxs[wk]
          assert(w_idx_ctxt >= 1 and w_idx_ctxt <= opt.ctxt_window)
          local w_id = ctxt_word_ids[k][w_idx_ctxt]
          if w_id ~= unk_w_id or (not seen_unk_w_id) then
            if w_id == unk_w_id  then
              seen_unk_w_id = true
            end
            local w = get_word_from_id(w_id)
            local score = debug_softmax_word_weights[k][w_idx_ctxt]
            assert(score == best_scores[wk])
              
            if score > 0.001 then 
              str_words = str_words .. w .. '[' .. string.format("%.3f", score) .. ']; '
            end
          end
        end
        print(str_words)            
      end ----------------- Done printing scores and weights

      
      -- Count how many of the winning entities do not have a valid ent vector
      local _, argmax_idx = torch.max(pred, 1)
      if targets[k] > 0 and get_thid(ent_wikiids[argmax_idx[1]]) == unk_ent_thid then 
        print(pred)
        print(ent_wikiids)
        print('\n\n' .. red('!!!!Entity w/o vec: ' .. ent_wikiids[argmax_idx[1]] .. ' line = ' .. doc_lines[k]))
        os.exit()
      end

      -- Accumulate statistics about the annotations of our system.
      if (targets[k] > 0) then
        
        local log_p_e_m = get_log_p_e_m(inputs)
        
        local nn_is_good = true
        local pem_is_good = true
  
        -- Grd trth entity
        local grd_idx = targets[k]
        
        for j = 1,max_num_cand do
          if j ~= grd_idx and pred[grd_idx] < pred[j] then
            assert(ent_wikiids[j] ~= unk_ent_wikiid)
            nn_is_good = false
          end
          if j ~= grd_idx and log_p_e_m[k][grd_idx] < log_p_e_m[k][j] then
            assert(ent_wikiids[j] ~= unk_ent_wikiid)
            pem_is_good = false
          end          
        end
        
        if nn_is_good and pem_is_good then
          both_pem_ours = both_pem_ours + 1
        elseif nn_is_good then
          only_ours_not_pem = only_ours_not_pem + 1
        elseif pem_is_good then
          only_pem_not_ours = only_pem_not_ours + 1
        else
          not_ours_not_pem = not_ours_not_pem + 1
        end

        assert(ent_wikiids[targets[k]] ~= unk_ent_wikiid, ' Something is terribly wrong here ')
        confusion:add(pred, targets[k])

        -- Update number of true positives
        local _, argmax_idx = torch.max(pred, 1)
        local winning_entiid = ent_wikiids[argmax_idx[1]]
        local grd_entiid = ent_wikiids[targets[k]]
        
        local grd_ent_freq = get_ent_freq(grd_entiid)
        add_freq_to_ent_freq_map(grd_ent_freq_map, grd_ent_freq)
        
        local grd_ent_prior = math.exp(log_p_e_m[k][grd_idx])
        add_prior_to_ent_prior_map(grd_ent_prior_map, grd_ent_prior)
        if winning_entiid == grd_entiid then
          add_freq_to_ent_freq_map(correct_classified_ent_freq_map, grd_ent_freq)
          add_prior_to_ent_prior_map(correct_classified_ent_prior_map, grd_ent_prior)
          num_true_positives = num_true_positives + 1
        end

      else -- grd trth is not between the given set of candidates, so we cannot be right
        num_mentions_without_gold_ent_in_candidates = num_mentions_without_gold_ent_in_candidates + 1
        
        confusion:add(torch.zeros(max_num_cand), max_num_cand)
      end
    end
    -- disp progress
    xlua.progress(processed_mentions, dataset_num_mentions)    
  end -- done with this mini batch

  -- Now plotting results
  time = sys.clock() - time
  time = time / dataset_num_mentions
  print("==> time to test 1 sample = " .. (time*1000) .. 'ms')

  confusion:__tostring__()

  -- We refrain from solving mentions without at least one candidate
  local precision = 100.0 * num_true_positives / get_dataset_num_non_empty_candidates(dataset_lines)
  local recall = 100.0 * num_true_positives / dataset_num_mentions
  assert(math.abs(recall - confusion.totalValid * 100.0) < 0.01, 'Difference in recalls.')
  local f1 = 2.0 * precision * recall / (precision + recall)
  f1_scores[banner] = f1

  local f1_str = red(string.format("%.2f", f1) .. '%')
  if banner == 'aida-A' and f1 >= 90.20 then
    f1_str = green(string.format("%.2f", f1) .. '%')
  end
  
  print('==> '.. red(banner) .. ' ' .. banner .. ' ; EPOCH = ' .. epoch ..  
    ': Micro recall = ' .. string.format("%.2f", confusion.totalValid * 100.0) .. '%' .. 
    ' ; Micro F1 = ' .. f1_str)
  
  -- Lower learning rate if we got close to minimum
  if banner == 'aida-A' and f1 >= 90 then
    opt.lr = 1e-5
  end
  
  -- We slow down training a little bit if we passed the 90% F1 score threshold.
  -- And we start saving (good quality) models from now on.
  if banner == 'aida-A' then
    if f1 >= 90.0 then
      opt.save = true
    else
      opt.save = false
    end
  end
  
  print_ent_freq_maps_stats(correct_classified_ent_freq_map, grd_ent_freq_map)
  print_ent_prior_maps_stats(correct_classified_ent_prior_map, grd_ent_prior_map)
  
  print(' num_mentions_w/o_gold_ent_in_candidates = ' ..
    num_mentions_without_gold_ent_in_candidates .. 
    ' total num mentions in dataset = ' .. dataset_num_mentions)
  
  print(' percentage_mentions_w/o_gold_ent_in_candidates = ' ..
    string.format("%.2f", 100.0 * num_mentions_without_gold_ent_in_candidates / dataset_num_mentions) .. '%; ' ..
    ' percentage_mentions_solved : ' ..
    'both_pem_ours = ' .. string.format("%.2f", 100.0 * both_pem_ours / dataset_num_mentions) .. '%; ' ..
    ' only_pem_not_ours = ' .. string.format("%.2f", 100.0 * only_pem_not_ours / dataset_num_mentions) .. '%; ' ..
    ' only_ours_not_pem = ' .. string.format("%.2f", 100.0 * only_ours_not_pem / dataset_num_mentions) .. '%; ' ..
    ' not_ours_not_pem = ' .. string.format("%.2f", 100.0 * not_ours_not_pem / dataset_num_mentions) .. '%')
end


-------- main test function:
function test(epoch)
  if not epoch then
    epoch = 0
  end
  
  print('\n\n=====> TESTING <=== ')  
  f1_scores = {}
  for banner,dataset_file in pairs(datasets) do
    test_one(banner, f1_scores, epoch)
  end

  -- Plot accuracies
  if train_and_test then
    num_batch_train_between_plots = num_batches_per_epoch
    
    testAccLogger:add(f1_scores)
    styles = {}
    for banner,_ in pairs(f1_scores) do
      styles[banner] = '-'
    end
    testAccLogger:style(styles)
    testAccLogger:plot('F1', 'x ' .. num_batch_train_between_plots .. ' mini-batches')
  end
end