-- Definition of the local neural network with attention used for local (independent per each mention) ED. 
-- Section 4 of our paper.
-- To run a simple unit test that checks the forward and backward passes, just run :
--    th ed/models/model_local.lua

if not opt then -- unit tests
  unit_tests_now = true
  dofile 'utils/utils.lua'
  require 'nn'
  opt = {type = 'double', ctxt_window = 100, R = 25, model = 'local', nn_pem_interm_size = 100}
  
  word_vecs_size = 300
  ent_vecs_size = 300
  max_num_cand = 6
  unk_ent_wikiid = 1
  unk_ent_thid = 1
  unk_w_id = 1
  dofile 'ed/models/linear_layers.lua'  
  word_lookup_table = nn.LookupTable(5, ent_vecs_size)
  ent_lookup_table = nn.LookupTable(5, ent_vecs_size)
else
  word_lookup_table = nn.LookupTable(w2vutils.M:size(1), ent_vecs_size)
  word_lookup_table.weight = w2vutils.M

  ent_lookup_table = nn.LookupTable(e2vutils.lookup:size(1), ent_vecs_size)
  ent_lookup_table.weight = e2vutils.lookup

  if string.find(opt.type, 'cuda') then
    word_lookup_table = word_lookup_table:cuda()
    ent_lookup_table = ent_lookup_table:cuda()
  end
end

assert(word_vecs_size == 300 and ent_vecs_size == 300)


----------------- Define the model
function local_model(num_mentions, param_A_linear, param_B_linear)
  
  assert(num_mentions)
  assert(param_A_linear)
  assert(param_B_linear)
  
  model = nn.Sequential()
  
  ctxt_embed_and_ent_lookup = nn.ConcatTable()
    :add(nn.Sequential()
      :add(nn.SelectTable(1))
      :add(nn.SelectTable(2))   -- 1 : Context words W : num_mentions x opt.ctxt_window x ent_vecs_size
    )
    :add(nn.Sequential()
      :add(nn.SelectTable(2))
      :add(nn.SelectTable(2))     -- 2 : Candidate entity vectors E : num_mentions x max_num_cand x ent_vecs_size
    )
    :add(nn.SelectTable(3))     -- 3 : log p(e|m) : num_mentions x max_num_cand
  
  model:add(ctxt_embed_and_ent_lookup)

  
  local mem_weights_p_2 = nn.ConcatTable()
    :add(nn.Identity()) -- 1 : {W, E, logp(e|m)}
    :add(nn.Sequential()
      :add(nn.ConcatTable()
        :add(nn.Sequential()
          :add(nn.SelectTable(2))
          :add(nn.View(num_mentions * max_num_cand, ent_vecs_size)) -- E : num_mentions x max_num_cand x ent_vecs_size
          :add(param_A_linear)
          :add(nn.View(num_mentions, max_num_cand, ent_vecs_size)) -- (A*) E 
        )
        :add(nn.SelectTable(1)) --- W : num_mentions x opt.ctxt_window x ent_vecs_size
      )
      :add(nn.MM(false, true)) -- 2 : E^t * A * W : num_mentions x max_num_cand x opt.ctxt_window
    )
  model:add(mem_weights_p_2)


  local mem_weights_p_3 = nn.ConcatTable()
    :add(nn.SelectTable(1)) -- 1 : {W, E, logp(e|m)}
    :add(nn.Sequential()
      :add(nn.SelectTable(2))
      :add(nn.Max(2)) --- 2 : max(word-entity scores) : num_mentions x opt.ctxt_window
    ) 

  model:add(mem_weights_p_3)


  local mem_weights_p_4 = nn.ConcatTable()
    :add(nn.SelectTable(1)) -- 1 : {W, E, logp(e|m)}
    :add(nn.Sequential()
      :add(nn.SelectTable(2))
      -- keep only top K scored words
      :add(nn.ConcatTable()
        :add(nn.Identity()) -- all w-e scores
        :add(nn.Sequential()  
          :add(nn.View(num_mentions, opt.ctxt_window, 1))
          :add(nn.TemporalDynamicKMaxPooling(opt.R))
          :add(nn.Min(2)) -- k-th largest w-e score
          :add(nn.View(num_mentions))
          :add(nn.Replicate(opt.ctxt_window, 2))
        )
      )
      :add(nn.ConcatTable() -- top k w-e scores (the rest are set to -infty)
        :add(nn.SelectTable(2)) -- k-th largest w-e score that we substract and then add again back after nn.Threshold
        :add(nn.Sequential()
          :add(nn.CSubTable())
          :add(nn.Threshold(0, -50, true))
        )
      )
      :add(nn.CAddTable())
      :add(nn.SoftMax()) -- 2 : sigma (attention weights normalized): num_mentions x opt.ctxt_window
      :add(nn.View(num_mentions, opt.ctxt_window, 1))
    )      

  model:add(mem_weights_p_4)


  local ctxt_full_embeddings = nn.ConcatTable()
    :add(nn.SelectTable(1)) -- 1 : {W, E, logp(e|m)}
    :add(nn.Sequential()
      :add(nn.ConcatTable()
        :add(nn.Sequential()
          :add(nn.SelectTable(1))
          :add(nn.SelectTable(1)) -- W
        )
        :add(nn.SelectTable(2)) -- sigma
      )
      :add(nn.MM(true, false)) 
      :add(nn.View(num_mentions, ent_vecs_size)) -- 2 : ctxt embedding = (W * B)^\top * sigma  : num_mentions x ent_vecs_size
    )

  model:add(ctxt_full_embeddings) 


  local entity_context_sim_scores = nn.ConcatTable()
    :add(nn.SelectTable(1)) -- 1 : {W, E, logp(e|m)}
    :add(nn.Sequential()
      :add(nn.ConcatTable()
        :add(nn.Sequential()
          :add(nn.SelectTable(1))
          :add(nn.SelectTable(2)) -- E
        )
        :add(nn.Sequential()
          :add(nn.SelectTable(2))
          :add(param_B_linear)
          :add(nn.View(num_mentions, ent_vecs_size, 1))  -- context vectors
        )
      )
      :add(nn.MM()) --> 2. context * E^T
      :add(nn.View(num_mentions, max_num_cand))
    )
  
  model:add(entity_context_sim_scores)

  if opt.model == 'local' then
    model = nn.Sequential()
      :add(nn.ConcatTable()
        :add(nn.Sequential()
          :add(model)
          :add(nn.SelectTable(2))  -- context - entity similarity scores
          :add(nn.View(num_mentions * max_num_cand, 1))
        )
        :add(nn.Sequential()
          :add(nn.SelectTable(3))  -- log p(e|m) scores
          :add(nn.View(num_mentions * max_num_cand, 1))
        )        
      )
      :add(nn.JoinTable(2))
      :add(f_network)
      :add(nn.View(num_mentions, max_num_cand))
  else
    model:add(nn.SelectTable(2)) -- context - entity similarity scores
  end
  
  
  ------------- Visualizing weights: 

  -- sigma (attention weights normalized): num_mentions x opt.ctxt_window
  local model_debug_softmax_word_weights = nn.Sequential()
    :add(ctxt_embed_and_ent_lookup)
    :add(mem_weights_p_2)
    :add(mem_weights_p_3)
    :add(mem_weights_p_4)
    :add(nn.SelectTable(2))
    :add(nn.View(num_mentions, opt.ctxt_window))
    
  ------- Cuda conversions:
  if string.find(opt.type, 'cuda') then
    model = model:cuda()
    model_debug_softmax_word_weights = model_debug_softmax_word_weights:cuda()
  end

  local additional_local_submodels  = {
    model_final_local = model,
    model_debug_softmax_word_weights = model_debug_softmax_word_weights, 
  }
  
  return model, additional_local_submodels 
end


--- Unit tests
if unit_tests_now then
  print('Network model unit tests:')
  local num_mentions = 13
  
  local inputs = {}
  
  -- ctxt_w_vecs
  inputs[1] = {}
  inputs[1][1] = torch.ones(num_mentions, opt.ctxt_window):int():mul(unk_w_id)
  inputs[1][2] = torch.randn(num_mentions, opt.ctxt_window, ent_vecs_size)
  -- e_vecs
  inputs[2] = {}
  inputs[2][1] = torch.ones(num_mentions, max_num_cand):int():mul(unk_ent_thid)
  inputs[2][2] = torch.randn(num_mentions, max_num_cand, ent_vecs_size)
  -- p(e|m)
  inputs[3] = torch.zeros(num_mentions, max_num_cand) 
  
  local model, additional_local_submodels = local_model(num_mentions, A_linear, B_linear, opt)
  
  print(additional_local_submodels.model_debug_softmax_word_weights:forward(inputs):size())
  
  local outputs = model:forward(inputs)
  assert(outputs:size(1) == num_mentions and outputs:size(2) == max_num_cand)
  print('FWD success!')

  model:backward(inputs, torch.randn(num_mentions, max_num_cand))
  print('BKWD success!')
    
  parameters,gradParameters = model:getParameters()
  print(parameters:size())
  print(gradParameters:size())  
end
