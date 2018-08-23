-- Definition of the neural network used for global (joint) ED. Section 5 of our paper. 
-- It unrolls a fixed number of LBP iterations allowing training of the CRF potentials using backprop.
-- To run a simple unit test that checks the forward and backward passes, just run :
--    th ed/models/model_global.lua

if not opt then -- unit tests
  dofile 'ed/models/model_local.lua'
  dofile 'ed/models/SetConstantDiag.lua'
  dofile 'ed/models/linear_layers.lua' 
  
  opt.lbp_iter = 10
  opt.lbp_damp = 0.5
  opt.model = 'global'
end


function global_model(num_mentions, model_ctxt, param_C_linear, param_f_network)
  
  assert(num_mentions)
  assert(model_ctxt)
  assert(param_C_linear)
  assert(param_f_network)
  
  local unary_plus_pairwise = nn.Sequential()
    :add(nn.ParallelTable()
      :add(nn.Sequential() -- Pairwise scores s_{ij}(y_i, y_j)
        :add(nn.View(num_mentions * max_num_cand, ent_vecs_size)) -- e_vecs
        :add(nn.ConcatTable()
          :add(nn.Identity())
          :add(param_C_linear)
        )
        :add(nn.MM(false, true)) -- s_{ij}(y_i, y_j) is s[i][y_i][j][y_j] = <e_{y_i}, C * e_{y_j}>
        :add(nn.View(num_mentions, max_num_cand, num_mentions, max_num_cand))
        :add(nn.MulConstant(2.0 / num_mentions, true))
      )
      :add(nn.Sequential() -- Unary scores s_j(y_j)
        :add(nn.Replicate(num_mentions * max_num_cand, 1))
        :add(nn.Reshape(num_mentions, max_num_cand, num_mentions, max_num_cand))
      )
    )
    :add(nn.CAddTable()) -- q[i][y_i][j][y_j] = s_j(y_j) + s_{ij}(y_i, y_j): num_mentions x max_num_cand x num_mentions x max_num_cand
    
  
  -- Input is q[i][y_i][j] : num_mentions x max_num_cand x num_mentions
  local messages_one_round = nn.ConcatTable()
    :add(nn.SelectTable(1)) -- 1. unary_plus_pairwise : num_mentions, max_num_cand, num_mentions, max_num_cand
    :add(nn.Sequential()
      :add(nn.ConcatTable()
        :add(nn.Sequential()
          :add(nn.SelectTable(2)) -- old messages
          :add(nn.Exp())
          :add(nn.MulConstant(1.0 - opt.lbp_damp, false))
        )
        :add(nn.Sequential()
          :add(nn.ParallelTable()
            :add(nn.Identity())  -- unary plus pairwise
            :add(nn.Sequential() -- old messages: num_mentions, max_num_cand, num_mentions
              :add(nn.Sum(3)) -- g[i][y_i] := \sum_{k != i} q[i][y_i][k]
              :add(nn.Replicate(num_mentions * max_num_cand, 1))
              :add(nn.Reshape(num_mentions, max_num_cand, num_mentions, max_num_cand))
            )
          )
          :add(nn.CAddTable()) --  s_{j}(y_j) + s_{ij}(y_i, y_j) + g[j][y_j] : num_mentions, max_num_cand, num_mentions, max_num_cand
          :add(nn.Max(4)) -- unnorm_q[i][y_i][j] : num_mentions x max_num_cand x num_mentions
          :add(nn.Transpose({2,3})) 
          :add(nn.View(num_mentions * num_mentions, max_num_cand))
          :add(nn.LogSoftMax())  -- normalization: \sum_{y_i} exp(q[i][y_i][j]) = 1
          :add(nn.View(num_mentions, num_mentions, max_num_cand))
          :add(nn.Transpose({2,3})) 
          :add(nn.Transpose({1,2}))
          :add(nn.SetConstantDiag(0, true)) -- we make q[i][y_i][i] = 0, \forall i and y_i
          :add(nn.Transpose({1,2})) 
          :add(nn.Exp())
          :add(nn.MulConstant(opt.lbp_damp, false))
        )
      )
      :add(nn.CAddTable()) -- 2. messages for next round: num_mentions, max_num_cand, num_mentions
      :add(nn.Log())
    )


  local messages_all_rounds = nn.Sequential()
  messages_all_rounds:add(nn.Identity())
  for i = 1, opt.lbp_iter do
    messages_all_rounds:add(messages_one_round:clone('weight','bias','gradWeight','gradBias'))
  end

  local model_gl = nn.Sequential()
    :add(nn.ConcatTable()
      :add(nn.Sequential()
        :add(nn.SelectTable(2))
        :add(nn.SelectTable(2))  -- e_vecs : num_mentions, max_num_cand, ent_vecs_size
      )
      :add(model_ctxt) -- unary scores : num_mentions, max_num_cand
    )
    :add(nn.ConcatTable()
      :add(nn.Sequential()
        :add(unary_plus_pairwise)
        :add(nn.ConcatTable()
          :add(nn.Identity())
          :add(nn.Sequential()
            :add(nn.Max(4)) 
            :add(nn.MulConstant(0, false)) -- first_round_zero_messages
          )
        )
        :add(messages_all_rounds)
        :add(nn.SelectTable(2))
        :add(nn.Sum(3)) -- \sum_{j} msgs[i][y_i]: num_mentions x max_num_cand
      )
      :add(nn.SelectTable(2)) -- unary scores : num_mentions x max_num_cand
    )
    :add(nn.CAddTable()) 
    :add(nn.LogSoftMax()) -- belief[i][y_i] (lbp marginals in log scale)

    
  -- Combine lbp marginals with log p(e|m) using the simple f neural network
  local pem_layer = nn.SelectTable(3)
  model_gl = nn.Sequential()
    :add(nn.ConcatTable()
      :add(nn.Sequential()
        :add(model_gl)
        :add(nn.View(num_mentions * max_num_cand, 1))
      )
      :add(nn.Sequential()
        :add(pem_layer)
        :add(nn.View(num_mentions * max_num_cand, 1))
      )        
    )
    :add(nn.JoinTable(2))
    :add(param_f_network)
    :add(nn.View(num_mentions, max_num_cand))

  ------- Cuda conversions:
  if string.find(opt.type, 'cuda') then
    model_gl = model_gl:cuda()
  end

  return model_gl
end


--- Unit tests
if unit_tests_now then
  print('\n Global network model unit tests:')
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
  inputs[3] = torch.log(torch.rand(num_mentions, max_num_cand))

  local model_ctxt, _ = local_model(num_mentions, A_linear, B_linear, opt)

  local model_gl = global_model(num_mentions, model_ctxt, C_linear, f_network)
  local outputs = model_gl:forward(inputs)
  print(outputs)
  print('MIN: ' .. torch.min(outputs) .. ' MAX: ' ..  torch.max(outputs))
  assert(outputs:size(1) == num_mentions and outputs:size(2) == max_num_cand)
  print('Global FWD success!')

  model_gl:backward(inputs, torch.randn(num_mentions, max_num_cand))
  print('Global BKWD success!')
  
  parameters,gradParameters = model_gl:getParameters()
  print(parameters:size())
  print(gradParameters:size())
end