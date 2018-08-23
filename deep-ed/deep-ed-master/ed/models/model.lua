dofile 'ed/models/SetConstantDiag.lua'
dofile 'ed/models/linear_layers.lua'
dofile 'ed/models/model_local.lua'
dofile 'ed/models/model_global.lua'

function get_model(num_mentions)
  local model_ctxt, additional_local_submodels =  local_model(num_mentions, A_linear, B_linear)
  local model = model_ctxt
  
  if opt.model == 'global' then
    model = global_model(num_mentions, model_ctxt, C_linear, f_network)
  else
    assert(opt.model == 'local')
  end
  
  return model, additional_local_submodels 
end
