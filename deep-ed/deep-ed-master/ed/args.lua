-- We add params abbreviations to the abbv map if they are important hyperparameters
-- used to differentiate between different methods.
abbv = {}

cmd = torch.CmdLine()
cmd:text()
cmd:text('Deep Joint Entity Disambiguation w/ Local Neural Attention')
cmd:text('Command line options:')

---------------- runtime options:
cmd:option('-root_data_dir', '', 'Root path of the data, $DATA_PATH.')

cmd:option('-unit_tests', false, 'Run unit tests or not')

---------------- CUDA:
cmd:option('-type', 'cudacudnn', 'Type: cuda | float | cudacudnn')

---------------- train data:
cmd:option('-store_train_data', 'RAM', 'Where to read the training data from: RAM (tensors) | DISK (text, parsed all the time)')

---------------- loss:
cmd:option('-loss', 'max-margin', 'Loss: nll | max-margin')
abbv['-loss'] = ''

---------------- optimization:
cmd:option('-opt', 'ADAM', 'Optimization method: SGD | ADADELTA | ADAGRAD | ADAM')
abbv['-opt'] = ''

cmd:option('-lr', 1e-4, 'Learning rate. Will be divided by 10 after validation F1 >= 90%.')
abbv['-lr'] = 'lr'

cmd:option('-batch_size', 1, 'Batch size in terms of number of documents.')
abbv['-batch_size'] = 'bs'

---------------- word vectors
cmd:option('-word_vecs', 'w2v', 'Word vectors type: glove | w2v (Word2Vec)')
abbv['-word_vecs'] = ''

---------------- entity vectors
cmd:option('-entities', 'RLTD', 'Which entity vectors to use, either just those that appear as candidates in all datasets, or all. All is impractical when storing on GPU. RLTD | ALL')

cmd:option('-ent_vecs_filename', 'ent_vecs__ep_228.t7', 'File name containing entity vectors generated with entities/learn_e2v/learn_a.lua.')
  
---------------- context
cmd:option('-ctxt_window', 100, 'Number of context words at the left plus right of each mention')
abbv['-ctxt_window'] = 'ctxtW'

cmd:option('-R', 25, 'Hard attention threshold: top R context words are kept, the rest are discarded.')
abbv['-R'] = 'R'

---------------- model
cmd:option('-model', 'global', 'ED model: local | global')

cmd:option('-nn_pem_interm_size', 100, 'Number of hidden units in the f function described in Section 4 - Local score combination.')
abbv['-nn_pem_interm_size'] = 'nnPEMintermS'

-------------- model regularization:
cmd:option('-mat_reg_norm', 1, 'Maximum norm of columns of matrices of the f network.')
abbv['-mat_reg_norm'] = 'matRegNorm'

---------------- global model parameters
cmd:option('-lbp_iter', 10, 'Number iterations of LBP hard-coded in a NN. Referred as T in the paper.')
abbv['-lbp_iter'] = 'lbpIt'

cmd:option('-lbp_damp', 0.5, 'Damping factor for LBP')
abbv['-lbp_damp'] = 'lbpDamp'

----------------- reranking of candidates
cmd:option('-num_cand_before_rerank', 30, '')
abbv['-num_cand_before_rerank'] = 'numERerank'

cmd:option('-keep_p_e_m', 4, '')
abbv['-keep_p_e_m'] = 'keepPEM'

cmd:option('-keep_e_ctxt', 3, '')
abbv['-keep_e_ctxt'] = 'keepEC'

----------------- coreference:
cmd:option('-coref', true, 'Coreference heuristic to match persons names.')
abbv['-coref'] = 'coref'

------------------ test one model with saved pretrained parameters
cmd:option('-test_one_model_file', '', 'Saved pretrained model filename from folder $DATA_PATH/generated/ed_models/.')

------------------ banner:
cmd:option('-banner_header', '', ' Banner header to be used for plotting')

cmd:text()
opt = cmd:parse(arg or {})

-- Whether to save the current ED model or not during training.
-- It will become true after the model gets > 90% F1 score on validation set (see test.lua).
opt.save = false 

-- Creates a nice banner from the command line arguments
function get_banner(arg, abbv)
  local num_args = table_len(arg)
  local banner = opt.banner_header 

  if opt.model == 'global' then
    banner = banner .. 'GLB'
  else
    banner = banner .. 'LCL'
  end
  
  for i = 1,num_args do
    if abbv[arg[i]] then
      banner = banner .. '|' .. abbv[arg[i]] .. '=' .. tostring(opt[arg[i]:sub(2)])
    end
  end
  return banner
end


function serialize_params(arg)
  local num_args = table_len(arg)
  local str = opt.banner_header 
  if opt.banner_header:len() > 0 then
    str = str .. '|'
  end
  
  str = str .. 'model=' .. opt.model

  for i = 1,num_args do
    if abbv[arg[i]] then
      str = str .. '|' .. arg[i]:sub(2) .. '=' .. tostring(opt[arg[i]:sub(2)])
    end
  end
  return str
end

banner = get_banner(arg, abbv)

params_serialized = serialize_params(arg)
print('PARAMS SERIALIZED: ' .. params_serialized)
print('BANNER : ' .. banner .. '\n')

assert(params_serialized:len() < 255, 'Parameters string length should be < 255.')


function extract_args_from_model_title(title)
  local x,y = title:find('model=')
  local parts = split(title:sub(x), '|')
  for _,part in pairs(parts) do
    if string.find(part, '=') then
      local components = split(part, '=')
      opt[components[1]] = components[2]
      if tonumber(components[2]) then
        opt[components[1]] = tonumber(components[2])
      end
      if components[2] == 'true' then
        opt[components[1]] = true
      end
      if components[2] == 'false' then
        opt[components[1]] = false
      end
    end
  end
end
