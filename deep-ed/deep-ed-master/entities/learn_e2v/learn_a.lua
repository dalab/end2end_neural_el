--  Training of entity embeddings. 

-- To run:
-- i) delete all _RLTD files
-- ii) th entities/relatedness/filter_wiki_canonical_words_RLTD.lua ; th entities/relatedness/filter_wiki_hyperlink_contexts_RLTD.lua
-- iii) th entities/learn_e2v/learn_a.lua  -root_data_dir /path/to/your/ed_data/files/

-- Training of entity vectors
require 'optim'
require 'torch'
require 'gnuplot'
require 'nn'
require 'xlua'

dofile 'utils/utils.lua'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Learning entity vectors')
cmd:text()
cmd:text('Options:')

---------------- runtime options:
cmd:option('-type', 'cudacudnn', 'Type: double | float | cuda | cudacudnn')

cmd:option('-root_data_dir', '', 'Root path of the data, $DATA_PATH.')

cmd:option('-optimization', 'ADAGRAD', 'Optimization method: RMSPROP | ADAGRAD | ADAM | SGD')

cmd:option('-lr', 0.3, 'Learning rate')

cmd:option('-batch_size', 500, 'Mini-batch size (1 = pure stochastic)')

cmd:option('-word_vecs', 'w2v', '300d word vectors type: glove | w2v')

cmd:option('-num_words_per_ent', 20, 'Num positive words sampled for the given entity at ' .. 
  'each iteration.')

cmd:option('-num_neg_words', 5, 'Num negative words sampled for each positive word.')

cmd:option('-unig_power', 0.6, 'Negative sampling unigram power (0.75 used in Word2Vec).')

cmd:option('-entities', 'RLTD',
  'Set of entities for which we train embeddings: 4EX (tiny, for debug) | ' .. 
  'RLTD (restricted set) | ALL (all Wiki entities, too big to fit on a single GPU)')

cmd:option('-init_vecs_title_words', true, 'whether the entity embeddings should be initialized with the average of ' .. 
  'title word embeddings. Helps to speed up convergence speed of entity embeddings learning.')

cmd:option('-loss', 'maxm', 'Loss function: nce (noise contrastive estimation) | ' .. 
  'neg (negative sampling) | is (importance sampling) | maxm (max-margin)')

cmd:option('-data', 'wiki-canonical-hyperlinks', 'Training data: wiki-canonical (only) | ' ..
  'wiki-canonical-hyperlinks')

-- Only when opt.data = wiki-canonical-hyperlinks
cmd:option('-num_passes_wiki_words', 200, 'Num passes (per entity) over Wiki canonical pages before ' ..
  'changing to using Wiki hyperlinks.')

cmd:option('-hyp_ctxt_len', 10, 'Left and right context window length for hyperlinks.')

cmd:option('-banner_header', '', 'Banner header')

cmd:text()
opt = cmd:parse(arg or {})

banner = '' .. opt.banner_header .. ';obj-' .. opt.loss .. ';' .. opt.data
if opt.data ~= 'wiki-canonical' then
  banner = banner .. ';hypCtxtL-' .. opt.hyp_ctxt_len
  banner = banner .. ';numWWpass-' .. opt.num_passes_wiki_words
end
banner = banner .. ';WperE-' .. opt.num_words_per_ent 
banner = banner .. ';' .. opt.word_vecs .. ';negW-' .. opt.num_neg_words
banner = banner .. ';ents-' .. opt.entities .. ';unigP-' .. opt.unig_power 
banner = banner .. ';bs-' .. opt.batch_size .. ';' .. opt.optimization  .. '-lr-' .. opt.lr

print('\n' .. blue('BANNER : ' .. banner))

print('\n===> RUN TYPE: ' .. opt.type) 

torch.setdefaulttensortype('torch.FloatTensor')
if string.find(opt.type, 'cuda') then
  print('==> switching to CUDA (GPU)')
  require 'cunn'
  require 'cutorch'
  require 'cudnn'
  cudnn.benchmark = true 
  cudnn.fastest = true
else
  print('==> running on CPU')
end

dofile 'utils/logger.lua'
dofile 'entities/relatedness/relatedness.lua'
dofile 'entities/ent_name2id_freq/ent_name_id.lua'
dofile 'words/load_w_freq_and_vecs.lua'
dofile 'words/w2v/w2v.lua'
dofile 'entities/learn_e2v/minibatch_a.lua'
dofile 'entities/learn_e2v/model_a.lua'    
dofile 'entities/learn_e2v/e2v_a.lua'
dofile 'entities/learn_e2v/batch_dataset_a.lua'

if opt.loss == 'neg' or opt.loss == 'nce' then
  criterion = nn.SoftMarginCriterion()
elseif opt.loss == 'maxm' then
  criterion = nn.MultiMarginCriterion(1, torch.ones(opt.num_neg_words), 0.1)
elseif opt.loss == 'is' then
  criterion = nn.CrossEntropyCriterion()
end

if string.find(opt.type, 'cuda') then
  criterion = criterion:cuda()
end


----------------------------------------------------------------------
if opt.optimization == 'ADAGRAD' then  -- See: http://cs231n.github.io/neural-networks-3/#update 
   dofile 'utils/optim/adagrad_mem.lua'
   optimMethod = adagrad_mem
   optimState = {
    learningRate = opt.lr 
  }
elseif opt.optimization == 'RMSPROP' then  -- See: cs231n.github.io/neural-networks-3/#update 
   dofile 'utils/optim/rmsprop_mem.lua'
   optimMethod = rmsprop_mem
   optimState = {
    learningRate = opt.lr
  }
elseif opt.optimization == 'SGD' then  -- See: http://cs231n.github.io/neural-networks-3/#update 
   optimState = {
      learningRate = opt.lr,
      learningRateDecay = 5e-7
   }
   optimMethod = optim.sgd 
elseif opt.optimization == 'ADAM' then  -- See: http://cs231n.github.io/neural-networks-3/#update 
   optimState = {
      learningRate = opt.lr,
   }
   optimMethod = optim.adam
else
   error('unknown optimization method')
end


----------------------------------------------------------------------
function train_ent_vecs()
  print('Training entity vectors w/ params: ' .. banner)
  
  -- Retrieve parameters and gradients:
  -- extracts and flattens all model's parameters into a 1-dim vector
  parameters,gradParameters = model:getParameters()
  gradParameters:zero()

  local processed_so_far = 0
  if opt.entities == 'ALL' then
    num_batches_per_epoch = 4000
  elseif opt.entities == 'RLTD' then
    num_batches_per_epoch = 2000
  elseif opt.entities == '4EX' then
    num_batches_per_epoch = 400
  end
  local test_every_num_epochs = 1 
  local save_every_num_epochs = 3
  
  -- epoch tracker
  epoch = 1

  -- Initial testing:
  geom_unit_tests() -- Show some examples 
  
  print('Training params: ' .. banner)

  -- do one epoch
  print('\n==> doing epoch on training data:')
  print("==> online epoch # " .. epoch .. ' [batch size = ' .. opt.batch_size .. ']')

  while true do
    
    local time = sys.clock()    
    print(green('\n===> TRAINING EPOCH #' .. epoch .. '; num batches ' ..  num_batches_per_epoch .. ' <==='))
    
    local avg_loss_before_opt_per_epoch = 0.0
    local avg_loss_after_opt_per_epoch = 0.0
    
    for batch_index = 1,num_batches_per_epoch  do      
      -- Read one mini-batch from one data_thread:
      inputs, targets = get_minibatch()      
      
      -- Move data to GPU:
      minibatch_to_correct_type(inputs)
      targets = correct_type(targets)
   
      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
          -- get new parameters
          if x ~= parameters then
            parameters:copy(x)
          end

          -- reset gradients
          gradParameters:zero()

          -- evaluate function for complete mini batch        
          local outputs = model:forward(inputs)
          assert(outputs:size(1) == opt.batch_size * opt.num_words_per_ent and 
                 outputs:size(2) == opt.num_neg_words)

          local f = criterion:forward(outputs, targets)
          
          -- estimate df/dW
          local df_do = criterion:backward(outputs, targets)
          local gradInput = model:backward(inputs, df_do)

          -- return f and df/dX
          return f,gradParameters
      end
 
      -- Debug info:
      local loss_before_opt = criterion:forward(model:forward(inputs), targets)
      avg_loss_before_opt_per_epoch = avg_loss_before_opt_per_epoch + loss_before_opt
      
      -- Optimize on current mini-batch
      optimMethod(feval, parameters, optimState)
      
      local loss_after_opt = criterion:forward(model:forward(inputs), targets)
      avg_loss_after_opt_per_epoch = avg_loss_after_opt_per_epoch + loss_after_opt
      if loss_after_opt > loss_before_opt then
        print(red('!!!!!! LOSS INCREASED: ' .. loss_before_opt .. ' --> ' .. loss_after_opt))
      end
      
      -- Display progress
      train_size = 17000000 ---------- 4 passes over the Wiki entity set
      processed_so_far = processed_so_far + opt.batch_size
      if processed_so_far > train_size then
        processed_so_far = processed_so_far - train_size
      end
      xlua.progress(processed_so_far, train_size)
    end
    
    avg_loss_before_opt_per_epoch = avg_loss_before_opt_per_epoch / num_batches_per_epoch
    avg_loss_after_opt_per_epoch = avg_loss_after_opt_per_epoch / num_batches_per_epoch
    print(yellow('\nAvg loss before opt = ' ..  avg_loss_before_opt_per_epoch .. 
        '; Avg loss after opt = ' .. avg_loss_after_opt_per_epoch))

    -- time taken
    time = sys.clock() - time
    time = time / (num_batches_per_epoch * opt.batch_size)
    print("==> time to learn 1 full entity = " .. (time*1000) .. 'ms')
        
    geom_unit_tests() -- Show some entity examples
    
    -- Various testing measures:
    if (epoch % test_every_num_epochs == 0) then
      if opt.entities ~= '4EX' then
        compute_relatedness_metrics(entity_similarity)
      end
    end

    -- Save model:
    if (epoch % save_every_num_epochs == 0) then
      print('==> saving model to ' .. opt.root_data_dir .. 'generated/ent_vecs/ent_vecs__ep_' .. epoch .. '.t7')
      torch.save(opt.root_data_dir .. 'generated/ent_vecs/ent_vecs__ep_' .. epoch .. '.t7', nn.Normalize(2):forward(lookup_ent_vecs.weight:float()))
    end
 
    print('Training params: ' .. banner)
    
    -- next epoch
    epoch = epoch + 1
  end
end

train_ent_vecs()
