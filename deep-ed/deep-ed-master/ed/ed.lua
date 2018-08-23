require 'optim'
require 'torch'
require 'gnuplot'
require 'nn'
require 'xlua'

tds = tds or require 'tds'
dofile 'utils/utils.lua'

print('\n' .. green('==========> TRAINING of ED NEURAL MODELS <==========') .. '\n')
  
dofile 'ed/args.lua'

print('===> RUN TYPE (CPU/GPU): ' .. opt.type)
    
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
dofile 'entities/ent_name2id_freq/e_freq_index.lua'
dofile 'words/load_w_freq_and_vecs.lua' -- w ids
dofile 'words/w2v/w2v.lua'
dofile 'entities/pretrained_e2v/e2v.lua'
dofile 'ed/minibatch/build_minibatch.lua'
dofile 'ed/minibatch/data_loader.lua'
dofile 'ed/models/model.lua'
dofile 'ed/loss.lua'
dofile 'ed/train.lua'
dofile 'ed/test/test.lua'

geom_unit_tests() -- Show some entity examples
compute_relatedness_metrics(entity_similarity)  -- UNCOMMENT

train_and_test()
