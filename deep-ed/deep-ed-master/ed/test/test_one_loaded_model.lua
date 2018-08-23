-- Test one single ED model trained using ed/ed.lua

-- Run: CUDA_VISIBLE_DEVICES=0 th ed/test/test_one_loaded_model.lua -root_data_dir $DATA_PATH -model global -ent_vecs_filename $ENTITY_VECS  -test_one_model_file $ED_MODEL_FILENAME
require 'optim'
require 'torch'
require 'gnuplot'
require 'nn'
require 'xlua'
tds = tds or require 'tds'

dofile 'utils/utils.lua'
print('\n' .. green('==========> Test a pre-trained ED neural model <==========') .. '\n')

dofile 'ed/args.lua'

print('===> RUN TYPE: ' .. opt.type)
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

extract_args_from_model_title(opt.test_one_model_file)

dofile 'utils/logger.lua'
dofile 'entities/relatedness/relatedness.lua'
dofile 'entities/ent_name2id_freq/ent_name_id.lua'
dofile 'entities/ent_name2id_freq/e_freq_index.lua'
dofile 'words/load_w_freq_and_vecs.lua'
dofile 'words/w2v/w2v.lua'
dofile 'entities/pretrained_e2v/e2v.lua'
dofile 'ed/minibatch/build_minibatch.lua'
dofile 'ed/models/model.lua'
dofile 'ed/test/test.lua'

local saved_linears = torch.load(opt.root_data_dir .. 'generated/ed_models/' .. opt.test_one_model_file)
unpack_saveable_weights(saved_linears)

test()
