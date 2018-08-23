if not opt then
  cmd = torch.CmdLine()
  cmd:option('-root_data_dir', '', 'Root path of the data, $DATA_PATH.')
  cmd:option('-ent_vecs_filename', 'ent_vecs__ep_228.t7', 'File name containing entity vectors generated with entities/learn_e2v/learn_a.lua.')
  cmd:text()
  opt = cmd:parse(arg or {})
  assert(opt.root_data_dir ~= '', 'Specify a valid root_data_dir path argument.')  
end


require 'optim'
require 'torch'
require 'gnuplot'
require 'nn'
require 'xlua'

dofile 'utils/utils.lua'
dofile 'ed/args.lua'

tds = require 'tds'


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

dofile 'utils/logger.lua'
dofile 'entities/relatedness/relatedness.lua'
dofile 'entities/ent_name2id_freq/ent_name_id.lua'
dofile 'entities/ent_name2id_freq/e_freq_index.lua'
dofile 'words/load_w_freq_and_vecs.lua'
dofile 'entities/pretrained_e2v/e2v.lua'