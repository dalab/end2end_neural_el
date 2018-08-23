-- Runs our trivial coreference resolution method and outputs the new set of 
-- entity candidates. Used for debugging the coreference resolution method.

if not opt then
  cmd = torch.CmdLine()
  cmd:option('-root_data_dir', '', 'Root path of the data, $DATA_PATH.')
  cmd:text()
  opt = cmd:parse(arg or {})
  assert(opt.root_data_dir ~= '', 'Specify a valid root_data_dir path argument.')  
end


require 'torch'
dofile 'utils/utils.lua'

tds = tds or require 'tds'

dofile 'entities/ent_name2id_freq/ent_name_id.lua'
dofile 'ed/test/coref.lua'

file = opt.root_data_dir .. 'generated/test_train_data/aida_testB.csv'

opt = {}
opt.coref = true

  it, _ = io.open(file)
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
  build_coreference_dataset(all_doc_lines, 'aida-B')
