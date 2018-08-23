dofile 'utils/utils.lua'

if opt.entities == 'ALL' then
  wiki_words_train_file = opt.root_data_dir .. 'generated/wiki_canonical_words.txt'
  wiki_hyp_train_file = opt.root_data_dir .. 'generated/wiki_hyperlink_contexts.csv'
else
  wiki_words_train_file = opt.root_data_dir .. 'generated/wiki_canonical_words_RLTD.txt'
  wiki_hyp_train_file = opt.root_data_dir .. 'generated/wiki_hyperlink_contexts_RLTD.csv'
end

wiki_words_it, _ = io.open(wiki_words_train_file)
wiki_hyp_it, _ = io.open(wiki_hyp_train_file)

assert(opt.num_passes_wiki_words)
local train_data_source = 'wiki-canonical'
local num_passes_wiki_words = 1

local function read_one_line()
  if train_data_source == 'wiki-canonical' then
    line = wiki_words_it:read()
  else
    assert(train_data_source == 'wiki-canonical-hyperlinks')
    line = wiki_hyp_it:read()
  end
  if (not line) then
    if num_passes_wiki_words == opt.num_passes_wiki_words then
      train_data_source = 'wiki-canonical-hyperlinks'
      print('\n\n' .. 'Start training on Wiki Hyperlinks' .. '\n\n')
    end
    print('Training file is done. Num passes = ' .. num_passes_wiki_words .. '. Reopening.')
    num_passes_wiki_words = num_passes_wiki_words + 1
    if train_data_source == 'wiki-canonical' then
      wiki_words_it, _ = io.open(wiki_words_train_file)
      line = wiki_words_it:read()
    else
      wiki_hyp_it, _ = io.open(wiki_hyp_train_file)
      line = wiki_hyp_it:read()
    end
  end 
  return line
end

local line = nil

local function patch_of_lines(num)
  local lines = {}
  local cnt = 0
  assert(num > 0)
  
  while cnt < num do
    line = read_one_line()
    cnt = cnt + 1
    table.insert(lines, line)
  end

  assert(table_len(lines) == num)
  return lines
end


function get_minibatch()
  -- Create empty mini batch:
  local lines = patch_of_lines(opt.batch_size)
  local inputs = empty_minibatch()      
  local targets = correct_type(torch.ones(opt.batch_size, opt.num_words_per_ent))

  -- Fill in each example:
  for i = 1,opt.batch_size do
    local sample_line = lines[i]  -- load new example line
    local target = process_one_line(sample_line, inputs, i)
    targets[i]:copy(target)
  end
  
  --- Minibatch post processing:
  postprocess_minibatch(inputs, targets)
  targets = targets:view(opt.batch_size * opt.num_words_per_ent)          

  -- Special target for the NEG and NCE losses
  if opt.loss == 'neg' or opt.loss == 'nce' then
    nce_targets = torch.ones(opt.batch_size * opt.num_words_per_ent, opt.num_neg_words):mul(-1)
    for j = 1,opt.batch_size * opt.num_words_per_ent do
      nce_targets[j][targets[j]] = 1
    end
    targets = nce_targets
  end

  return inputs, targets
end
