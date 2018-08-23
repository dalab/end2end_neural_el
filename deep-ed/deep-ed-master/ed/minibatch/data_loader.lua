-- Data loader for training of ED models.

train_file = opt.root_data_dir .. 'generated/test_train_data/aida_train.csv'
it_train, _ = io.open(train_file)

print('==> Loading training data with option ' .. opt.store_train_data)
local function one_doc_to_minibatch(doc_lines)
  -- Create empty mini batch:
  local num_mentions = #doc_lines
  assert(num_mentions > 0)

  local inputs = empty_minibatch_with_ids(num_mentions)
  local targets = torch.zeros(num_mentions)

  -- Fill in each example:
  for i = 1, num_mentions do
    local target = process_one_line(doc_lines[i], inputs, i, true)
    targets[i] = target
    assert(target >= 1 and target == targets[i])
  end

  return inputs, targets  
end

if opt.store_train_data == 'RAM' then
  all_docs_inputs = tds.Hash()
  all_docs_targets = tds.Hash()
  doc2id = tds.Hash()
  id2doc = tds.Hash()

  local cur_doc_lines = tds.Hash()
  local prev_doc_id = nil

  local line = it_train:read()
  while line do
    local parts = split(line, '\t')
    local doc_name = parts[1]
    if not doc2id[doc_name] then
      if prev_doc_id then
        local inputs, targets = one_doc_to_minibatch(cur_doc_lines)
        all_docs_inputs[prev_doc_id] = minibatch_table2tds(inputs)
        all_docs_targets[prev_doc_id] = targets
      end
      local cur_docid = 1 + #doc2id
      id2doc[cur_docid] = doc_name
      doc2id[doc_name] = cur_docid
      cur_doc_lines = tds.Hash()
      prev_doc_id = cur_docid
    end
    cur_doc_lines[1 + #cur_doc_lines] = line
    line = it_train:read()
  end
  if prev_doc_id then
    local inputs, targets = one_doc_to_minibatch(cur_doc_lines)
    all_docs_inputs[prev_doc_id] = minibatch_table2tds(inputs)
    all_docs_targets[prev_doc_id] = targets
  end  
  assert(#doc2id == #all_docs_inputs, #doc2id .. ' ' .. #all_docs_inputs)

else
  all_doc_lines = tds.Hash()
  doc2id = tds.Hash()
  id2doc = tds.Hash()

  local line = it_train:read()
  while line do
    local parts = split(line, '\t')
    local doc_name = parts[1]
    if not doc2id[doc_name] then
      local cur_docid = 1 + #doc2id
      id2doc[cur_docid] = doc_name
      doc2id[doc_name] = cur_docid
      all_doc_lines[cur_docid] = tds.Hash()
    end
    all_doc_lines[doc2id[doc_name]][1 + #all_doc_lines[doc2id[doc_name]]] = line
    line = it_train:read()
  end
  assert(#doc2id == #all_doc_lines)
end


get_minibatch = function()
  -- Create empty mini batch:
  local inputs = nil
  local targets = nil

  if opt.store_train_data == 'RAM' then
    local random_docid = math.random(#id2doc)
    inputs = minibatch_tds2table(all_docs_inputs[random_docid])
    targets = all_docs_targets[random_docid]
  else
    local doc_lines = all_doc_lines[math.random(#id2doc)]
    inputs, targets = one_doc_to_minibatch(doc_lines)
  end

  -- Move data to GPU:
  inputs, targets = minibatch_to_correct_type(inputs, targets, true)
  targets = correct_type(targets)

  return inputs, targets
end

print('    Done loading training data.')
