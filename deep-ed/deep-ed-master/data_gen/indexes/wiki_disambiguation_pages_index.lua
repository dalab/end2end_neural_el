-- Loads the link disambiguation index from Wikipedia

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

print('==> Loading disambiguation index')
it, _ = io.open(opt.root_data_dir .. 'basic_data/wiki_disambiguation_pages.txt')
line = it:read()

wiki_disambiguation_index = tds.Hash()
while (line) do
  parts = split(line, "\t")
  assert(tonumber(parts[1]))
  wiki_disambiguation_index[tonumber(parts[1])] = 1
  line = it:read()
end

assert(wiki_disambiguation_index[579])
assert(wiki_disambiguation_index[41535072])

print('    Done loading disambiguation index')
