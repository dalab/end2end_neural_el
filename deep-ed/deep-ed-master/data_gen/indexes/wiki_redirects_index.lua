-- Loads the link redirect index from Wikipedia

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

print('==> Loading redirects index')
it, _ = io.open(opt.root_data_dir .. 'basic_data/wiki_redirects.txt')
line = it:read()

local wiki_redirects_index = tds.Hash()
while (line) do
  parts = split(line, "\t")
  wiki_redirects_index[parts[1]] = parts[2]
  line = it:read()
end

assert(wiki_redirects_index['Coercive'] == 'Coercion')
assert(wiki_redirects_index['Hosford, FL'] == 'Hosford, Florida')

print('    Done loading redirects index')


function get_redirected_ent_title(ent_name)
  if wiki_redirects_index[ent_name] then
    return wiki_redirects_index[ent_name]
  else
    return ent_name
  end
end
