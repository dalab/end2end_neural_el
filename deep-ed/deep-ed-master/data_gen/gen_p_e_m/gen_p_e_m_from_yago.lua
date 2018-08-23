-- Generate p(e|m) index from Wikipedia
-- Run: th data_gen/gen_p_e_m/gen_p_e_m_from_yago.lua -root_data_dir $DATA_PATH

cmd = torch.CmdLine()
cmd:option('-root_data_dir', '', 'Root path of the data, $DATA_PATH.')
cmd:text()
opt = cmd:parse(arg or {})
assert(opt.root_data_dir ~= '', 'Specify a valid root_data_dir path argument.')


require 'torch'
dofile 'utils/utils.lua'
dofile 'data_gen/gen_p_e_m/unicode_map.lua'
if not get_redirected_ent_title then
  dofile 'data_gen/indexes/wiki_redirects_index.lua'
end
if not get_ent_name_from_wikiid then
  dofile 'entities/ent_name2id_freq/ent_name_id.lua'
end

tds = tds or require 'tds' 

print('\nComputing YAGO p_e_m')

local it, _ = io.open(opt.root_data_dir .. 'basic_data/p_e_m_data/aida_means.tsv')
local line = it:read()

local num_lines = 0
local wiki_e_m_counts = tds.Hash()

while (line) do
  num_lines = num_lines + 1
  if num_lines % 5000000 == 0 then
    print('Processed ' .. num_lines .. ' lines.')
  end
  local parts = split(line, '\t')
  assert(table_len(parts) == 2)
  assert(parts[1]:sub(1,1) == '"')
  assert(parts[1]:sub(parts[1]:len(),parts[1]:len()) == '"')
  
  local mention = parts[1]:sub(2, parts[1]:len() - 1)
  mention = trim1(mention)   -- NICK
  local ent_name = parts[2]
  ent_name = string.gsub(ent_name, '&amp;', '&')
  ent_name = string.gsub(ent_name, '&quot;', '"')
  while ent_name:find('\\u') do
    local x = ent_name:find('\\u')
    local code = ent_name:sub(x, x + 5)
    assert(unicode2ascii[code], code)
    replace = unicode2ascii[code]
    if(replace == "%") then
	    replace = "%%"
    end
    ent_name = string.gsub(ent_name, code, replace)
  end
  
  ent_name = preprocess_ent_name(ent_name)
  local ent_wikiid = get_ent_wikiid_from_name(ent_name, true)
  if ent_wikiid ~= unk_ent_wikiid and mention ~= "" then   -- NICK
    if not wiki_e_m_counts[mention] then
      wiki_e_m_counts[mention] = tds.Hash()
    end
    wiki_e_m_counts[mention][ent_wikiid] = 1
  end
  
  line = it:read()
end


print('Now sorting and writing ..')
out_file = opt.root_data_dir .. 'generated/yago_p_e_m.txt'
ouf = assert(io.open(out_file, "w"))

for mention, list in pairs(wiki_e_m_counts) do
  local str = ''
  local total_freq = 0
  for ent_wikiid, _ in pairs(list) do
    str = str .. ent_wikiid .. ',' .. get_ent_name_from_wikiid(ent_wikiid):gsub(' ', '_') .. '\t'
    total_freq = total_freq + 1
  end
  ouf:write(mention .. '\t' .. total_freq .. '\t' .. str .. '\n')
end
ouf:flush()
io.close(ouf)

print('    Done sorting and writing.')
