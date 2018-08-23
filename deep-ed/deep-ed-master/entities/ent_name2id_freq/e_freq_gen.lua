-- Creates a file that contains entity frequencies.

if not opt then
  cmd = torch.CmdLine()
  cmd:option('-root_data_dir', '', 'Root path of the data, $DATA_PATH.')
  cmd:text()
  opt = cmd:parse(arg or {})
  assert(opt.root_data_dir ~= '', 'Specify a valid root_data_dir path argument.')  
end


require 'torch'
tds = tds or require 'tds' 

dofile 'utils/utils.lua'
dofile 'entities/ent_name2id_freq/ent_name_id.lua'

entity_freqs = tds.Hash()

local num_lines = 0
it, _ = io.open(opt.root_data_dir .. 'generated/crosswikis_wikipedia_p_e_m.txt')
line = it:read()

while (line) do
  num_lines = num_lines + 1
  if num_lines % 2000000 == 0 then
    print('Processed ' .. num_lines .. ' lines. ')
  end
  
  local parts = split(line , '\t')
  local num_parts = table_len(parts)
  for i = 3, num_parts do
    local ent_str = split(parts[i], ',')
    local ent_wikiid = tonumber(ent_str[1])
    local freq = tonumber(ent_str[2])
    assert(ent_wikiid)
    assert(freq)
    
    if (not entity_freqs[ent_wikiid]) then
      entity_freqs[ent_wikiid] = 0
    end
    entity_freqs[ent_wikiid] = entity_freqs[ent_wikiid] + freq
  end
  line = it:read()
end


-- Writing word frequencies
print('Sorting and writing')
sorted_ent_freq = {}
for ent_wikiid,freq in pairs(entity_freqs) do
  if freq >= 10 then
    table.insert(sorted_ent_freq, {ent_wikiid = ent_wikiid, freq = freq})
  end
end

table.sort(sorted_ent_freq, function(a,b) return a.freq > b.freq end)

out_file = opt.root_data_dir .. 'generated/ent_wiki_freq.txt'
ouf = assert(io.open(out_file, "w"))
total_freq = 0
for _,x in pairs(sorted_ent_freq) do
  ouf:write(x.ent_wikiid .. '\t' .. get_ent_name_from_wikiid(x.ent_wikiid) .. '\t' .. x.freq .. '\n')
  total_freq = total_freq + x.freq
end
ouf:flush()
io.close(ouf)

print('Total freq = ' .. total_freq .. '\n')
