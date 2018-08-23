-- Merge Wikipedia and Crosswikis p(e|m) indexes
-- Run: th data_gen/gen_p_e_m/merge_crosswikis_wiki.lua -root_data_dir $DATA_PATH

cmd = torch.CmdLine()
cmd:option('-root_data_dir', '', 'Root path of the data, $DATA_PATH.')
cmd:text()
opt = cmd:parse(arg or {})
assert(opt.root_data_dir ~= '', 'Specify a valid root_data_dir path argument.')


require 'torch'
dofile 'utils/utils.lua'
dofile 'entities/ent_name2id_freq/ent_name_id.lua'

print('\nMerging Wikipedia and Crosswikis p_e_m')

tds = tds or require 'tds' 
merged_e_m_counts = tds.Hash()

print('Process Wikipedia')
it, _ = io.open(opt.root_data_dir .. 'generated/wikipedia_p_e_m.txt')
line = it:read()

while (line) do
  local parts = split(line, "\t")
  local mention  = parts[1]  

  if (not mention:find('Wikipedia')) and (not mention:find('wikipedia')) then
    if not merged_e_m_counts[mention] then
      merged_e_m_counts[mention] = tds.Hash()
    end  

    local total_freq = tonumber(parts[2])
    assert(total_freq, line)
    local num_ents = table_len(parts)
    for i = 3, num_ents do
      local ent_str = split(parts[i], ",")
      local ent_wikiid = tonumber(ent_str[1])
      assert(ent_wikiid)
      local freq = tonumber(ent_str[2])
      assert(freq)

      if not merged_e_m_counts[mention][ent_wikiid] then
        merged_e_m_counts[mention][ent_wikiid] = 0
      end
      merged_e_m_counts[mention][ent_wikiid] = merged_e_m_counts[mention][ent_wikiid] + freq
    end
  end
  line = it:read()
end


print('Process Crosswikis')

it, _ = io.open(opt.root_data_dir .. 'basic_data/p_e_m_data/crosswikis_p_e_m.txt')
line = it:read()

while (line) do
  local parts = split(line, "\t")
  local mention  = trim1(parts[1])     -- NICK not sure if it is necessary

  if mention ~= "" and (not mention:find('Wikipedia')) and (not mention:find('wikipedia')) then
    if not merged_e_m_counts[mention] then
      merged_e_m_counts[mention] = tds.Hash()
    end  

    local total_freq = tonumber(parts[2])
    assert(total_freq)
    local num_ents = table_len(parts)
    for i = 3, num_ents do
      local ent_str = split(parts[i], ",")
      local ent_wikiid = tonumber(ent_str[1])
      assert(ent_wikiid)
      local freq = tonumber(ent_str[2])
      assert(freq)

      if not merged_e_m_counts[mention][ent_wikiid] then
        merged_e_m_counts[mention][ent_wikiid] = 0
      end
      merged_e_m_counts[mention][ent_wikiid] = merged_e_m_counts[mention][ent_wikiid] + freq
    end
  end
  line = it:read()
end


print('Now sorting and writing ..')
out_file = opt.root_data_dir .. 'generated/crosswikis_wikipedia_p_e_m.txt'
ouf = assert(io.open(out_file, "w"))
-- NICK
out_file2 = opt.root_data_dir .. 'generated/prob_crosswikis_wikipedia_p_e_m.txt'
ouf2 = assert(io.open(out_file2, "w"))

for mention, list in pairs(merged_e_m_counts) do
  if mention:len() >= 1 then
    local tbl = {}
    for ent_wikiid, freq in pairs(list) do
      table.insert(tbl, {ent_wikiid = ent_wikiid, freq = freq})
    end
    table.sort(tbl, function(a,b) return a.freq > b.freq end)

    local str = ''
    local total_freq = 0
    local num_ents = 0
    for _,el in pairs(tbl) do
      if is_valid_ent(el.ent_wikiid) then
        str = str .. el.ent_wikiid .. ',' .. el.freq 
        str = str .. ',' .. get_ent_name_from_wikiid(el.ent_wikiid):gsub(' ', '_') .. '\t'
        num_ents = num_ents + 1
        total_freq = total_freq + el.freq

        if num_ents >= 100 then -- At most 100 candidates
          break
        end
      end
    end
    ouf:write(mention .. '\t' .. total_freq .. '\t' .. str .. '\n')

    -- NICK
    str = ''
    num_ents = 0
    for _,el in pairs(tbl) do
      if is_valid_ent(el.ent_wikiid) then
        str = str .. el.ent_wikiid .. ',' .. (el.freq/total_freq) 
        str = str .. ',' .. get_ent_name_from_wikiid(el.ent_wikiid):gsub(' ', '_') .. '\t'
        num_ents = num_ents + 1

        if num_ents >= 100 then -- At most 100 candidates
          break
        end
      end
    end
    ouf2:write(mention .. '\t' .. 1 .. '\t' .. str .. '\n')   
    -- NICK end

  end
end
ouf:flush()
io.close(ouf)

ouf2:flush()
io.close(ouf2)

print('    Done sorting and writing.')

