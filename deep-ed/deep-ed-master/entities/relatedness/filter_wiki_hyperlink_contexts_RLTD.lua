-- Filter all training data s.t. only candidate entities and ground truth entities for which
-- we have a valid entity embedding are kept.

if not opt then
  cmd = torch.CmdLine()
  cmd:option('-root_data_dir', '', 'Root path of the data, $DATA_PATH.')
  cmd:text()
  opt = cmd:parse(arg or {})
  assert(opt.root_data_dir ~= '', 'Specify a valid root_data_dir path argument.')  
end


dofile 'utils/utils.lua'
dofile 'entities/relatedness/relatedness.lua'

input = opt.root_data_dir .. 'generated/wiki_hyperlink_contexts.csv'

output = opt.root_data_dir .. 'generated/wiki_hyperlink_contexts_RLTD.csv'
ouf = assert(io.open(output, "w"))

print('\nStarting dataset filtering.')
local cnt = 0
for line in io.lines(input) do
  cnt = cnt + 1
  if cnt % 50000 == 0 then
    print('    =======> processed ' .. cnt .. ' lines')
  end

  local parts = split(line, '\t')
  local grd_str = parts[table_len(parts)]
  assert(parts[table_len(parts) - 1] == 'GT:')
  local grd_str_parts = split(grd_str, ',')

  local grd_pos = tonumber(grd_str_parts[1])
  assert(grd_pos)
  
  local grd_ent_wikiid = tonumber(grd_str_parts[2])
  assert(grd_ent_wikiid)
  
  if rewtr.reltd_ents_wikiid_to_rltdid[grd_ent_wikiid] then
    assert(parts[6] == 'CANDIDATES')
    
    local output_line = parts[1] .. '\t' .. parts[2] .. '\t' .. parts[3] .. '\t' .. parts[4] .. '\t' .. parts[5] .. '\t' .. parts[6] .. '\t'
    
    local new_grd_pos = -1
    local new_grd_str_without_idx = nil
    
    local i = 1
    local added_ents = 0
    while (parts[6 + i] ~= 'GT:') do
      local str = parts[6 + i]
      local str_parts = split(str, ',')
      local ent_wikiid = tonumber(str_parts[1])
      if rewtr.reltd_ents_wikiid_to_rltdid[ent_wikiid] then
        added_ents = added_ents + 1
        output_line = output_line .. str .. '\t'
      end
      if (i == grd_pos) then
        assert(ent_wikiid == grd_ent_wikiid, 'Error for: ' .. line)
        new_grd_pos = added_ents
        new_grd_str_without_idx = str
      end
      
      i = i + 1
    end
    
    assert(new_grd_pos > 0)
    output_line = output_line .. 'GT:\t' .. new_grd_pos .. ',' .. new_grd_str_without_idx
    
    ouf:write(output_line .. '\n')
  end  
end  

ouf:flush() 
io.close(ouf)
