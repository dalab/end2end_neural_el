
if not opt then
  cmd = torch.CmdLine()
  cmd:option('-root_data_dir', '', 'Root path of the data, $DATA_PATH.')
  cmd:text()
  opt = cmd:parse(arg or {})
  assert(opt.root_data_dir ~= '', 'Specify a valid root_data_dir path argument.')  
end


dofile 'utils/utils.lua'
dofile 'entities/relatedness/relatedness.lua'

input = opt.root_data_dir .. 'generated/wiki_canonical_words.txt'

output = opt.root_data_dir .. 'generated/wiki_canonical_words_RLTD.txt'
ouf = assert(io.open(output, "w"))

print('\nStarting dataset filtering.')

local cnt = 0
for line in io.lines(input) do
  cnt = cnt + 1
  if cnt % 500000 == 0 then
    print('    =======> processed ' .. cnt .. ' lines')
  end

  local parts = split(line, '\t')
  assert(table_len(parts) == 3)

  local ent_wikiid = tonumber(parts[1])
  local ent_name = parts[2]
  assert(ent_wikiid)
  
  if rewtr.reltd_ents_wikiid_to_rltdid[ent_wikiid] then
    ouf:write(line .. '\n')
  end  
end  

ouf:flush() 
io.close(ouf)
