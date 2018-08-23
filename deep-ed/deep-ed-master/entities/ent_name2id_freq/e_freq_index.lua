-- Loads an index containing entity -> frequency pairs. 
-- TODO: rewrite this file in a simpler way (is complicated because of some past experiments).
tds = tds or require 'tds' 

print('==> Loading entity freq map') 

local ent_freq_file = opt.root_data_dir .. 'generated/ent_wiki_freq.txt'

min_freq = 1
e_freq = tds.Hash()
e_freq.ent_f_start = tds.Hash()
e_freq.ent_f_end = tds.Hash()
e_freq.total_freq = 0
e_freq.sorted = tds.Hash()

cur_start = 1
cnt = 0
for line in io.lines(ent_freq_file) do
  local parts = split(line, '\t')
  local ent_wikiid = tonumber(parts[1])
  local ent_f = tonumber(parts[3])
  assert(ent_wikiid)
  assert(ent_f)
  
  if (not rewtr) or rewtr.reltd_ents_wikiid_to_rltdid[ent_wikiid] then
    e_freq.ent_f_start[ent_wikiid] = cur_start
    e_freq.ent_f_end[ent_wikiid] = cur_start + ent_f - 1
    e_freq.sorted[cnt] = ent_wikiid
    cur_start = cur_start + ent_f
    cnt = cnt + 1
  end
end

e_freq.total_freq = cur_start - 1

print('    Done loading entity freq index. Size = ' .. cnt)

function get_ent_freq(ent_wikiid) 
  if e_freq.ent_f_start[ent_wikiid] then
    return e_freq.ent_f_end[ent_wikiid] - e_freq.ent_f_start[ent_wikiid] + 1
  end
  return 0
end

