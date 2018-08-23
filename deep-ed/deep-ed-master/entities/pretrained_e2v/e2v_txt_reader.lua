print('==> loading e2v')

local V =  torch.ones(get_total_num_ents(), ent_vecs_size):mul(1e-10) -- not zero because of cosine_distance layer

local cnt = 0
for line in io.lines(e2v_txtfilename) do
  cnt = cnt + 1
  if cnt % 1000000 == 0 then
    print('=======> processed ' .. cnt .. ' lines')
  end

  local parts = split(line, ' ')
  assert(table_len(parts) == ent_vecs_size + 1)
  local ent_wikiid = tonumber(parts[1])
  local vec = torch.zeros(ent_vecs_size)
  for i=1,ent_vecs_size do
    vec[i] = tonumber(parts[i + 1])
  end   
  
  if (contains_thid(ent_wikiid)) then 
    V[get_thid(ent_wikiid)] = vec
  else
    print('Ent id = ' .. ent_wikiid .. ' does not have a vector. ')
  end
end

print('    Done loading entity vectors. Size = ' .. cnt .. '\n')

print('Writing t7 File for future usage. Next time Ent2Vec will load faster!')
torch.save(e2v_t7filename, V)
print('    Done saving.\n')

return V