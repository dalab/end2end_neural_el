-- Statistics of annotated entities based on their frequency in Wikipedia corpus 
-- Table 6 (left) from our paper
local function ent_freq_to_key(f)
  if f == 0 then
    return '0'
  elseif f == 1 then
    return '1'
  elseif f <= 5 then
    return '2-5'
  elseif f <= 10 then
    return '6-10'
  elseif f <= 20 then
    return '11-20'
  elseif f <= 50 then
    return '21-50'
  else
    return '50+'
  end
end


function new_ent_freq_map()
  local m = {}
  m['0'] = 0.0
  m['1'] = 0.0
  m['2-5'] = 0.0
  m['6-10'] = 0.0  
  m['11-20'] = 0.0  
  m['21-50'] = 0.0  
  m['50+'] = 0.0  
  return m
end

function add_freq_to_ent_freq_map(m, f)
  m[ent_freq_to_key(f)] = m[ent_freq_to_key(f)] + 1
end

function print_ent_freq_maps_stats(smallm, bigm)
  print(' ===> entity frequency stats :')
  for k,_ in pairs(smallm) do
    local perc = 0
    if bigm[k] > 0 then 
      perc = 100.0 * smallm[k] / bigm[k]
    end
    assert(perc <= 100)
    print('freq = ' .. k .. ' : num = ' .. bigm[k] .. 
      ' ; correctly classified = ' .. smallm[k] .. 
      ' ; perc = ' .. string.format("%.2f", perc))
  end
  print('')
end

  
  