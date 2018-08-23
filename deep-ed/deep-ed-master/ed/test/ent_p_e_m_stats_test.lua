-- Statistics of annotated entities based on their p(e|m) prio
-- Table 6 (right) from our paper

local function ent_prior_to_key(f)
  if f <= 0.001 then
    return '<=0.001'
  elseif f <= 0.003 then
    return '0.001-0.003'
  elseif f <= 0.01 then
    return '0.003-0.01'
  elseif f <= 0.03 then
    return '0.01-0.03'
  elseif f <= 0.1 then
    return '0.03-0.1'
  elseif f <= 0.3 then
    return '0.1-0.3'
  else
    return '0.3+'
  end
end


function new_ent_prior_map()
  local m = {}
  m['<=0.001'] = 0.0
  m['0.001-0.003'] = 0.0
  m['0.003-0.01'] = 0.0
  m['0.01-0.03'] = 0.0  
  m['0.03-0.1'] = 0.0  
  m['0.1-0.3'] = 0.0  
  m['0.3+'] = 0.0  
  return m
end

function add_prior_to_ent_prior_map(m, f)
  m[ent_prior_to_key(f)] = m[ent_prior_to_key(f)] + 1
end

function print_ent_prior_maps_stats(smallm, bigm)
  print(' ===> entity p(e|m) stats :')  
  for k,_ in pairs(smallm) do
    local perc = 0
    if bigm[k] > 0 then 
      perc = 100.0 * smallm[k] / bigm[k]
    end
    assert(perc <= 100)
    print('p(e|m) = ' .. k .. ' : num = ' .. bigm[k] .. 
      ' ; correctly classified = ' .. smallm[k] .. 
      ' ; perc = ' .. string.format("%.2f", perc))
  end
end

  
  