function topk(one_dim_tensor, k) 
  local bestk, indices = torch.topk(one_dim_tensor, k, true)
  local sorted, newindices = torch.sort(bestk, true)
  local oldindices = torch.LongTensor(k)
  for i = 1,k do
    oldindices[i] = indices[newindices[i]]
  end
  return sorted, oldindices
end


function list_with_scores_to_str(list, scores)
  local str = ''
  for i,v in pairs(list) do
    str = str .. list[i] .. '[' .. string.format("%.2f", scores[i]) .. ']; '
  end
  return str
end

function table_len(t)
  local count = 0
  for _ in pairs(t) do count = count + 1 end
  return count
end


function split(inputstr, sep)
  if sep == nil then
    sep = "%s"
  end
  local t={} ; i=1
  for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
    t[i] = str
    i = i + 1
  end
  return t
end
-- Unit test:
assert(6 == #split('aa_bb cc__dd   ee  _  _   __ff' , '_ '))



function correct_type(data) 
  if opt.type == 'float' then return data:float()
  elseif opt.type == 'double' then return data:double()
  elseif string.find(opt.type, 'cuda') then return data:cuda()
  else print('Unsuported type')
  end
end

-- color fonts:
function red(s) 
  return '\27[31m' .. s .. '\27[39m'
end

function green(s)
  return '\27[32m' .. s .. '\27[39m'
end

function yellow(s)
  return '\27[33m' .. s .. '\27[39m'
end

function blue(s)
  return '\27[34m' .. s .. '\27[39m'
end

function violet(s)
  return '\27[35m' .. s .. '\27[39m'
end

function skyblue(s)
  return '\27[36m' .. s .. '\27[39m'
end



function split_in_words(inputstr)
  local words = {}
  for word in inputstr:gmatch("%w+") do table.insert(words, word) end
  return words
end


function first_letter_to_uppercase(s)
  return s:sub(1,1):upper() .. s:sub(2)
end

function modify_uppercase_phrase(s)
  if (s == s:upper()) then
    local words = split_in_words(s:lower())
    local res = {}
    for _,w in pairs(words) do
      table.insert(res, first_letter_to_uppercase(w))
    end
    return table.concat(res, ' ') 
  else
    return s
  end
end

function blue_num_str(n)
  return blue(string.format("%.3f", n))
end



function string_starts(s, m)
   return string.sub(s,1,string.len(m)) == m
end

-- trim:
function trim1(s)
  return (s:gsub("^%s*(.-)%s*$", "%1"))
end


function nice_print_red_green(a,b)
  local s = string.format("%.3f", a) .. ':' .. string.format("%.3f", b) .. '['
  if a > b then
    return s .. red(string.format("%.3f", a-b)) .. ']'
  elseif a < b then
    return s .. green(string.format("%.3f", b-a)) .. ']'
  else
    return s .. '0]'
  end
end
