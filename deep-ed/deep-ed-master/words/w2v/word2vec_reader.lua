-- Adapted from https://github.com/rotmanmi/word2vec.torch
function read_string_w2v(file)  
  local str = {}
  while true do
    local char = file:readChar()
    if char == 32 or char == 10 or char == 0 then
      break
    else
      str[#str+1] = char
    end
  end
  str = torch.CharStorage(str)
  return str:string()
end


file = torch.DiskFile(w2v_binfilename,'r')

--Reading Header
file:ascii()
local vocab_size = file:readInt()
local size = file:readInt()
assert(size == word_vecs_size, 'Wrong size : ' .. size .. ' vs ' .. word_vecs_size)

local M = torch.zeros(total_num_words(), word_vecs_size):float()

--Reading Contents
file:binary()
local num_phrases = 0
for i = 1,vocab_size do
  local w = read_string_w2v(file)
  local v = torch.FloatTensor(file:readFloat(word_vecs_size))
  local w_id = get_id_from_word(w)
  if w_id ~= unk_w_id then
    M[w_id]:copy(v)
  end
end

print('Num words = ' .. total_num_words() .. '. Num phrases = ' .. num_phrases)

return M
