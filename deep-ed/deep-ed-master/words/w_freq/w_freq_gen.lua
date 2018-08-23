-- Computes an unigram frequency of each word in the Wikipedia corpus

if not opt then
  cmd = torch.CmdLine()
  cmd:option('-root_data_dir', '', 'Root path of the data, $DATA_PATH.')
  cmd:text()
  opt = cmd:parse(arg or {})
  assert(opt.root_data_dir ~= '', 'Specify a valid root_data_dir path argument.')  
end


require 'torch'
dofile 'utils/utils.lua'

tds = tds or require 'tds' 

word_freqs = tds.Hash()

local num_lines = 0
it, _ = io.open(opt.root_data_dir .. 'generated/wiki_canonical_words.txt')
line = it:read()

while (line) do
  num_lines = num_lines + 1
  if num_lines % 100000 == 0 then
    print('Processed ' .. num_lines .. ' lines. ')
  end
  
  local parts = split(line , '\t')
  local words = split(parts[3], ' ')
  for _,w in pairs(words) do
    if (not word_freqs[w]) then
      word_freqs[w] = 0
    end
    word_freqs[w] = word_freqs[w] + 1
  end
  line = it:read()
end


-- Writing word frequencies
print('Sorting and writing')
sorted_word_freq = {}
for w,freq in pairs(word_freqs) do
  if freq >= 10 then
    table.insert(sorted_word_freq, {w = w, freq = freq})
  end
end

table.sort(sorted_word_freq, function(a,b) return a.freq > b.freq end)

out_file = opt.root_data_dir .. 'generated/word_wiki_freq.txt'
ouf = assert(io.open(out_file, "w"))
total_freq = 0
for _,x in pairs(sorted_word_freq) do
  ouf:write(x.w .. '\t' .. x.freq .. '\n')
  total_freq = total_freq + x.freq
end
ouf:flush()
io.close(ouf)

print('Total freq = ' .. total_freq .. '\n')
