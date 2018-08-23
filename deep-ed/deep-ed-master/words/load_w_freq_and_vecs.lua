-- Loads all common words in both Wikipedia and Word2vec/Glove , their unigram frequencies and their pre-trained Word2Vec embeddings.

-- To load this as a standalone file do:
--    th> opt = {word_vecs = 'w2v', root_data_dir = '$DATA_PATH'}
--    th> dofile 'words/load_w_freq_and_vecs.lua'

if not opt then
  cmd = torch.CmdLine()
  cmd:option('-root_data_dir', '', 'Root path of the data, $DATA_PATH.')
  cmd:text()
  opt = cmd:parse(arg or {})
  assert(opt.root_data_dir ~= '', 'Specify a valid root_data_dir path argument.')  
end


default_path = opt.root_data_dir .. 'basic_data/wordEmbeddings/'

tds = tds or require 'tds' 
torch.setdefaulttensortype('torch.FloatTensor')
if not list_with_scores_to_str then
  dofile 'utils/utils.lua'
end
if not is_stop_word_or_number then
  dofile 'words/stop_words.lua'
end

assert(opt, 'Define opt')
assert(opt.word_vecs, 'Define opt.word_vecs')

print('==> Loading common w2v + top freq list of words')

local output_t7filename = opt.root_data_dir .. 'generated/common_top_words_freq_vectors_' .. opt.word_vecs .. '.t7'

if paths.filep(output_t7filename) then
  print('  ---> from t7 file.')
  common_w2v_freq_words = torch.load(output_t7filename)
  
else
  print('  ---> t7 file NOT found. Loading from disk instead (slower). Out file = ' .. output_t7filename)
  local freq_words = tds.Hash()
  
  print('   word freq index ...')
  local num_freq_words = 1
  local w_freq_file = opt.root_data_dir .. 'generated/word_wiki_freq.txt'
  for line in io.lines(w_freq_file) do
    local parts = split(line, '\t')
    local w = parts[1]
    local w_f = tonumber(parts[2])
    if not is_stop_word_or_number(w) then
      freq_words[w] = w_f
      num_freq_words = num_freq_words + 1
    end
  end

  common_w2v_freq_words = tds.Hash()

  print('   word vectors index ...')
  if opt.word_vecs == 'glove' then
    w2v_txtfilename = default_path .. 'Glove/glove.840B.300d.txt'
    local line_num = 0
    for line in io.lines(w2v_txtfilename) do
      line_num = line_num + 1
      if line_num % 200000 == 0 then
        print('   Processed ' .. line_num)
      end
      local parts = split(line, ' ')
      local w = parts[1]
      if freq_words[w] then
        common_w2v_freq_words[w] = 1
      end
    end

  else
    assert(opt.word_vecs == 'w2v')
    w2v_binfilename = default_path .. 'Word2Vec/GoogleNews-vectors-negative300.bin'  
    local word_vecs_size = 300
    file = torch.DiskFile(w2v_binfilename,'r')
    file:ascii()
    local vocab_size = file:readInt()
    local size = file:readInt()
    assert(size == word_vecs_size, 'Wrong size : ' .. size .. ' vs ' .. word_vecs_size)

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

    --Reading Contents
    file:binary()
    local line_num = 0
    for i = 1,vocab_size do
      line_num = line_num + 1
      if line_num % 200000 == 0 then
        print('Processed ' .. line_num)
      end
      local w = read_string_w2v(file)
      local v = torch.FloatTensor(file:readFloat(word_vecs_size))
      if freq_words[w] then
        common_w2v_freq_words[w] = 1
      end
    end
  end
  
  print('Writing t7 File for future usage. Next time loading will be faster!')
  torch.save(output_t7filename, common_w2v_freq_words)
end

-- Now load the freq and w2v indexes
dofile 'words/w_freq/w_freq_index.lua'
