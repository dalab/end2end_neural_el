-- Loads pre-trained word embeddings from either Word2Vec or Glove

assert(get_id_from_word)
assert(common_w2v_freq_words)
assert(total_num_words)

word_vecs_size = 300

-- Loads pre-trained glove or word2vec embeddings: 
if opt.word_vecs == 'glove' then
  -- Glove downloaded from: http://nlp.stanford.edu/projects/glove/
  w2v_txtfilename = default_path .. 'Glove/glove.840B.300d.txt'
  w2v_t7filename = opt.root_data_dir .. 'generated/glove.840B.300d.t7'
  w2v_reader = 'words/w2v/glove_reader.lua'
elseif opt.word_vecs == 'w2v' then
  -- Word2Vec downloaded from: https://code.google.com/archive/p/word2vec/
  w2v_binfilename = default_path .. 'Word2Vec/GoogleNews-vectors-negative300.bin'
  w2v_t7filename = opt.root_data_dir .. 'generated/GoogleNews-vectors-negative300.t7'
  w2v_reader = 'words/w2v/word2vec_reader.lua'
end

---------------------- Code: -----------------------
w2vutils = {}

print('==> Loading ' .. opt.word_vecs .. ' vectors')
if not paths.filep(w2v_t7filename) then
  print('  ---> t7 file NOT found. Loading w2v from the bin/txt file instead (slower).')
  w2vutils.M = require(w2v_reader)
  print('Writing t7 File for future usage. Next time Word2Vec loading will be faster!')
  torch.save(w2v_t7filename, w2vutils.M)
else
  print('  ---> from t7 file.')
  w2vutils.M = torch.load(w2v_t7filename)
end

-- Move the word embedding matrix on the GPU if we do some training. 
-- In this way we can perform word embedding lookup much faster.
if opt and string.find(opt.type, 'cuda') then
  w2vutils.M = w2vutils.M:cuda()
end

---------- Define additional functions -----------------
-- word -> vec
w2vutils.get_w_vec = function (self,word)
  local w_id = get_id_from_word(word)
  return w2vutils.M[w_id]:clone()
end

-- word_id -> vec
w2vutils.get_w_vec_from_id = function (self,w_id)
  return w2vutils.M[w_id]:clone()
end

w2vutils.lookup_w_vecs = function (self,word_id_tensor)
  assert(word_id_tensor:dim() <= 2, 'Only word id tensors w/ 1 or 2 dimensions are supported.')
  local output = torch.FloatTensor()
  local word_ids = word_id_tensor:long()
  if opt and string.find(opt.type, 'cuda') then
    output = output:cuda()
    word_ids = word_ids:cuda()
  end
  
  if word_ids:dim() == 2 then
    output:index(w2vutils.M, 1, word_ids:view(-1))
    output = output:view(word_ids:size(1), word_ids:size(2), w2vutils.M:size(2))
  elseif word_ids:dim() == 1 then
    output:index(w2vutils.M, 1, word_ids)
    output = output:view(word_ids:size(1), w2vutils.M:size(2))
  end

  return output
end

-- Normalize word vectors to have norm 1 .
w2vutils.renormalize = function (self)
  w2vutils.M[unk_w_id]:mul(0)
  w2vutils.M[unk_w_id]:add(1)
  w2vutils.M:cdiv(w2vutils.M:norm(2,2):expand(w2vutils.M:size()))
  local x = w2vutils.M:norm(2,2):view(-1) - 1
  assert(x:norm() < 0.1, x:norm())
  assert(w2vutils.M[100]:norm() < 1.001 and w2vutils.M[100]:norm() > 0.99)
  w2vutils.M[unk_w_id]:mul(0)
end

w2vutils:renormalize()

print('    Done reading w2v data. Word vocab size = ' .. w2vutils.M:size(1))

-- Phrase embedding using average of vectors of words in the phrase
w2vutils.phrase_avg_vec = function(self, phrase)
  local words = split_in_words(phrase)
  local num_words = table_len(words)
  local num_existent_words = 0
  local vec = torch.zeros(word_vecs_size)
  for i = 1,num_words do
    local w = words[i]
    local w_id = get_id_from_word(w)
    if w_id ~= unk_w_id then
      vec:add(w2vutils:get_w_vec_from_id(w_id))
      num_existent_words = num_existent_words + 1
    end
  end
  if (num_existent_words > 0) then
    vec:div(num_existent_words)
  end
  return vec
end

w2vutils.top_k_closest_words = function (self,vec, k, mat)
  local k = k or 1
  vec = vec:float()
  local distances = torch.mv(mat, vec)
  local best_scores, best_word_ids = topk(distances, k)
  local returnwords = {}
  local returndistances = {}
  for i = 1,k do
    local w = get_word_from_id(best_word_ids[i])
    if is_stop_word_or_number(w) then
      table.insert(returnwords, red(w))
    else
      table.insert(returnwords, w)
    end
    assert(best_scores[i] == distances[best_word_ids[i]], best_scores[i] .. '  ' .. distances[best_word_ids[i]])
    table.insert(returndistances, distances[best_word_ids[i]])
  end
  return returnwords, returndistances
end

w2vutils.most_similar2word = function(self, word, k)
  local k = k or 1
  local v = w2vutils:get_w_vec(word)
  neighbors, scores = w2vutils:top_k_closest_words(v, k, w2vutils.M)
  print('To word ' .. skyblue(word) .. ' : ' .. list_with_scores_to_str(neighbors, scores))
end

w2vutils.most_similar2vec = function(self, vec, k)
  local k = k or 1
  neighbors, scores = w2vutils:top_k_closest_words(vec, k, w2vutils.M)
  print(list_with_scores_to_str(neighbors, scores))
end


--------------------- Unit tests ----------------------------------------
local unit_tests = opt.unit_tests or false
if (unit_tests) then
  print('\nWord to word similarity test:')
  w2vutils:most_similar2word('nice', 5)
  w2vutils:most_similar2word('france', 5)
  w2vutils:most_similar2word('hello', 5)
end

-- Computes for each word w : \sum_v exp(<v,w>) and \sum_v <v,w>
w2vutils.total_word_correlation = function(self, k, j)
  local exp_Z = torch.zeros(w2vutils.M:narrow(1, 1, j):size(1))
  
  local sum_t = w2vutils.M:narrow(1, 1, j):sum(1) -- 1 x d
  local sum_Z = (w2vutils.M:narrow(1, 1, j) * sum_t:t()):view(-1) -- num_w

  print(red('Top words by sum_Z:'))
  best_sum_Z, best_word_ids = topk(sum_Z, k)
  for i = 1,k do
    local w = get_word_from_id(best_word_ids[i])
    assert(best_sum_Z[i] == sum_Z[best_word_ids[i]])
    print(w .. ' [' .. best_sum_Z[i] .. ']; ')
  end

  print('\n' .. red('Bottom words by sum_Z:'))
  best_sum_Z, best_word_ids = topk(- sum_Z, k)
  for i = 1,k do
    local w = get_word_from_id(best_word_ids[i])
    assert(best_sum_Z[i] == - sum_Z[best_word_ids[i]])
    print(w .. ' [' .. sum_Z[best_word_ids[i]] .. ']; ')
  end
end


-- Plot with gnuplot:
-- set palette model RGB defined ( 0 'white', 1 'pink', 2 'green' , 3 'blue', 4 'red' )
-- plot 'tsne-w2v-vecs.txt_1000' using 1:2:3 with labels offset 0,1, '' using 1:2:4 w points pt 7 ps 2 palette
w2vutils.tsne = function(self, num_rand_words)
  local topic1 = {'japan', 'china', 'france', 'switzerland', 'romania', 'india', 'australia', 'country', 'city', 'tokyo', 'nation', 'capital', 'continent', 'europe', 'asia', 'earth', 'america'}
  local topic2 = {'football', 'striker', 'goalkeeper', 'basketball', 'coach', 'championship', 'cup',
    'soccer', 'player', 'captain', 'qualifier', 'goal', 'under-21', 'halftime', 'standings', 'basketball', 
    'games', 'league', 'rugby', 'hockey', 'fifa', 'fans', 'maradona', 'mutu', 'hagi', 'beckham', 'injury', 'game', 
    'kick', 'penalty'}
  local topic_avg = {'japan national football team', 'germany national football team', 
    'china national football team', 'brazil soccer', 'japan soccer', 'germany soccer', 'china soccer', 
    'fc barcelona', 'real madrid'}  
  
  local stop_words_array = {}
  for w,_ in pairs(stop_words) do
    table.insert(stop_words_array, w)
  end
  
  local topic1_len = table_len(topic1)
  local topic2_len = table_len(topic2)
  local topic_avg_len = table_len(topic_avg)
  local stop_words_len = table_len(stop_words_array)
  
  torch.setdefaulttensortype('torch.DoubleTensor')
  w2vutils.M = w2vutils.M:double()
  
  local tensor = torch.zeros(num_rand_words + stop_words_len + topic1_len + topic2_len + topic_avg_len, word_vecs_size)
  local tensor_w_ids = torch.zeros(num_rand_words)
  local tensor_colors = torch.zeros(tensor:size(1))
  
  for i = 1,num_rand_words do
    tensor_w_ids[i] = math.random(1,25000)
    tensor_colors[i] = 0
    tensor[i]:copy(w2vutils.M[tensor_w_ids[i]])
  end
  
  for i = 1, stop_words_len do
    tensor_colors[num_rand_words + i] = 1
    tensor[num_rand_words + i]:copy(w2vutils:phrase_avg_vec(stop_words_array[i]))    
  end
  
  for i = 1, topic1_len do
    tensor_colors[num_rand_words + stop_words_len + i] = 2
    tensor[num_rand_words + stop_words_len + i]:copy(w2vutils:phrase_avg_vec(topic1[i]))    
  end
  
  for i = 1, topic2_len do
    tensor_colors[num_rand_words + stop_words_len + topic1_len + i] = 3
    tensor[num_rand_words + stop_words_len + topic1_len + i]:copy(w2vutils:phrase_avg_vec(topic2[i]))    
  end

  for i = 1, topic_avg_len do
    tensor_colors[num_rand_words + stop_words_len + topic1_len  + topic2_len + i] = 4
    tensor[num_rand_words + stop_words_len + topic1_len  + topic2_len + i]:copy(w2vutils:phrase_avg_vec(topic_avg[i]))    
  end
  
  local manifold = require 'manifold'
  opts = {ndims = 2, perplexity = 30, pca = 50, use_bh = false}
  mapped_x1 = manifold.embedding.tsne(tensor, opts)
  assert(mapped_x1:size(1) == tensor:size(1) and mapped_x1:size(2) == 2)
  ouf_vecs = assert(io.open('tsne-w2v-vecs.txt_' .. num_rand_words, "w"))
  for i = 1,mapped_x1:size(1) do
    local w = nil
    if tensor_colors[i] == 0 then 
      w = get_word_from_id(tensor_w_ids[i])
    elseif tensor_colors[i] == 1 then 
      w = stop_words_array[i - num_rand_words]:gsub(' ', '-')
    elseif tensor_colors[i] == 2 then 
      w = topic1[i - num_rand_words - stop_words_len]:gsub(' ', '-')
    elseif tensor_colors[i] == 3 then 
      w = topic2[i - num_rand_words - stop_words_len - topic1_len]:gsub(' ', '-')
    elseif tensor_colors[i] == 4 then 
      w = topic_avg[i - num_rand_words - stop_words_len - topic1_len - topic2_len]:gsub(' ', '-')    
    end
    assert(w)
    
    local v = mapped_x1[i]
    for j = 1,2 do
      ouf_vecs:write(v[j] .. ' ')
    end
    ouf_vecs:write(w .. ' ' .. tensor_colors[i] .. '\n')
  end
  io.close(ouf_vecs)
  print('    DONE')
end
