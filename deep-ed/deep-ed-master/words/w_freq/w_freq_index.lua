-- Loads all words and their frequencies and IDs from a dictionary.
assert(common_w2v_freq_words)
if not opt.unig_power then
  opt.unig_power = 0.6
end
  
print('==> Loading word freq map with unig power ' .. red(opt.unig_power))
local w_freq_file = opt.root_data_dir .. 'generated/word_wiki_freq.txt'

local w_freq = {}
w_freq.id2word = tds.Hash()
w_freq.word2id = tds.Hash()

w_freq.w_f_start = tds.Hash()
w_freq.w_f_end = tds.Hash()
w_freq.total_freq = 0.0

w_freq.w_f_at_unig_power_start = tds.Hash()
w_freq.w_f_at_unig_power_end = tds.Hash()
w_freq.total_freq_at_unig_power = 0.0

-- UNK word id
unk_w_id = 1
w_freq.word2id['UNK_W'] = unk_w_id 
w_freq.id2word[unk_w_id] = 'UNK_W'

local tmp_wid = 1
for line in io.lines(w_freq_file) do  
  local parts = split(line, '\t')
  local w = parts[1]
  if common_w2v_freq_words[w] then
    tmp_wid = tmp_wid + 1
    local w_id = tmp_wid
    w_freq.id2word[w_id] = w
    w_freq.word2id[w] = w_id

    local w_f = tonumber(parts[2])
    assert(w_f)
    if w_f < 100 then
      w_f = 100
    end
    w_freq.w_f_start[w_id] = w_freq.total_freq
    w_freq.total_freq = w_freq.total_freq + w_f
    w_freq.w_f_end[w_id] = w_freq.total_freq

    w_freq.w_f_at_unig_power_start[w_id] = w_freq.total_freq_at_unig_power
    w_freq.total_freq_at_unig_power = w_freq.total_freq_at_unig_power + math.pow(w_f, opt.unig_power)
    w_freq.w_f_at_unig_power_end[w_id] = w_freq.total_freq_at_unig_power
  end
end

w_freq.total_num_words = tmp_wid

print('    Done loading word freq index. Num words = ' .. w_freq.total_num_words  .. '; total freq = ' .. w_freq.total_freq)

--------------------------------------------
total_num_words = function()
  return w_freq.total_num_words
end

contains_w_id = function(w_id)
  assert(w_id >= 1 and w_id <= total_num_words(), w_id)
  return (w_id ~= unk_w_id)
end

-- id -> word
get_word_from_id = function(w_id)
  assert(w_id >= 1 and w_id <= total_num_words(), w_id)
  return w_freq.id2word[w_id]
end

-- word -> id
get_id_from_word = function(w)
  local w_id = w_freq.word2id[w]
  if w_id == nil then
    return unk_w_id
  end
  return w_id
end

contains_w = function(w)
  return contains_w_id(get_id_from_word(w))
end

-- word frequency:
function get_w_id_freq(w_id)
  assert(contains_w_id(w_id), w_id)
  return w_freq.w_f_end[w_id] - w_freq.w_f_start[w_id] + 1
end

-- p(w) prior:
function get_w_id_unigram(w_id)
  return get_w_id_freq(w_id) / w_freq.total_freq
end

function get_w_tensor_log_unigram(vec_w_ids)
  assert(vec_w_ids:dim() == 2)
  local v = torch.zeros(vec_w_ids:size(1), vec_w_ids:size(2))
  for i= 1,vec_w_ids:size(1) do
    for j = 1,vec_w_ids:size(2) do
      v[i][j] = math.log(get_w_id_unigram(vec_w_ids[i][j]))
    end
  end
  return v  
end


if (opt.unit_tests) then
  print(get_w_id_unigram(get_id_from_word('the')))
  print(get_w_id_unigram(get_id_from_word('of')))
  print(get_w_id_unigram(get_id_from_word('and')))
  print(get_w_id_unigram(get_id_from_word('romania')))
end

-- Generates an random word sampled from the word unigram frequency.
local function random_w_id(total_freq, w_f_start, w_f_end)
  local j = math.random() * total_freq
  local i_start = 2
  local i_end = total_num_words()

  while i_start <= i_end do
    local i_mid = math.floor((i_start + i_end) / 2)
    local w_id_mid = i_mid
    if w_f_start[w_id_mid] <= j and j <= w_f_end[w_id_mid] then
      return w_id_mid
    elseif (w_f_start[w_id_mid] > j) then
      i_end = i_mid - 1
    elseif (w_f_end[w_id_mid] < j) then
      i_start = i_mid + 1
    end
  end
  print(red('Binary search error !!'))
end

-- Frequent word subsampling procedure from the Word2Vec paper.
function random_unigram_at_unig_power_w_id()
  return random_w_id(w_freq.total_freq_at_unig_power, w_freq.w_f_at_unig_power_start, w_freq.w_f_at_unig_power_end)
end


function get_w_unnorm_unigram_at_power(w_id)
  return math.pow(get_w_id_unigram(w_id), opt.unig_power)
end


function unit_test_random_unigram_at_unig_power_w_id(k_samples)
  local empirical_dist = {}
  for i=1,k_samples do
    local w_id = random_unigram_at_unig_power_w_id()
    assert(w_id ~= unk_w_id)
    if not empirical_dist[w_id] then
      empirical_dist[w_id] = 0
    end
    empirical_dist[w_id] = empirical_dist[w_id] + 1
  end
  print('Now sorting ..')
  local sorted_empirical_dist = {}
  for k,v in pairs(empirical_dist) do
    table.insert(sorted_empirical_dist, {w_id = k, f = v})
  end
  table.sort(sorted_empirical_dist, function(a,b) return a.f > b.f end)

  local str = ''
  for i = 1,math.min(100, table_len(sorted_empirical_dist)) do
    str = str .. get_word_from_id(sorted_empirical_dist[i].w_id) .. '{' .. sorted_empirical_dist[i].f .. '}; '
  end
  print('Unit test random sampling: ' .. str)
end
