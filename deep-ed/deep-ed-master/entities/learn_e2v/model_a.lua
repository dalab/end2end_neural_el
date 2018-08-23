-- Definition of the neural network used to learn entity embeddings.
-- To run a simple unit test that checks the forward and backward passes, just run :
--    th entities/learn_e2v/model_a.lua

if not opt then -- unit tests
  unit_tests = true
  dofile 'utils/utils.lua'
  require 'nn'
  cmd = torch.CmdLine()
  cmd:option('-type', 'double', 'type: double | float | cuda | cudacudnn')
  cmd:option('-batch_size', 7, 'mini-batch size (1 = pure stochastic)')
  cmd:option('-num_words_per_ent', 100, 'num positive words per entity per iteration.')
  cmd:option('-num_neg_words', 25, 'num negative words in the partition function.')
  cmd:option('-loss', 'nce', 'nce | neg | is | maxm')
  cmd:option('-init_vecs_title_words', false, 'whether the entity embeddings should be initialized with the average of title word embeddings. Helps to speed up convergence speed of entity embeddings learning.')
  opt = cmd:parse(arg or {})
  word_vecs_size = 5
  ent_vecs_size = word_vecs_size
  lookup_ent_vecs = nn.LookupTable(100, ent_vecs_size)
end -- end unit tests


if not unit_tests then 
  ent_vecs_size = word_vecs_size
  
  -- Init ents vectors
  print('\n==> Init entity embeddings matrix. Num ents = ' .. get_total_num_ents())
  lookup_ent_vecs = nn.LookupTable(get_total_num_ents(), ent_vecs_size)
  
  -- Zero out unk_ent_thid vector for unknown entities.
  lookup_ent_vecs.weight[unk_ent_thid]:copy(torch.zeros(ent_vecs_size))
  
  -- Init entity vectors with average of title word embeddings. 
  -- This would help speed-up training.
  if opt.init_vecs_title_words then
    print('Init entity embeddings with average of title word vectors to speed up learning.')
    for ent_thid = 1,get_total_num_ents() do
      local init_ent_vec = torch.zeros(ent_vecs_size)
      local ent_name = get_ent_name_from_wikiid(get_wikiid_from_thid(ent_thid))
      words_plus_stop_words = split_in_words(ent_name)
      local num_words_title = 0
      for _,w in pairs(words_plus_stop_words) do
        if contains_w(w) then -- Remove stop words.
          init_ent_vec:add(w2vutils.M[get_id_from_word(w)]:float())
          num_words_title = num_words_title + 1
        end
      end
      
      if num_words_title > 0 then
        if num_words_title > 3 then
          assert(init_ent_vec:norm() > 0, ent_name)
        end
        init_ent_vec:div(num_words_title)
      end
      
      if init_ent_vec:norm() > 0 then
        lookup_ent_vecs.weight[ent_thid]:copy(init_ent_vec)
      end
    end
  end

  collectgarbage(); collectgarbage();
  print('    Done init.')
end

---------------- Model Definition --------------------------------
cosine_words_ents = nn.Sequential()
  :add(nn.ConcatTable()
    :add(nn.Sequential()
      :add(nn.SelectTable(1))
      :add(nn.SelectTable(2)) -- ctxt words vectors
      :add(nn.Normalize(2))
      :add(nn.View(opt.batch_size, opt.num_words_per_ent * opt.num_neg_words, ent_vecs_size)))
    :add(nn.Sequential()
      :add(nn.SelectTable(3))
      :add(nn.SelectTable(1))
      :add(lookup_ent_vecs) -- entity vectors
      :add(nn.Normalize(2))
      :add(nn.View(opt.batch_size, 1, ent_vecs_size))))
  :add(nn.MM(false, true))
  :add(nn.View(opt.batch_size * opt.num_words_per_ent, opt.num_neg_words))
  
model = nn.Sequential()
  :add(cosine_words_ents)
  :add(nn.View(opt.batch_size * opt.num_words_per_ent, opt.num_neg_words))
  

if opt.loss == 'is' then
  model = nn.Sequential()
    :add(nn.ConcatTable()
      :add(model)
      :add(nn.Sequential()
        :add(nn.SelectTable(1))
        :add(nn.SelectTable(3)) -- unigram distributions at power
        :add(nn.Log())
        :add(nn.View(opt.batch_size * opt.num_words_per_ent, opt.num_neg_words))))
    :add(nn.CSubTable())  

elseif opt.loss == 'nce' then
  model = nn.Sequential()
    :add(nn.ConcatTable()
      :add(model)
      :add(nn.Sequential()
        :add(nn.SelectTable(1))
        :add(nn.SelectTable(3)) -- unigram distributions at power
        :add(nn.MulConstant(opt.num_neg_words - 1))
        :add(nn.Log())
        :add(nn.View(opt.batch_size * opt.num_words_per_ent, opt.num_neg_words))))
    :add(nn.CSubTable())
end
  
---------------------------------------------------------------------------------------------

------- Cuda conversions:
if string.find(opt.type, 'cuda') then
  model = model:cuda()  --- This has to be called always before cudnn.convert
end

if string.find(opt.type, 'cudacudnn') then
  cudnn.convert(model, cudnn)
end


--- Unit tests
if unit_tests then
  print('Network model unit tests:')
  local inputs = {}
  
  inputs[1] = {}
  inputs[1][1] = correct_type(torch.ones(opt.batch_size * opt.num_words_per_ent * opt.num_neg_words)) -- ctxt words
  
  inputs[2] = {}
  inputs[2][1] = correct_type(torch.ones(opt.batch_size * opt.num_words_per_ent)) -- ent wiki words
  
  inputs[3] = {}
  inputs[3][1] = correct_type(torch.ones(opt.batch_size)) -- ent th ids
  inputs[3][2] = torch.ones(opt.batch_size) -- ent wikiids
  
  -- ctxt word vecs
  inputs[1][2] = correct_type(torch.ones(opt.batch_size * opt.num_words_per_ent * opt.num_neg_words, word_vecs_size))

  inputs[1][3] = correct_type(torch.randn(opt.batch_size * opt.num_words_per_ent * opt.num_neg_words))

  local outputs = model:forward(inputs)
  
  assert(outputs:size(1) == opt.batch_size * opt.num_words_per_ent and
    outputs:size(2) == opt.num_neg_words)
  print('FWD success!')

  model:backward(inputs, correct_type(torch.randn(opt.batch_size * opt.num_words_per_ent, opt.num_neg_words)))
  print('BKWD success!')
end
