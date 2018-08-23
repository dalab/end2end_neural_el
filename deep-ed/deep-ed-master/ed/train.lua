-- Training of ED models.

if opt.opt == 'SGD' then
   optimState = {
      learningRate = opt.lr,
      momentum = 0.9,
      learningRateDecay = 5e-7
   }
   optimMethod = optim.sgd

elseif opt.opt == 'ADAM' then  -- See: http://cs231n.github.io/neural-networks-3/#update 
  optimState = {
    learningRate = opt.lr,
  }
  optimMethod = optim.adam

elseif opt.opt == 'ADADELTA' then  -- See: http://cs231n.github.io/neural-networks-3/#update 
  -- Run with default parameters, no need for learning rate and other stuff
  optimState = {}
  optimConfig = {}   
  optimMethod = optim.adadelta
  
elseif opt.opt == 'ADAGRAD' then  -- See: http://cs231n.github.io/neural-networks-3/#update 
  optimMethod = optim.adagrad
  optimState = {
    learningRate = opt.lr 
  } 
  
else
   error('unknown optimization method')
end
----------------------------------------------------------------------

-- Each batch is one document, so we test/validate/save the current model after each set of 
-- 5000 documents. Since aida-train contains 946 documents, this is equivalent with 5 full epochs.
num_batches_per_epoch = 5000 

function train_and_test()
  
  print('\nDone testing for ' .. banner)
  print('Params serialized = ' .. params_serialized)  
  
  -- epoch tracker
  epoch = 1
  
  local processed_so_far = 0
  
  local f_bs = 0
  local gradParameters_bs = nil

  while true do
    local time = sys.clock()
    print('\n')
    print('One epoch = ' .. (num_batches_per_epoch / 1000) .. ' full passes over AIDA-TRAIN in our case.')
    print(green('==> TRAINING EPOCH #' .. epoch .. ' <=='))
    
    print_net_weights()
    
    local processed_mentions = 0
    for batch_index = 1,num_batches_per_epoch  do
      -- Read one mini-batch from one data_thread:
      local inputs, targets = get_minibatch()
      
      local num_mentions = targets:size(1)
      processed_mentions = processed_mentions + num_mentions
      
      local model, _ = get_model(num_mentions)
      model:training()

      -- Retrieve parameters and gradients:
      -- extracts and flattens all model's parameters into a 1-dim vector
      parameters,gradParameters = model:getParameters()
      gradParameters:zero()

      -- Just in case:
      collectgarbage()
      collectgarbage()

      -- Reset gradients
      gradParameters:zero()

      -- Evaluate function for complete mini batch

      local outputs = model:forward(inputs)
      assert(outputs:size(1) == num_mentions and outputs:size(2) == max_num_cand)
      local f = criterion:forward(outputs, targets)

      -- Estimate df/dW
      local df_do = criterion:backward(outputs, targets)

      model:backward(inputs, df_do)

      if opt.batch_size == 1 or batch_index % opt.batch_size == 1 then
        gradParameters_bs = gradParameters:clone():zero()
        f_bs = 0
      end
      
      gradParameters_bs:add(gradParameters)
      f_bs = f_bs + f
      
      if opt.batch_size == 1 or batch_index % opt.batch_size == 0 then
      
        gradParameters_bs:div(opt.batch_size)
        f_bs = f_bs / opt.batch_size
        
        -- Create closure to evaluate f(X) and df/dX
        local feval = function(x)
          return f_bs, gradParameters_bs
        end

        -- Optimize on current mini-batch
        optimState.learningRate = opt.lr
        optimMethod(feval, parameters, optimState)
      
        -- Regularize the f_network with projected SGD.
        regularize_f_network()
      end
    
      -- Display progress
      processed_so_far = processed_so_far + num_mentions
      if processed_so_far > 100000000 then
        processed_so_far = processed_so_far - 100000000
      end
      xlua.progress(processed_so_far, 100000000)      
    end  

    -- Measure time taken
    time = sys.clock() - time
    time = time / processed_mentions
    
    print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')
  
    -- Test:
    test(epoch)
    print('\nDone testing for ' .. banner)
    print('Params serialized = ' .. params_serialized)
    
    -- Save the current model:
    if opt.save then
      local filename = opt.root_data_dir .. 'generated/ed_models/' .. params_serialized .. '|ep=' .. epoch
      print('==> saving model to '..filename)
      torch.save(filename, pack_saveable_weights())
    end
    
    -- Next epoch
    epoch = epoch + 1
  end
end
