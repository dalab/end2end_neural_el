-- Define all parametrized layers: linear layers (diagonal matrices) A,B,C + network f

function new_linear_layer(out_dim)
  cmul = nn.CMul(out_dim)
  -- init weights with ones to speed up convergence
  cmul.weight = torch.ones(out_dim)
  return cmul
end

---- Create shared weights
A_linear = new_linear_layer(ent_vecs_size)

-- Local ctxt bilinear weights
B_linear = new_linear_layer(ent_vecs_size)

-- Used only in the global model
C_linear = new_linear_layer(ent_vecs_size)

f_network = nn.Sequential()
  :add(nn.Linear(2,opt.nn_pem_interm_size))
  :add(nn.ReLU())
  :add(nn.Linear(opt.nn_pem_interm_size,1))


function regularize_f_network()
  if opt.mat_reg_norm < 10 then
    for i = 1,f_network:size() do
      if f_network:get(i).weight and (f_network:get(i).weight:norm() > opt.mat_reg_norm) then
        f_network:get(i).weight:mul(opt.mat_reg_norm / f_network:get(i).weight:norm())
      end
      if f_network:get(i).bias and (f_network:get(i).bias:norm() > opt.mat_reg_norm) then
        f_network:get(i).bias:mul(opt.mat_reg_norm / f_network:get(i).bias:norm())
      end
    end
  end
end

function pack_saveable_weights()
  local linears = nn.Sequential():add(A_linear):add(B_linear):add(C_linear):add(f_network)
  return linears:float()
end

function unpack_saveable_weights(saved_linears)
  A_linear = saved_linears:get(1)
  B_linear = saved_linears:get(2)
  C_linear = saved_linears:get(3)
  f_network = saved_linears:get(4)
end


function print_net_weights()
  print('\nNetwork norms of parameter weights :')
  print('A (attention mat)  = ' .. A_linear.weight:norm())
  print('B (ctxt embedding) = ' .. B_linear.weight:norm())
  print('C (pairwise mat)   = ' .. C_linear.weight:norm())

  if opt.mat_reg_norm < 10 then
    print('f_network norm = ' .. f_network:get(1).weight:norm() .. ' ' ..
      f_network:get(1).bias:norm() .. ' ' .. f_network:get(3).weight:norm() .. ' ' ..
      f_network:get(3).bias:norm())      
  else
    p,gp = f_network:getParameters()
    print('f_network norm = ' .. p:norm())
  end
end
