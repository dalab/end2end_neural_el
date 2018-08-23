if opt.loss == 'nll' then
  criterion = nn.CrossEntropyCriterion()
else
  -- max-margin with margin parameter = 0.01
  criterion = nn.MultiMarginCriterion(1, torch.ones(max_num_cand), 0.01)
end

if string.find(opt.type, 'cuda') then
  criterion = criterion:cuda()
end