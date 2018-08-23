-- Torch layer that receives as input a squared matrix and sets its diagonal to a constant value.

local SetConstantDiag, parent = torch.class('nn.SetConstantDiag', 'nn.Module')

function SetConstantDiag:__init(constant_scalar, ip)
  parent.__init(self)
  assert(type(constant_scalar) == 'number', 'input is not scalar!')
  self.constant_scalar = constant_scalar
  
  -- default for inplace is false
   self.inplace = ip or false
   if (ip and type(ip) ~= 'boolean') then
      error('in-place flag must be boolean')
   end
   if not opt then
     opt = {}
     opt.type = 'double'
     dofile 'utils/utils.lua'
   end
end

function SetConstantDiag:updateOutput(input)
  assert(input:dim() == 3)
  assert(input:size(2) == input:size(3))
  local n = input:size(3)
  local prod_mat = torch.ones(n,n) - torch.eye(n) 
  prod_mat = correct_type(prod_mat)
  local sum_mat = torch.eye(n):mul(self.constant_scalar)
  sum_mat = correct_type(sum_mat)
  if self.inplace then
    input:cmul(torch.repeatTensor(prod_mat, input:size(1), 1, 1))
    input:add(torch.repeatTensor(sum_mat, input:size(1), 1, 1))
    self.output:set(input)
  else
    self.output:resizeAs(input)
    self.output:copy(input)
    self.output:cmul(torch.repeatTensor(prod_mat, input:size(1), 1, 1))
    self.output:add(torch.repeatTensor(sum_mat, input:size(1), 1, 1))
  end
  return self.output
end 

function SetConstantDiag:updateGradInput(input, gradOutput)
  local n = input:size(3)
  local prod_mat = torch.ones(n,n) - torch.eye(n) 
  prod_mat = correct_type(prod_mat)
  if self.inplace then
    self.gradInput:set(gradOutput)
  else
    self.gradInput:resizeAs(gradOutput)
    self.gradInput:copy(gradOutput)
  end
  self.gradInput:cmul(torch.repeatTensor(prod_mat, input:size(1), 1, 1))
  return self.gradInput
end