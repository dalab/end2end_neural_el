local M = torch.zeros(total_num_words(), word_vecs_size):float()

--Reading Contents
for line in io.lines(w2v_txtfilename) do
  local parts = split(line, ' ')
  local w = parts[1]
  local w_id = get_id_from_word(w)
  if w_id ~= unk_w_id then
    for i=2, #parts do
      M[w_id][i-1] = tonumber(parts[i])
    end
  end
end

return M


