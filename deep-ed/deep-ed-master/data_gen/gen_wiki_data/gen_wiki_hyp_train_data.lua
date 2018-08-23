-- Generate training data from Wikipedia hyperlinks by keeping the context and
-- entity candidates for each hyperlink

-- Format: 
-- ent_wikiid \t ent_name \t mention \t left_ctxt \t right_ctxt \t CANDIDATES \t [ent_wikiid,p_e_m,ent_name]+ \t GT: \t pos,ent_wikiid,p_e_m,ent_name

if not opt then
  cmd = torch.CmdLine()
  cmd:option('-root_data_dir', '', 'Root path of the data, $DATA_PATH.')
  cmd:text()
  opt = cmd:parse(arg or {})
  assert(opt.root_data_dir ~= '', 'Specify a valid root_data_dir path argument.')  
end

require 'torch'
dofile 'data_gen/parse_wiki_dump/parse_wiki_dump_tools.lua'
dofile 'data_gen/indexes/yago_crosswikis_wiki.lua'
tds = tds or require 'tds' 

print('\nGenerating training data from Wiki dump')

it, _ = io.open(opt.root_data_dir .. 'basic_data/textWithAnchorsFromAllWikipedia2014Feb.txt')

out_file = opt.root_data_dir .. 'generated/wiki_hyperlink_contexts.csv'
ouf = assert(io.open(out_file, "w"))

-- Find anchors, e.g. <a href="wikt:anarchism">anarchism</a>
local num_lines = 0
local num_valid_hyp = 0

local cur_words_num = 0
local cur_words = {}
local cur_mentions = {}
local cur_mentions_num = 0
local cur_ent_wikiid = -1

local line = it:read()
while (line) do
  num_lines = num_lines + 1
  if num_lines % 1000000 == 0 then
    print('Processed ' .. num_lines .. ' lines. Num valid hyp = ' .. num_valid_hyp)
  end

  -- If it's a line from the Wiki page, add its text words and its hyperlinks
  if (not line:find('<doc id="')) and (not line:find('</doc>')) then
    list_hyp, text, _, _ , _ , _ = extract_text_and_hyp(line, true)
   
    local words_on_this_line = split_in_words(text)
    local num_added_hyp = 0
    local line_mentions = {}
    for _,w in pairs(words_on_this_line) do
      local wstart = string_starts(w, 'MMSTART')
      local wend = string_starts(w, 'MMEND')
      if (not wstart) and (not wend) then
        table.insert(cur_words, w)
        cur_words_num = cur_words_num + 1
      elseif wstart then
        local mention_idx = tonumber(w:sub(1 + ('MMSTART'):len()))
        assert(mention_idx, w)
        line_mentions[mention_idx] = {start_off = (cur_words_num + 1), end_off = -1}
      elseif wend then
        num_added_hyp = num_added_hyp + 1
        local mention_idx = tonumber(w:sub(1 + ('MMEND'):len()))
        assert(mention_idx, w)
        assert(line_mentions[mention_idx])
        line_mentions[mention_idx].end_off = cur_words_num
      end
    end
    
    assert(table_len(list_hyp) == num_added_hyp, line .. ' :: ' .. text .. ' :: ' .. num_added_hyp .. ' ' .. table_len(list_hyp))
    for _,hyp in pairs(list_hyp) do
      assert(line_mentions[hyp.cnt])
      cur_mentions_num = cur_mentions_num + 1
      cur_mentions[cur_mentions_num] = {}
      cur_mentions[cur_mentions_num].mention = hyp.mention
      cur_mentions[cur_mentions_num].ent_wikiid = hyp.ent_wikiid
      cur_mentions[cur_mentions_num].start_off = line_mentions[hyp.cnt].start_off
      cur_mentions[cur_mentions_num].end_off = line_mentions[hyp.cnt].end_off
    end
    
  elseif line:find('<doc id="') then
    
    -- Write results:
    if cur_ent_wikiid ~= unk_ent_wikiid and is_valid_ent(cur_ent_wikiid) then
      local header = cur_ent_wikiid .. '\t' .. get_ent_name_from_wikiid(cur_ent_wikiid) .. '\t'
      for _, hyp in pairs(cur_mentions) do
        if ent_p_e_m_index[hyp.mention] and #(ent_p_e_m_index[hyp.mention]) > 0 then
          assert(hyp.mention:len() > 0, line)
          local str = header .. hyp.mention .. '\t'

          local left_ctxt = {}
          for i = math.max(0, hyp.start_off - 100), hyp.start_off - 1 do
            table.insert(left_ctxt, cur_words[i])
          end
          if table_len(left_ctxt) == 0 then
            table.insert(left_ctxt, 'EMPTYCTXT')
          end
          str = str .. table.concat(left_ctxt, ' ') .. '\t'

          local right_ctxt = {}
          for i = hyp.end_off + 1, math.min(cur_words_num, hyp.end_off + 100) do
            table.insert(right_ctxt, cur_words[i])
          end
          if table_len(right_ctxt) == 0 then
            table.insert(right_ctxt, 'EMPTYCTXT')
          end
          str = str .. table.concat(right_ctxt, ' ') .. '\tCANDIDATES\t'
          
          -- Entity candidates from p(e|m) dictionary
          local unsorted_cand = {}
          for ent_wikiid,p in pairs(ent_p_e_m_index[hyp.mention]) do
            table.insert(unsorted_cand, {ent_wikiid = ent_wikiid, p = p})
          end
          table.sort(unsorted_cand, function(a,b) return a.p > b.p end)
          
          local candidates = {}
          local gt_pos = -1
          for pos,e in pairs(unsorted_cand) do
            if pos <= 32 then
              table.insert(candidates, e.ent_wikiid .. ',' .. string.format("%.3f", e.p) .. ',' .. get_ent_name_from_wikiid(e.ent_wikiid))
              if e.ent_wikiid == hyp.ent_wikiid then
                gt_pos = pos
              end
            else
              break
            end
          end
          str = str .. table.concat(candidates, '\t') .. '\tGT:\t'
          
          if gt_pos > 0 then
            num_valid_hyp = num_valid_hyp + 1
            ouf:write(str .. gt_pos .. ',' .. candidates[gt_pos] .. '\n')
          end
        end
      end
    end
    
    cur_ent_wikiid = extract_page_entity_title(line)
    cur_words = {}
    cur_words_num = 0
    cur_mentions = {}
    cur_mentions_num = 0
  end
  
  line = it:read()
end
ouf:flush()
io.close(ouf)

print('    Done generating training data from Wiki dump. Num valid hyp = ' .. num_valid_hyp)
