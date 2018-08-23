-- Generate test data from the AIDA dataset by keeping the context and
-- entity candidates for each annotated mention

-- Format: 
-- doc_name \t doc_name \t mention \t left_ctxt \t right_ctxt \t CANDIDATES \t [ent_wikiid,p_e_m,ent_name]+ \t GT: \t pos,ent_wikiid,p_e_m,ent_name

-- Stats:
--cat aida_testA.csv | wc -l
--4791
--cat aida_testA.csv | grep -P 'GT:\t-1' | wc -l
--43
--cat aida_testA.csv | grep -P 'GT:\t1,' | wc -l
--3401

--cat aida_testB.csv | wc -l
--4485
--cat aida_testB.csv | grep -P 'GT:\t-1' | wc -l
--19
--cat aida_testB.csv | grep -P 'GT:\t1,' | wc -l
--3084

if not ent_p_e_m_index then
  require 'torch'
  dofile 'data_gen/indexes/wiki_redirects_index.lua'
  dofile 'data_gen/indexes/yago_crosswikis_wiki.lua'
  dofile 'utils/utils.lua'
end

tds = tds or require 'tds'

print('\nGenerating test data from AIDA set ')

it, _ = io.open(opt.root_data_dir .. 'basic_data/test_datasets/AIDA/testa_testb_aggregate_original')

out_file_A = opt.root_data_dir .. 'generated/test_train_data/aida_testA.csv'
out_file_B = opt.root_data_dir .. 'generated/test_train_data/aida_testB.csv'

ouf_A = assert(io.open(out_file_A, "w"))
ouf_B = assert(io.open(out_file_B, "w"))

local ouf = ouf_A

local num_nme = 0
local num_nonexistent_ent_title = 0
local num_nonexistent_ent_id = 0
local num_nonexistent_both = 0
local num_correct_ents = 0
local num_total_ents = 0

local cur_words_num = 0
local cur_words = {}
local cur_mentions = {}
local cur_mentions_num = 0

local cur_doc_name = ''

local function write_results()
  -- Write results:
  if cur_doc_name ~= '' then
    local header = cur_doc_name .. '\t' .. cur_doc_name .. '\t'
    for _, hyp in pairs(cur_mentions) do
      assert(hyp.mention:len() > 0, line)     
      local mention = hyp.mention
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
      if ent_p_e_m_index[mention] and #(ent_p_e_m_index[mention]) > 0 then

        local sorted_cand = {}
        for ent_wikiid,p in pairs(ent_p_e_m_index[mention]) do
          table.insert(sorted_cand, {ent_wikiid = ent_wikiid, p = p})
        end
        table.sort(sorted_cand, function(a,b) return a.p > b.p end)

        local candidates = {}
        local gt_pos = -1
        for pos,e in pairs(sorted_cand) do
          if pos <= 100 then
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
          ouf:write(str .. gt_pos .. ',' .. candidates[gt_pos] .. '\n')
        else
          if hyp.ent_wikiid ~= unk_ent_wikiid then
            ouf:write(str .. '-1,' .. hyp.ent_wikiid .. ',' .. get_ent_name_from_wikiid(hyp.ent_wikiid) .. '\n')
          else
            ouf:write(str .. '-1\n')
          end
        end
      else
        if hyp.ent_wikiid ~= unk_ent_wikiid then
          ouf:write(str .. 'EMPTYCAND\tGT:\t-1,' .. hyp.ent_wikiid .. ',' .. get_ent_name_from_wikiid(hyp.ent_wikiid) .. '\n')
        else
          ouf:write(str .. 'EMPTYCAND\tGT:\t-1\n')
        end
      end
    end
  end
end



local line = it:read()
while (line) do
  if (not line:find('-DOCSTART-')) then
    local parts = split(line, '\t')
    local num_parts = table_len(parts)
    assert(num_parts == 0 or num_parts == 1 or num_parts == 4 or num_parts == 7 or num_parts == 6, line)
    if num_parts > 0 then
      if num_parts == 4 and parts[2] == 'B' then
        num_nme = num_nme + 1
      end
      
      if (num_parts == 7 or num_parts == 6) and parts[2] == 'B' then
        
        -- Find current mention. A few hacks here.
        local cur_mention = preprocess_mention(parts[3])
        
        local x,y = parts[5]:find('/wiki/')
        local cur_ent_title = parts[5]:sub(y + 1)
        local cur_ent_wikiid = tonumber(parts[6])
        local index_ent_title = get_ent_name_from_wikiid(cur_ent_wikiid)
        local index_ent_wikiid = get_ent_wikiid_from_name(cur_ent_title)
        
        local final_ent_wikiid = index_ent_wikiid
        if final_ent_wikiid == unk_ent_wikiid then
          final_ent_wikiid = cur_ent_wikiid
        end
        
        if (index_ent_title == cur_ent_title and cur_ent_wikiid == index_ent_wikiid) then
          num_correct_ents = num_correct_ents + 1
        elseif (index_ent_title ~= cur_ent_title and cur_ent_wikiid ~= index_ent_wikiid) then
          num_nonexistent_both = num_nonexistent_both + 1
        elseif index_ent_title ~= cur_ent_title then
          assert(cur_ent_wikiid == index_ent_wikiid)
          num_nonexistent_ent_title = num_nonexistent_ent_title + 1
        else
          assert(index_ent_title == cur_ent_title)
          assert(cur_ent_wikiid ~= index_ent_wikiid)
          num_nonexistent_ent_id = num_nonexistent_ent_id + 1
        end
        
        num_total_ents = num_total_ents + 1 -- Keep even incorrect links

        cur_mentions_num = cur_mentions_num + 1
        cur_mentions[cur_mentions_num] = {}
        cur_mentions[cur_mentions_num].mention = cur_mention
        cur_mentions[cur_mentions_num].ent_wikiid = final_ent_wikiid
        cur_mentions[cur_mentions_num].start_off = cur_words_num + 1
        cur_mentions[cur_mentions_num].end_off = cur_words_num + table_len(split(parts[3], ' '))
      end
      
      local words_on_this_line = split_in_words(parts[1])
      for _,w in pairs(words_on_this_line) do
        table.insert(cur_words, modify_uppercase_phrase(w))
        cur_words_num =  cur_words_num + 1
      end
    end
    
  else
    assert(line:find('-DOCSTART-'))
    write_results()
    
    if cur_doc_name:find('testa') and line:find('testb') then
      ouf = ouf_B
      print('Done validation testA : ')
      print('num_nme = ' .. num_nme .. '; num_nonexistent_ent_title = ' .. num_nonexistent_ent_title)
      print('num_nonexistent_ent_id = ' .. num_nonexistent_ent_id .. '; num_nonexistent_both = ' .. num_nonexistent_both)
      print('num_correct_ents = ' .. num_correct_ents .. '; num_total_ents = ' .. num_total_ents)
    end
    
    local words = split_in_words(line)
    for _,w in pairs(words) do
      if w:find('testa') or w:find('testb') then
        cur_doc_name = w
        break
      end
    end    
    cur_words = {}
    cur_words_num = 0
    cur_mentions = {}
    cur_mentions_num = 0
  end
  
  line = it:read()
end

write_results()

ouf_A:flush()
io.close(ouf_A)
ouf_B:flush()
io.close(ouf_B)


print('    Done AIDA.')
print('num_nme = ' .. num_nme .. '; num_nonexistent_ent_title = ' .. num_nonexistent_ent_title)
print('num_nonexistent_ent_id = ' .. num_nonexistent_ent_id .. '; num_nonexistent_both = ' .. num_nonexistent_both)
print('num_correct_ents = ' .. num_correct_ents .. '; num_total_ents = ' .. num_total_ents)
