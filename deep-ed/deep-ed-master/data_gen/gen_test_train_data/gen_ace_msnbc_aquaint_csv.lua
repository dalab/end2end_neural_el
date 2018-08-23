-- Generate test data from the ACE/MSNBC/AQUAINT datasets by keeping the context and
-- entity candidates for each annotated mention

-- Format: 
-- doc_name \t doc_name \t mention \t left_ctxt \t right_ctxt \t CANDIDATES \t [ent_wikiid,p_e_m,ent_name]+ \t GT: \t pos,ent_wikiid,p_e_m,ent_name

-- Stats:
--cat wned-ace2004.csv |  wc -l
--257
--cat wned-ace2004.csv |  grep -P 'GT:\t-1' | wc -l
--20
--cat wned-ace2004.csv | grep -P 'GT:\t1,' | wc -l
--217

--cat wned-aquaint.csv |  wc -l
--727
--cat wned-aquaint.csv |  grep -P 'GT:\t-1' | wc -l
--33
--cat wned-aquaint.csv | grep -P 'GT:\t1,' | wc -l
--604

--cat wned-msnbc.csv  | wc -l
--656
--cat wned-msnbc.csv |  grep -P 'GT:\t-1' | wc -l
--22
--cat wned-msnbc.csv | grep -P 'GT:\t1,' | wc -l
--496

if not ent_p_e_m_index then
  require 'torch'
  dofile 'data_gen/indexes/wiki_redirects_index.lua'
  dofile 'data_gen/indexes/yago_crosswikis_wiki.lua'
  dofile 'utils/utils.lua'
end

tds = tds or require 'tds'

local function gen_test_ace(dataset) 

  print('\nGenerating test data from ' .. dataset .. ' set ')

  local path = opt.root_data_dir .. 'basic_data/test_datasets/wned-datasets/' .. dataset .. '/'

  out_file = opt.root_data_dir .. 'generated/test_train_data/wned-' .. dataset .. '.csv'
  ouf = assert(io.open(out_file, "w"))

  annotations, _ = io.open(path .. dataset .. '.xml')
  
  local num_nonexistent_ent_id = 0
  local num_correct_ents = 0

  local cur_doc_text = ''
  local cur_doc_name = ''

  local line = annotations:read()
  while (line) do
    if (not line:find('document docName=\"')) then
      if line:find('<annotation>') then
        line = annotations:read()
        local x,y = line:find('<mention>')
        local z,t = line:find('</mention>')
        local cur_mention = line:sub(y + 1, z - 1)
        cur_mention = string.gsub(cur_mention, '&amp;', '&')
        
        line = annotations:read()
        x,y = line:find('<wikiName>')
        z,t = line:find('</wikiName>')
        cur_ent_title = ''
        if not line:find('<wikiName/>') then
          cur_ent_title = line:sub(y + 1, z - 1)
        end
        
        line = annotations:read()
        x,y = line:find('<offset>')
        z,t = line:find('</offset>')
        local offset = 1 + tonumber(line:sub(y + 1, z - 1))
        
        line = annotations:read()
        x,y = line:find('<length>')
        z,t = line:find('</length>')
        local length = tonumber(line:sub(y + 1, z - 1))
        length = cur_mention:len()

        line = annotations:read()
        if line:find('<entity/>') then
          line = annotations:read()
        end
        
        assert(line:find('</annotation>'))
        
        offset = math.max(1, offset - 10)
        while (cur_doc_text:sub(offset, offset + length - 1) ~= cur_mention) do
--          print(cur_mention .. ' ---> ' .. cur_doc_text:sub(offset, offset + length - 1))
          offset = offset + 1
        end
        
        cur_mention = preprocess_mention(cur_mention)
        
        if cur_ent_title ~= 'NIL' and cur_ent_title ~= '' and cur_ent_title:len() > 0 then
          local cur_ent_wikiid = get_ent_wikiid_from_name(cur_ent_title)
          if cur_ent_wikiid == unk_ent_wikiid then
            num_nonexistent_ent_id = num_nonexistent_ent_id + 1
            print(green(cur_ent_title))
          else
            num_correct_ents = num_correct_ents + 1
          end
          
          assert(cur_mention:len() > 0)
          local str = cur_doc_name .. '\t' .. cur_doc_name .. '\t' .. cur_mention .. '\t'
          
          local left_words = split_in_words(cur_doc_text:sub(1, offset - 1))
          local num_left_words = table_len(left_words)
          local left_ctxt = {}
          for i = math.max(1, num_left_words - 100 + 1), num_left_words do
            table.insert(left_ctxt, left_words[i])
          end
          if table_len(left_ctxt) == 0 then
            table.insert(left_ctxt, 'EMPTYCTXT')
          end
          str = str .. table.concat(left_ctxt, ' ') .. '\t'

          local right_words = split_in_words(cur_doc_text:sub(offset + length))
          local num_right_words = table_len(right_words)
          local right_ctxt = {}
          for i = 1, math.min(num_right_words, 100) do
            table.insert(right_ctxt, right_words[i])
          end
          if table_len(right_ctxt) == 0 then
            table.insert(right_ctxt, 'EMPTYCTXT')
          end
          str = str .. table.concat(right_ctxt, ' ') .. '\tCANDIDATES\t'


          -- Entity candidates from p(e|m) dictionary
          if ent_p_e_m_index[cur_mention] and #(ent_p_e_m_index[cur_mention]) > 0 then

            local sorted_cand = {}
            for ent_wikiid,p in pairs(ent_p_e_m_index[cur_mention]) do
              table.insert(sorted_cand, {ent_wikiid = ent_wikiid, p = p})
            end
            table.sort(sorted_cand, function(a,b) return a.p > b.p end)

            local candidates = {}
            local gt_pos = -1
            for pos,e in pairs(sorted_cand) do
              if pos <= 100 then
                table.insert(candidates, e.ent_wikiid .. ',' .. string.format("%.3f", e.p) .. ',' .. get_ent_name_from_wikiid(e.ent_wikiid))
                if e.ent_wikiid == cur_ent_wikiid then
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
              if cur_ent_wikiid ~= unk_ent_wikiid then
                ouf:write(str .. '-1,' .. cur_ent_wikiid .. ',' .. cur_ent_title .. '\n')
              else
                ouf:write(str .. '-1\n')
              end
            end
          else
            if cur_ent_wikiid ~= unk_ent_wikiid then
              ouf:write(str .. 'EMPTYCAND\tGT:\t-1,' .. cur_ent_wikiid .. ',' .. cur_ent_title .. '\n')
            else
              ouf:write(str .. 'EMPTYCAND\tGT:\t-1\n')
            end
          end

        end
      end
    else
      local x,y = line:find('document docName=\"')
      local z,t = line:find('\">')
      cur_doc_name = line:sub(y + 1, z - 1)
      cur_doc_name = string.gsub(cur_doc_name, '&amp;', '&')
      
      local it,_ = io.open(path .. 'RawText/' .. cur_doc_name)
      cur_doc_text = ''
      local cur_line = it:read()
      while cur_line do
        cur_doc_text = cur_doc_text .. cur_line .. ' '
        cur_line = it:read()
      end
      cur_doc_text = string.gsub(cur_doc_text, '&amp;', '&')
    end
    line = annotations:read()
  end

  ouf:flush()
  io.close(ouf)

  print('Done ' .. dataset .. '.')
  print('num_nonexistent_ent_id = ' .. num_nonexistent_ent_id .. '; num_correct_ents = ' .. num_correct_ents)
end


gen_test_ace('wikipedia')
gen_test_ace('clueweb')
gen_test_ace('ace2004')
gen_test_ace('msnbc') 
gen_test_ace('aquaint')
