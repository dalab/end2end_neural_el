-- Given a dataset, try to retrieve better entity candidates
-- for ambiguous mentions of persons. For example, suppose a document 
-- contains a mention of a person called 'Peter Such' that can be easily solved with 
-- the current system. Now suppose that, in the same document, there 
-- exists a mention 'Such' referring to the same person. For this 
-- second highly ambiguous mention, retrieving the correct entity in 
-- top K candidates would be very hard. We adopt here a simple heuristical strategy of 
-- searching in the same document all potentially coreferent mentions that strictly contain 
-- the given mention as a substring. If such mentions exist and they refer to 
-- persons (contain at least one person candidate entity), then the ambiguous 
-- shorter mention gets as candidates the candidates of the longer mention.

tds = tds or require 'tds'
assert(get_ent_wikiid_from_name)
print('==> Loading index of Wiki entities that represent persons.')

local persons_ent_wikiids = tds.Hash()
for line in io.lines(opt.root_data_dir .. 'basic_data/p_e_m_data/persons.txt') do
  local ent_wikiid = get_ent_wikiid_from_name(line, true)
  if ent_wikiid ~= unk_ent_wikiid then
    persons_ent_wikiids[ent_wikiid] = 1
  end
end

function is_person(ent_wikiid)
  return persons_ent_wikiids[ent_wikiid]
end

print('    Done loading persons index. Size = ' .. #persons_ent_wikiids)


local function mention_refers_to_person(m, mention_ent_cand)
  local top_p = 0
  local top_ent = -1
  for e_wikiid, p_e_m in pairs(mention_ent_cand[m]) do
    if p_e_m > top_p then
      top_ent = e_wikiid
      top_p = p_e_m
    end
  end
  return is_person(top_ent)
end


function build_coreference_dataset(dataset_lines, banner)
  if (not opt.coref) then
    return dataset_lines
  else
  
    -- Create new entity candidates
    local coref_dataset_lines = tds.Hash()
    for doc_id, lines_map in pairs(dataset_lines) do 
      
      coref_dataset_lines[doc_id] = tds.Hash()
      
      -- Collect entity candidates for each mention.
      local mention_ent_cand = {}
      for _,sample_line in pairs(lines_map)  do
        local parts = split(sample_line, "\t")
        assert(doc_id == parts[1])
        local mention = parts[3]:lower()
        if not mention_ent_cand[mention] then
          mention_ent_cand[mention] = {}
          assert(parts[6] == 'CANDIDATES')  
          if parts[7] ~= 'EMPTYCAND' then
            local num_cand = 1
            while parts[6 + num_cand] ~= 'GT:' do
              local cand_parts = split(parts[6 + num_cand], ',')
              local cand_ent_wikiid = tonumber(cand_parts[1])
              local cand_p_e_m = tonumber(cand_parts[2])
              assert(cand_p_e_m >= 0, cand_p_e_m)    
              assert(cand_ent_wikiid)
              mention_ent_cand[mention][cand_ent_wikiid] = cand_p_e_m
              num_cand = num_cand + 1
            end
          end
        end
      end
      
      -- Find coreferent mentions
      for _,sample_line in pairs(lines_map)  do
        local parts = split(sample_line, "\t")
        assert(doc_id == parts[1])
        local mention = parts[3]:lower()
        
        assert(mention_ent_cand[mention])
        assert(parts[table_len(parts) - 1] == 'GT:')
      
        -- Grd trth infos
        local grd_trth_parts = split(parts[table_len(parts)], ',')
        local grd_trth_idx = tonumber(grd_trth_parts[1])
        assert(grd_trth_idx == -1 or table_len(grd_trth_parts) >= 4, sample_line)
        local grd_trth_entwikiid = -1
        if table_len(grd_trth_parts) >= 3 then
          grd_trth_entwikiid = tonumber(grd_trth_parts[2])
        end
        assert(grd_trth_entwikiid)
      
        -- Merge lists of entity candidates
        local added_list = {}
        local num_added_mentions = 0
        for m,_ in pairs(mention_ent_cand) do
          local stupid_lua_pattern = string.gsub(mention, '%.', '%%%.')
          stupid_lua_pattern = string.gsub(stupid_lua_pattern, '%-', '%%%-')
          if m ~= mention and (string.find(m, ' ' .. stupid_lua_pattern) or string.find(m, stupid_lua_pattern .. ' ')) and mention_refers_to_person(m, mention_ent_cand) then
            
            if banner == 'aida-B' then
              print(blue('coref mention = ' .. m .. ' replaces original mention = ' .. mention) ..
                ' ; DOC = ' .. doc_id)
            end
            
            num_added_mentions = num_added_mentions + 1
            for e_wikiid, p_e_m in pairs(mention_ent_cand[m]) do
              if not added_list[e_wikiid] then
                added_list[e_wikiid] = 0.0
              end
              added_list[e_wikiid] = added_list[e_wikiid] + p_e_m
            end
          end
        end
        
        -- Average:
        for e_wikiid, _ in pairs(added_list) do
          added_list[e_wikiid] = added_list[e_wikiid] / num_added_mentions
        end
        
        -- Merge the two lists
        local merged_list = mention_ent_cand[mention]
        if num_added_mentions > 0 then
          merged_list = added_list
        end        
        
        local sorted_list = {}
        for ent_wikiid,p in pairs(merged_list) do
          table.insert(sorted_list, {ent_wikiid = ent_wikiid, p = p})
        end
        table.sort(sorted_list, function(a,b) return a.p > b.p end)
      
        -- Write the new line
        local str = parts[1] .. '\t' .. parts[2] .. '\t' .. parts[3] .. '\t' .. parts[4] .. '\t' 
          .. parts[5] .. '\t' .. parts[6] .. '\t' 
      
        if table_len(sorted_list) == 0 then
          str = str .. 'EMPTYCAND\tGT:\t-1'
          if grd_trth_entwikiid ~= unk_ent_wikiid then
            str = str .. ',' .. grd_trth_entwikiid .. ',' ..
            get_ent_name_from_wikiid(grd_trth_entwikiid)
          end
        else
          local candidates = {}
          local gt_pos = -1
          for pos,e in pairs(sorted_list) do
            if pos <= 100 then
              table.insert(candidates, e.ent_wikiid .. ',' ..
                string.format("%.3f", e.p) .. ',' .. get_ent_name_from_wikiid(e.ent_wikiid))
              if e.ent_wikiid == grd_trth_entwikiid then
                gt_pos = pos
              end
            else
              break
            end
          end
          str = str .. table.concat(candidates, '\t') .. '\tGT:\t'

          if gt_pos > 0 then
            str = str .. gt_pos .. ',' .. candidates[gt_pos]
          else
            str = str .. '-1'
            if grd_trth_entwikiid ~= unk_ent_wikiid then
              str = str .. ',' .. grd_trth_entwikiid .. ',' ..
                get_ent_name_from_wikiid(grd_trth_entwikiid)
            end
          end
        end
        
        coref_dataset_lines[doc_id][1 + #coref_dataset_lines[doc_id]] = str
      end  
    end
    
    assert(#dataset_lines == #coref_dataset_lines)
    for doc_id, lines_map in pairs(dataset_lines) do
      assert(table_len(dataset_lines[doc_id]) == table_len(coref_dataset_lines[doc_id]))
    end
    
    return coref_dataset_lines
  end
end