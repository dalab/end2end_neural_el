-- Utility functions to extract the text and hyperlinks from each page in the Wikipedia corpus.

if not table_len then
  dofile 'utils/utils.lua'
end
if not get_redirected_ent_title then
  dofile 'data_gen/indexes/wiki_redirects_index.lua'
end
if not get_ent_name_from_wikiid then
  dofile 'entities/ent_name2id_freq/ent_name_id.lua'
end


function extract_text_and_hyp(line, mark_mentions)
  local list_hyp = {} -- (mention, entity) pairs
  local text = ''
  local list_ent_errors = 0
  local parsing_errors = 0
  local disambiguation_ent_errors = 0
  local diez_ent_errors = 0
  
  local end_end_hyp = 0
  local begin_end_hyp = 0
  local begin_start_hyp, end_start_hyp = line:find('<a href="')
  
  local num_mentions = 0
  
  while begin_start_hyp do
    text = text .. line:sub(end_end_hyp + 1, begin_start_hyp - 1)
    
    local next_quotes,end_quotes = line:find('">', end_start_hyp + 1)
    if next_quotes then
      local ent_name = line:sub(end_start_hyp + 1, next_quotes - 1)
      begin_end_hyp, end_end_hyp = line:find('</a>', end_quotes + 1)
      if begin_end_hyp then
        local mention = line:sub(end_quotes + 1, begin_end_hyp - 1)
        local mention_marker = false
        
        local good_mention = true
        good_mention = good_mention and (not mention:find('Wikipedia'))
        good_mention = good_mention and (not mention:find('wikipedia'))
        good_mention = good_mention and (mention:len() >= 1)
        
        if good_mention then
          local i = ent_name:find('wikt:')
          if i == 1 then
            ent_name = ent_name:sub(6)
          end
          ent_name = preprocess_ent_name(ent_name)

          i = ent_name:find('List of ')
          if (not i) or (i ~= 1) then
            if ent_name:find('#') then
              diez_ent_errors = diez_ent_errors + 1
            else
              local ent_wikiid = get_ent_wikiid_from_name(ent_name, true)
              if ent_wikiid == unk_ent_wikiid then
                disambiguation_ent_errors = disambiguation_ent_errors + 1
              else
                -- A valid (entity,mention) pair
                num_mentions = num_mentions + 1
                table.insert(list_hyp, {mention = mention, ent_wikiid = ent_wikiid, cnt = num_mentions})
                if mark_mentions then
                  mention_marker = true
                end
              end
            end
          else
            list_ent_errors = list_ent_errors + 1
          end
        end
        
        if (not mention_marker) then
          text = text .. ' ' .. mention .. ' '
        else
          text = text .. ' MMSTART' .. num_mentions .. ' ' .. mention .. ' MMEND' .. num_mentions .. ' '
        end
      else 
        parsing_errors = parsing_errors + 1
        begin_start_hyp = nil
      end
    else 
      parsing_errors = parsing_errors + 1
      begin_start_hyp = nil
    end
    
    if begin_start_hyp then
      begin_start_hyp, end_start_hyp = line:find('<a href="', end_start_hyp + 1)
    end
  end
  
  if end_end_hyp then
    text = text .. line:sub(end_end_hyp + 1)
  else
    if (not mark_mentions) then
      text = line -- Parsing did not succed, but we don't throw this line away.
    else
      text = ''
      list_hyp = {}
    end
  end
  
  return list_hyp, text, list_ent_errors, parsing_errors, disambiguation_ent_errors, diez_ent_errors
end

----------------------------- Unit tests -------------
print('\n Unit tests:')
local test_line_1 = '<a href="Anarchism">Anarchism</a> is a <a href="political philosophy">political philosophy</a> that advocates<a href="stateless society">stateless societies</a>often defined as <a href="self-governance">self-governed</a> voluntary institutions, but that several authors have defined as more specific institutions based on non-<a href="Hierarchy">hierarchical</a> <a href="Free association (communism and anarchism)">free associations</a>..<a href="Anarchism">Anarchism</a>'

local test_line_2 = 'CSF pressure, as measured by <a href="lumbar puncture">lumbar puncture</a> (LP), is 10-18 <a href="Pressure#H2O">'
local test_line_3 = 'Anarchism'

list_hype, text = extract_text_and_hyp(test_line_1, false)
print(list_hype)
print(text)
print()

list_hype, text = extract_text_and_hyp(test_line_1, true)
print(list_hype)
print(text)
print()

list_hype, text = extract_text_and_hyp(test_line_2, true)
print(list_hype)
print(text)
print()

list_hype, text = extract_text_and_hyp(test_line_3, false)
print(list_hype)
print(text)
print()
print('    Done unit tests.')
---------------------------------------------------------


function extract_page_entity_title(line)
  local startoff, endoff = line:find('<doc id="')
  assert(startoff, line)
  local startquotes, _ = line:find('"', endoff + 1)
  local ent_wikiid = tonumber(line:sub(endoff + 1, startquotes - 1))
  assert(ent_wikiid, line .. ' ==> ' .. line:sub(startoff + 1, startquotes - 1))
  local starttitlestartoff, starttitleendoff = line:find(' title="')
  local endtitleoff, _ = line:find('">')
  local ent_name = line:sub(starttitleendoff + 1, endtitleoff - 1)
  if (ent_wikiid ~= get_ent_wikiid_from_name(ent_name, true)) then
    -- Most probably this is a disambiguation or list page
    local new_ent_wikiid = get_ent_wikiid_from_name(ent_name, true)
--    print(red('Error in Wiki dump: ' .. line .. ' ' .. ent_wikiid .. ' ' .. new_ent_wikiid))
    return new_ent_wikiid
  end
  return ent_wikiid
end


local test_line_4 = '<doc id="12" url="http://en.wikipedia.org/wiki?curid=12" title="Anarchism">'

print(extract_page_entity_title(test_line_4))
