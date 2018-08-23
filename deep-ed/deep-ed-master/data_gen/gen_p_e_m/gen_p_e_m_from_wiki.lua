-- Generate p(e|m) index from Wikipedia
-- Run: th data_gen/gen_p_e_m/gen_p_e_m_from_wiki.lua -root_data_dir $DATA_PATH

cmd = torch.CmdLine()
cmd:option('-root_data_dir', '', 'Root path of the data, $DATA_PATH.')
cmd:text()
opt = cmd:parse(arg or {})
assert(opt.root_data_dir ~= '', 'Specify a valid root_data_dir path argument.')


require 'torch'
dofile 'utils/utils.lua'
dofile 'data_gen/parse_wiki_dump/parse_wiki_dump_tools.lua'
tds = tds or require 'tds' 

print('\nComputing Wikipedia p_e_m')

it, _ = io.open(opt.root_data_dir .. 'basic_data/textWithAnchorsFromAllWikipedia2014Feb.txt')
line = it:read()

wiki_e_m_counts = tds.Hash()

-- Find anchors, e.g. <a href="wikt:anarchism">anarchism</a>
local num_lines = 0
local parsing_errors = 0
local list_ent_errors = 0
local diez_ent_errors = 0
local disambiguation_ent_errors = 0
local num_valid_hyperlinks = 0

while (line) do
  num_lines = num_lines + 1
  if num_lines % 5000000 == 0 then
    print('Processed ' .. num_lines .. ' lines. Parsing errs = ' .. 
          parsing_errors .. ' List ent errs = ' .. 
          list_ent_errors .. ' diez errs = ' .. diez_ent_errors .. 
          ' disambig errs = ' .. disambiguation_ent_errors .. 
          ' . Num valid hyperlinks = ' .. num_valid_hyperlinks)
  end

  if not line:find('<doc id="') then
    list_hyp, text, le_errs, p_errs, dis_errs, diez_errs = extract_text_and_hyp(line, false)
    parsing_errors = parsing_errors + p_errs
    list_ent_errors = list_ent_errors + le_errs
    disambiguation_ent_errors = disambiguation_ent_errors + dis_errs
    diez_ent_errors = diez_ent_errors + diez_errs
    for _,el in pairs(list_hyp) do
      local mention = trim1(el.mention)    -- NICK
      if mention ~= "" then
        local ent_wikiid = el.ent_wikiid
      
        -- A valid (entity,mention) pair
        num_valid_hyperlinks = num_valid_hyperlinks + 1

        if not wiki_e_m_counts[mention] then
          wiki_e_m_counts[mention] = tds.Hash()
        end
        if not wiki_e_m_counts[mention][ent_wikiid] then
          wiki_e_m_counts[mention][ent_wikiid] = 0
        end
        wiki_e_m_counts[mention][ent_wikiid] = wiki_e_m_counts[mention][ent_wikiid] + 1
      end
    end
  end
  
  line = it:read()
end

print('    Done computing Wikipedia p(e|m). Num valid hyperlinks = ' .. num_valid_hyperlinks)

print('Now sorting and writing ..')
out_file = opt.root_data_dir .. 'generated/wikipedia_p_e_m.txt'
ouf = assert(io.open(out_file, "w"))

-- NICK print also probabilities
out_file2 = opt.root_data_dir .. 'generated/prob_wikipedia_p_e_m.txt'
ouf2 = assert(io.open(out_file2, "w"))


for mention, list in pairs(wiki_e_m_counts) do
  local tbl = {}
  for ent_wikiid, freq in pairs(list) do
    table.insert(tbl, {ent_wikiid = ent_wikiid, freq = freq})
  end
  table.sort(tbl, function(a,b) return a.freq > b.freq end)

  local str = ''
  local total_freq = 0
  for _,el in pairs(tbl) do
    str = str .. el.ent_wikiid .. ',' .. el.freq  
    str = str .. ',' .. get_ent_name_from_wikiid(el.ent_wikiid):gsub(' ', '_') .. '\t'
    total_freq = total_freq + el.freq
  end
  ouf:write(mention .. '\t' .. total_freq .. '\t' .. str .. '\n')

  -- NICK
  str = ''
  for _,el in pairs(tbl) do
    str = str .. el.ent_wikiid .. ',' .. (el.freq/total_freq)  
    str = str .. ',' .. get_ent_name_from_wikiid(el.ent_wikiid):gsub(' ', '_') .. '\t'
  end
  ouf2:write(mention .. '\t' .. 1 .. '\t' .. str .. '\n')  

end
ouf:flush()
io.close(ouf)

ouf2:flush()
io.close(ouf2)

print('    Done sorting and writing.')



