if not opt then
  cmd = torch.CmdLine()
  cmd:option('-root_data_dir', '', 'Root path of the data, $DATA_PATH.')
  cmd:text()
  opt = cmd:parse(arg or {})
  assert(opt.root_data_dir ~= '', 'Specify a valid root_data_dir path argument.')  
end


require 'torch'
dofile 'utils/utils.lua'
dofile 'data_gen/parse_wiki_dump/parse_wiki_dump_tools.lua'
dofile 'entities/ent_name2id_freq/e_freq_index.lua'
tds = tds or require 'tds' 

print('\nExtracting text only from Wiki dump. Output is wiki_canonical_words.txt containing on each line an Wiki entity with the list of all words in its canonical Wiki page.')

it, _ = io.open(opt.root_data_dir .. 'basic_data/textWithAnchorsFromAllWikipedia2014Feb.txt')

out_file = opt.root_data_dir .. 'generated/wiki_canonical_words.txt'
ouf = assert(io.open(out_file, "w"))

line = it:read()

-- Find anchors, e.g. <a href="wikt:anarchism">anarchism</a>
local num_lines = 0
local num_valid_ents = 0
local num_error_ents = 0 -- Probably list or disambiguation pages.

local empty_valid_ents = get_map_all_valid_ents()

local cur_words = ''
local cur_ent_wikiid = -1

while (line) do
  num_lines = num_lines + 1
  if num_lines % 5000000 == 0 then
    print('Processed ' .. num_lines .. ' lines. Num valid ents = ' .. num_valid_ents .. '. Num errs = ' .. num_error_ents)
  end

  if (not line:find('<doc id="')) and (not line:find('</doc>')) then
    _, text, _, _ , _ , _ = extract_text_and_hyp(line, false)
    local words = split_in_words(text)
    cur_words = cur_words .. table.concat(words, ' ') .. ' '
    
  elseif line:find('<doc id="') then
    if (cur_ent_wikiid > 0 and cur_words ~= '') then
      if cur_ent_wikiid ~= unk_ent_wikiid and is_valid_ent(cur_ent_wikiid) then
        ouf:write(cur_ent_wikiid .. '\t' .. get_ent_name_from_wikiid(cur_ent_wikiid) .. '\t' .. cur_words .. '\n')
        empty_valid_ents[cur_ent_wikiid] = nil
        num_valid_ents = num_valid_ents + 1
      else
        num_error_ents = num_error_ents + 1
      end
    end
    
    cur_ent_wikiid = extract_page_entity_title(line)
    cur_words = ''
  end
  
  line = it:read()
end
ouf:flush()
io.close(ouf)

-- Num valid ents = 4126137. Num errs = 332944
print('    Done extracting text only from Wiki dump. Num valid ents = ' .. num_valid_ents .. '. Num errs = ' .. num_error_ents)


print('Create file with all entities with empty Wikipedia pages.')
local empty_ents = {}
for ent_wikiid, _ in pairs(empty_valid_ents) do
  table.insert(empty_ents, {ent_wikiid = ent_wikiid, f = get_ent_freq(ent_wikiid)})
end
table.sort(empty_ents, function(a,b) return a.f > b.f end)

local ouf2 = assert(io.open(opt.root_data_dir .. 'generated/empty_page_ents.txt', "w"))
for _,x in pairs(empty_ents) do
  ouf2:write(x.ent_wikiid .. '\t' .. get_ent_name_from_wikiid(x.ent_wikiid) .. '\t' .. x.f .. '\n')
end
ouf2:flush()
io.close(ouf2)
print('    Done')