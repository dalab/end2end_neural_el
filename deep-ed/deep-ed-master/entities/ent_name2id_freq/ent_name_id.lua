------------------ Load entity name-id mappings ------------------
-- Each entity has:
--   a) a Wikipedia URL referred as 'name' here
--   b) a Wikipedia ID referred as 'ent_wikiid' or 'wikiid' here
--   c) an ID that will be used in the entity embeddings lookup table. Referred as 'ent_thid' or 'thid' here.

tds = tds or require 'tds' -- saves lots of memory for ent_name_id.lua. Mem overflow with normal {}
local rltd_only = false
if opt and opt.entities and opt.entities ~= 'ALL' then
  assert(rewtr.reltd_ents_wikiid_to_rltdid, 'Import relatedness.lua before ent_name_id.lua')
  rltd_only = true
end

-- Unk entity wikid:
unk_ent_wikiid = 1

local entity_wiki_txtfilename = opt.root_data_dir .. 'basic_data/wiki_name_id_map.txt'
local entity_wiki_t7filename = opt.root_data_dir .. 'generated/ent_name_id_map.t7'
if rltd_only then
  entity_wiki_t7filename = opt.root_data_dir .. 'generated/ent_name_id_map_RLTD.t7'
end

print('==> Loading entity wikiid - name map') 

local e_id_name = nil

if paths.filep(entity_wiki_t7filename) then
  print('  ---> from t7 file: ' .. entity_wiki_t7filename)
  e_id_name = torch.load(entity_wiki_t7filename)

else
  print('  ---> t7 file NOT found. Loading from disk (slower). Out f = ' .. entity_wiki_t7filename)
  dofile 'data_gen/indexes/wiki_disambiguation_pages_index.lua'
  print('    Still loading entity wikiid - name map ...') 
  
  e_id_name = tds.Hash()
  
  -- map for entity name to entity wiki id
  e_id_name.ent_wikiid2name = tds.Hash()
  e_id_name.ent_name2wikiid = tds.Hash()

  -- map for entity wiki id to tensor id. Size = 4.4M
  if not rltd_only then
    e_id_name.ent_wikiid2thid = tds.Hash()
    e_id_name.ent_thid2wikiid = tds.Hash()
  end
  
  local cnt = 0
  local cnt_freq = 0
  for line in io.lines(entity_wiki_txtfilename) do
    local parts = split(line, '\t')
    local ent_name = parts[1]
    local ent_wikiid = tonumber(parts[2])      
    
    if (not wiki_disambiguation_index[ent_wikiid]) then
      if (not rltd_only) or rewtr.reltd_ents_wikiid_to_rltdid[ent_wikiid] then
        e_id_name.ent_wikiid2name[ent_wikiid] = ent_name
        e_id_name.ent_name2wikiid[ent_name] = ent_wikiid
      end
      if not rltd_only then
        cnt = cnt + 1
        e_id_name.ent_wikiid2thid[ent_wikiid] = cnt
        e_id_name.ent_thid2wikiid[cnt] = ent_wikiid    
      end
    end
  end

  if not rltd_only then
    cnt = cnt + 1
    e_id_name.ent_wikiid2thid[unk_ent_wikiid] = cnt
    e_id_name.ent_thid2wikiid[cnt] = unk_ent_wikiid
  end
  e_id_name.ent_wikiid2name[unk_ent_wikiid] = 'UNK_ENT'
  e_id_name.ent_name2wikiid['UNK_ENT'] = unk_ent_wikiid

  torch.save(entity_wiki_t7filename, e_id_name)
end

if not rltd_only then
  unk_ent_thid = e_id_name.ent_wikiid2thid[unk_ent_wikiid]
else
  unk_ent_thid = rewtr.reltd_ents_wikiid_to_rltdid[unk_ent_wikiid]
end

------------------------ Functions for wikiids and names-----------------
function get_map_all_valid_ents()
  local m = tds.Hash()
  for ent_wikiid, _ in pairs(e_id_name.ent_wikiid2name) do
    m[ent_wikiid] = 1
  end
  return m  
end

is_valid_ent = function(ent_wikiid)
  if e_id_name.ent_wikiid2name[ent_wikiid] then
    return true
  end
  return false
end


get_ent_name_from_wikiid = function(ent_wikiid)
  local ent_name = e_id_name.ent_wikiid2name[ent_wikiid]
  if (not ent_wikiid) or (not ent_name) then
    return 'NIL'
  end
  return ent_name
end

preprocess_ent_name = function(ent_name)
  ent_name = trim1(ent_name)
  ent_name = string.gsub(ent_name, '&amp;', '&')
  ent_name = string.gsub(ent_name, '&quot;', '"')
  ent_name = ent_name:gsub('_', ' ')
  ent_name = first_letter_to_uppercase(ent_name)
  if get_redirected_ent_title then
    ent_name = get_redirected_ent_title(ent_name)
  end
  return ent_name
end

get_ent_wikiid_from_name = function(ent_name, not_verbose)
  local verbose = (not not_verbose)
  ent_name = preprocess_ent_name(ent_name)
  local ent_wikiid = e_id_name.ent_name2wikiid[ent_name]
  if (not ent_wikiid) or (not ent_name) then
    if verbose then
      print(red('Entity ' .. ent_name .. ' not found. Redirects file needs to be loaded for better performance.'))
    end
    return unk_ent_wikiid
  end
  return ent_wikiid
end

------------------------ Functions for thids and wikiids -----------------
-- ent wiki id -> thid
get_thid = function (ent_wikiid)
  if rltd_only then
    ent_thid = rewtr.reltd_ents_wikiid_to_rltdid[ent_wikiid]
  else
    ent_thid = e_id_name.ent_wikiid2thid[ent_wikiid]
  end
  if (not ent_wikiid) or (not ent_thid) then
    return unk_ent_thid
  end
  return ent_thid
end

contains_thid = function (ent_wikiid)
  if rltd_only then
    ent_thid = rewtr.reltd_ents_wikiid_to_rltdid[ent_wikiid]
  else
    ent_thid = e_id_name.ent_wikiid2thid[ent_wikiid]
  end
  if ent_wikiid == nil or ent_thid == nil then
    return false
  end
  return true
end

get_total_num_ents = function()
  if rltd_only then
    assert(table_len(rewtr.reltd_ents_wikiid_to_rltdid) == rewtr.num_rltd_ents)
    return table_len(rewtr.reltd_ents_wikiid_to_rltdid)
  else
    return #e_id_name.ent_thid2wikiid
  end
end
  
get_wikiid_from_thid = function(ent_thid)
  if rltd_only then
    ent_wikiid = rewtr.reltd_ents_rltdid_to_wikiid[ent_thid]
  else
    ent_wikiid = e_id_name.ent_thid2wikiid[ent_thid]
  end  
  if ent_wikiid == nil or ent_thid == nil then
    return unk_ent_wikiid
  end
  return ent_wikiid
end

-- tensor of ent wiki ids --> tensor of thids
get_ent_thids = function (ent_wikiids_tensor)
  local ent_thid_tensor = ent_wikiids_tensor:clone()
  if ent_wikiids_tensor:dim() == 2 then
    for i = 1,ent_thid_tensor:size(1) do
      for j = 1,ent_thid_tensor:size(2) do
        ent_thid_tensor[i][j] = get_thid(ent_wikiids_tensor[i][j])
      end
    end
  elseif ent_wikiids_tensor:dim() == 1 then
    for i = 1,ent_thid_tensor:size(1) do
      ent_thid_tensor[i] = get_thid(ent_wikiids_tensor[i])
    end
  else
    print('Tensor with > 2 dimentions not supported')
    os.exit()
  end
  return ent_thid_tensor
end

print('    Done loading entity name - wikiid. Size thid index = ' .. get_total_num_ents())
