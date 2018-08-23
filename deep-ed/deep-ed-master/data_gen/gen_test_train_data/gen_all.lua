-- Generates all training and test data for entity disambiguation.

if not ent_p_e_m_index then
  require 'torch'
  dofile 'data_gen/indexes/wiki_redirects_index.lua'
  dofile 'data_gen/indexes/yago_crosswikis_wiki.lua'
  dofile 'utils/utils.lua'
  tds = tds or require 'tds'
end

dofile 'data_gen/gen_test_train_data/gen_aida_test.lua'
dofile 'data_gen/gen_test_train_data/gen_aida_train.lua'
dofile 'data_gen/gen_test_train_data/gen_ace_msnbc_aquaint_csv.lua'