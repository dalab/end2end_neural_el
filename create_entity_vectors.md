# Creating Entity Vectors
As it is already mentioned, we have created entity vectors for 502661 entities from many
different popular datasets. 

## Description
The paper that describes how the entity vectors are created and the corresponding code can be found 
[here](https://github.com/dalab/deep-ed). That code is written in Lua and Torch, so in the current project 
we provide some wrapper code that helps in creating the entity vectors through Python and for Tensorflow 
and also assists in detecting which entity vectors we need to create for the corpus at hand. But you can
always directly use the original lua code and modify it for your needs.

This code is like glue code that connects this new project with an old one and also provides code
 for processing the different dataset's formats. So it is not something I would recommend to spend time
  on. Specifically, 
the following functionality is provided:
- The first step in creating the entity vectors is to decide which entities we want in our universe. 
  The original Lua code considers only the candidate entities (from the p(e|m) dictionary) for the 
  gold mention of the evaluation datasets  (AIDA, ACE2004, AQUAINT, MSNBC, CLUEWEB). In this project, since
  we work on end to end entity linking we consider all possible text spans (and not only the gold mentions) 
  and for each span we take the first 30 candidate entities (first in terms of p(e|m) score). However, since
  the available datasets are given in different formats we have written code for each one of them:
  - first prepro_aida.py and prepro_other_datasets.py is executed in order to transform all of them in a
  common and simplified format. Then we run the code``preprocessing.prepro_util --create_entity_universe=True`` 
  which processes all these documents (considers all possible spans and for each span all candidate entities and returns back 
  a set of wiki ids i.e. the wiki ids for which we must create entity vectors). Those wiki ids are
  stored in the file data/entities/entities_universe.txt
  - ``gerbil/server.py --build_entity_uninverse=True`` for the format of the all the gerbil datasets. 
    The Gerbil platform reads the different datasets and sends the tested documents in a common format. 
    This code accepts the documents sent by gerbil and does the same processing as described above i.e.
    considers all possible spans and for each span all candidate entities and returns back a set of wiki ids. Those wiki ids
    are stored in the file data/entities/extension_entities/extension_entities.txt

  So depending on the format of the corpus that we are interested in we ran the corresponding code. 
  The gerbil format is actually simple text so you can use this format for arbitrary text, look section
  ["Trying the system on random user input text"](readme.md#Trying-the-system-on-random-user-input-text)
  
  It is advisable to merge the files entities_universe.txt and extension_entities.txt so that we execute the 
  commands of the next section only once and all the entity vectors are trained together.
  
  **Alternative:** directly edit the file data/entities/entities_universe.txt and 
  put there all the wiki ids for which we want entity vectors. 
 
## Commands to be executed for recreating all the entities used for the paper experiments
Creating a set with the wikipedia ids of all the candidate entities from all spans for the
datasets: AIDA-TRAINING, AIDA_TEST-A, AIDA-TEST-B, ACE2004, AQUAINT, MSNBC (file entities_universe.txt is created)
```
EL_PATH=/home/end2end_neural_el
cd end2end_neural_el/
python -m preprocessing.prepro_util --create_entity_universe=True
```

Additional entities for gerbil evaluation:

Now we create another set of wikipedia ids necessary for the evaluation on all the other datasets of the Gerbil platform
(DBpediaSpotlight, Derczynski, ERD2014, GERDAQ-Dev, GERDAQ-Test, GERDAQ-TrainingA, GERDAQ-TrainingB, KORE50, 
Microposts2016-Dev, Microposts2016-Test, Microposts2016-Train, N3-RSS-500, N3-Reuters-128,
OKE 2015 Task1, OKE 2016 Task1). 
<!---
The reason that we split it to two set of entities is so that during training we load only the one set in
order to save memory and we load the second set only during evaluation on these datasets (This provides flexibility 
since we can add new entity vectors on the fly depending on the dataset at hand. However, it 
is advisable to just add the wiki ids from the file data/entities/extension_entities/extension_entities.txt
inside the data/entities/entities_universe.txt to simplify things and run the lua code only once).
--->

Download Gerbil from https://github.com/dice-group/gerbil Find the datasets that you are interested in
and place them in the appropriate locations so that gerbil can use them (follow gerbil instructions).
On one terminal execute:
```
cd gerbil/                         
./start.sh
```
On another terminal execute:
```
cd end2end_neural_el/gerbil-SpotWrapNifWS4Test/
mvn clean -Dmaven.tomcat.port=1235 tomcat:run
```

On a third terminal execute:
```
cd end2end_neural_el/code 
python -m gerbil.server --build_entity_uninverse=True
```
Open the url http://localhost:1234/gerbil
- Configure experiment
- In URI field write: http://localhost:1235/gerbil-spotWrapNifWS4Test/myalgorithm
- Name: whatever you wish
- Press add annotator button
- Select datasets that you wish to create entity vectors
- Run experiment
- Terminate with ctrl+C the third terminal (the one that runs the command: python -m gerbil.server)
         when all the datasets have been processed (you can see the results on the gerbil web-page)
         Here instead of replying we just process the text the gerbil sends to extract the
         candidate entities and we always return an empty response. So the scores will be zero.
(file extension_entities.txt has been created). 

We can now merge the files entities_universe.txt and extension_entities.txt to simplify the process and train
them all together. Alternatively, we can train the two sets of entity vectors separately and use the extension entities
on the fly only when needed for the evaluation on the specific datasets.

Now create the entity vectors:
-	Follow instructions from https://github.com/dalab/deep-ed
-   the code of this project is under end2end_neural_el/deep-ed/
	There are only some minor changes at the deep-ed-master/entities/relatedness/relatedness.lua
	to read our file of entities (the entities for whom we want to create the entity vectors). 
	Replace lines 277-278 with 280-281 for the extension entities.
	 
<!---
	 (i.e. the following instructions will be executed twice).
    Option 1: merge the normal set of entities with the extension entities and train them all
    together (this approach is easier and cleaner). Option 2: train the main set of entities and
    the extension entities separately. The advantages of the second approach is that in this
    way we can have a small set of entities (e.g. only from AIDA) for training thus saving
    gpu memory (so possibly increasing batch size) and using the extension entities together
    with the normal set only for the gerbil evaluation (where we process each document separately anyway
    i.e. batch size = 1 due to the way gerbil works). Also this functionality in terms of code it provides
    some flexibility since you can always change the extra entities and use the system to a different dataset.
--->

After you follow the instructions from https://github.com/dalab/deep-ed do the following in 
order to get the entity vectors from the torch-lua framework to python-tensorflow
- pick the epoch that has the best scores and copy this file to our project
```
cp $EL_PATH/deep-ed/data/generated/ent_vecs/ent_vecs__ep_147.t7  $EL_PATH/data/entities/ent_vecs/       
```
- converts the file to simple txt format
```
end2end_neural_el/code$ th preprocessing/bridge_code_lua/ent_vecs_to_txt.lua -ent_vecs_filepath /home/end2end_neural_el/data/entities/ent_vecs/ent_vecs__ep_147.t7
```
- and then to numpy array ready to be used from tensorflow
```
end2end_neural_el/code$ python -m preprocessing.bridge_code_lua.ent_vecs_from_txt_to_npy
```

Now the same for the extension entities (optional steps):
- Replace lines 277-278 with 280-281 in relatedness.lua file.
- Delete the t7 files, otherwise the changed relatedness.lua script will not be executed 
(in order to work on the extension entities this time) 
```
rm end2end_neual_el/deep-ed/data/generated/*t7
```
- Repeat the steps of https://github.com/dalab/deep-ed starting from step 13 (no need to repeat the previous ones)
- After the training is finished execute the following commands (similar to before)
```
mkdir -p $EL_PATH/data/entities/extension_entities/ent_vecs/
cp $EL_PATH/deep-ed/data/generated/ent_vecs/ent_vecs__ep_54.t7  $EL_PATH/data/entities/extension_entities/ent_vecs/
end2end_neural_el/code$ th preprocessing/bridge_code_lua/ent_vecs_to_txt.lua -ent_vecs_filepath /home/end2end_neural_el/data/entities/extension_entities/ent_vecs/ent_vecs__ep_54.t7     
end2end_neural_el/code$ python -m preprocessing.bridge_code_lua.ent_vecs_from_txt_to_npy --entity_extension=True
```

## Files Description

***In data/entities/:

- entities_universe.txt: This file contains the wiki ids of the entities that we want to build entity vectors.
 In our case we were creating this file with the command ``python -m preprocessing.prepro_util --create_entity_universe=True``
 i.e. top 30 candidate entities for all spans of the datasets AIDA-TRAINING, AIDA_TEST-A, AIDA-TEST-B, ACE2004, AQUAINT, MSNBC.
 This file in our case contains 470105 wiki ids <sup>1</sup>. The deep-ed (lua) project then reads this file and 
 create entity vectors for all of them plus some extra entities used for the evaluation of the training procedure
 (Ceccarelli et al., 2013) (deep-ed-master/entities/relatedness/relatedness.lua) and hence sum to 484048 entities. When we
 run the Gerbil evaluation we add some "extension entities" necessary for the rest of the datasets, so we add 18613 for a
 total of 502661 entities. 
 
 <sup>1</sup> These entities were created a year ago, but if I remember correctly, the lines 719 and 721 were: 
 gmonly_files = ['aida_train.txt']
 allspans_files = ['aida_dev.txt', 'aida_test.txt', 'ace2004.txt', 'aquaint.txt', 'clueweb.txt', 'msnbc.txt', 'wikipedia.txt']

- wikiid2nnid/*.txt (484,048 lines each): maps between wikiids and nnids (rows in the embedding array). 


***in data/basic_data/:
- wiki_name_id_map.txt (4’399,390 lines): a list of wiki_name – wiki_id pairs for *all* the 
wikipages in the dump, but not including the disambiguation and redirect pages.
The disambiguation and redirect pages are not included since the result of our algorithm should be a concrete and 
specific entity not a disambiguation page.
- prob_yago_crosswikis_wikipedia_p_e_m.txt (21’587,744 lines): 
 This pem dictionary is created by the whole wikipedia dump + crosswiki dataset + yago dataset.
For more details look at [this](https://arxiv.org/pdf/1704.04920.pdf) paper section 6 "Candidate Selection".  
"It is computed by averaging probabilities from two indexes build from mention entity hyperlink count 
statistics from Wikipedia and a large Web corpus (Spitkovsky and Chang, 2012). Moreover, we 
add the YAGO dictionary of (Hoffart et al., 2011), where each candidate receives a uniform prior."
Also in terms of code it is implemented in the following files    deep-ed/deep-ed-master/data_gen/gen_p_e_m/.
For instructions of executing this code please look at [this](https://github.com/dalab/deep-ed) link




