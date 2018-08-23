# End-to-End Neural Entity Linking (CoNLL 2018, full paper)
### Python source code


1: Setting up the environment:
```
git clone arxiv_url
cd end2end_neural_el
python3 -m pip install virtualenv
python3 -m virtualenv end2end_neural_el_env
source end2end_neural_el_env/bin/activate
pip install -r requirements.txt
```

TODO upload the data folders, the pretrained model, and the entity vectors and provide instructions
on where they should be placed.

2: Preprocessing stage 1:
Transform the AIDA, ACE2004, AQUAINT, MSNBC, CLUEWEB datasets to a common easy to use format.
Only AIDA train is used by our system for training, and the AIDA-TESTA for hyperparameter tuning.
```
cd code
python -m preprocessing.prepro_aida
python -m preprocessing.prepro_other_datasets
```

3: An essential part of our system are the entity vectors (the equivalent of word-embeddings for entities). 
You can create your entity vectors by following the instructions of the next chapter, otherwise you can use
the provided pretrained ones. We have pretrained 322245 (from all spans of AIDA-Training, AIDA-TestA, 
AIDA-TestB, ACE2004, AQUAINT, MSNBC) + 44010 (from all spans of DBpediaSpotlight, Derczynski, 
ERD2014, GERDAQ-Dev, GERDAQ-Test, GERDAQ-TrainingA, GERDAQ-TrainingB, KORE50, 
Microposts2016-Dev, Microposts2016-Test, Microposts2016-Train, N3-RSS-500, N3-Reuters-128,
OKE 2015 Task1, OKE 2016 Task1). Specifically, this is done by considering all possible spans
of the document as a candidate span and querying our p(e|m) dictionary for all the candidate entities
for this span (we keep only the top 30 for each candidate span). 

Preprocessing stage 2: converting the datasets to tfrecords (we encode the words and characters
to numbers, we find the candidate spans and for each one the candidate entities (using the p(e|m) dictionary) and
encode these to numbers as well).
```
python -m preprocessing.prepro_util
```

4: Training the Neural Network:

Base Model + att + global for EL (according to the paper)
```
for v in 1 2 3
do
bsub -n 2 -W 24:00 -R "rusage[mem=10000,ngpus_excl_p=1]" python3 -m model.train  \
                    --batch_size=4   --experiment_name=corefmerge \
                    --training_name=group_global/global_model_v$v \
                    --ent_vecs_regularization=l2dropout  --evaluation_minutes=10 --nepoch_no_imprv=6 \
                    --span_emb="boundaries"  \
                    --dim_char=50 --hidden_size_char=50 --hidden_size_lstm=150 \
                    --nn_components=pem_lstm_attention_global \
                    --fast_evaluation=True  --all_spans_training=True \
                    --attention_ent_vecs_no_regularization=True --final_score_ffnn=0_0 \
                    --attention_R=10 --attention_K=100 \
                    --train_datasets=aida_train \
                    --el_datasets=aida_dev_z_aida_test_z_aida_train --el_val_datasets=0 \
                    --global_thr=0.0 --global_score_ffnn=0_0           
done
```

Base Model + att + global (top2) for EL. It is exactly the same with the previous one except it has one more condition in 
the global component. Specifically, at most two candidate entities from each candidate span
participate in the global voting even if more have local score above the threshold \gamma'.
```
for v in 1 2 3
do
bsub -n 2 -W 24:00 -R "rusage[mem=10000,ngpus_excl_p=1]" python3 -m model.train  \
                    --batch_size=4   --experiment_name=corefmerge \
                    --training_name=group_global/global_model_top2$v \
                    --ent_vecs_regularization=l2dropout  --evaluation_minutes=10 --nepoch_no_imprv=6 \
                    --span_emb="boundaries"  \
                    --dim_char=50 --hidden_size_char=50 --hidden_size_lstm=150 \
                    --nn_components=pem_lstm_attention_global \
                    --fast_evaluation=True --all_spans_training=True \
                    --attention_ent_vecs_no_regularization=True --final_score_ffnn=0_0 \
                    --attention_R=10 --attention_K=100 \
                    --global_topk=2 --global_topkthr=0.001 --global_norm_or_mean=norm   \
                    --global_gmask_based_on_localscore=True --global_score_ffnn=0_0\
                    --train_datasets=aida_train \
                    --el_datasets=aida_dev_z_aida_test_z_aida_train \
                    --el_val_datasets=0     --improvement_threshold=0.1  --ablations=True
done
```

Base Model for EL (according to the paper):
```
for v in 1 2 3
do
bsub -n 2 -W 24:00 -R "rusage[mem=10000,ngpus_excl_p=1]" python3 -m model.train  \
                    --batch_size=4   --experiment_name=corefmerge \
                    --training_name=group_lstm/base_model_v$v \
                    --ent_vecs_regularization=l2dropout  --evaluation_minutes=10 --nepoch_no_imprv=6 \
                    --span_emb="boundaries"  \
                    --dim_char=50 --hidden_size_char=50 --hidden_size_lstm=150 \
                    --nn_components=pem_lstm \
                    --fast_evaluation=True  --all_spans_training=True \
                    --final_score_ffnn=0_0 \
                    --train_datasets=aida_train \
                    --el_datasets=aida_dev_z_aida_test_z_aida_train --el_val_datasets=0           
done
```


Base Model + att + global (top-2) for ED  (the only difference from the corresponding command
for the EL model is the absence of the argument --all_spans_training=True which means that we train
exactly the same architecture but now only with the gold mentions).
```
for v in 1 2 3
do
bsub -n 2 -W 24:00 -R "rusage[mem=10000,ngpus_excl_p=1]" python3 -m model.train  \
                    --batch_size=4   --experiment_name=corefmerge \
                    --training_name=group_20_8/global_top2norm_localscoregmaskscalingv$v \
                    --ent_vecs_regularization=l2dropout  --evaluation_minutes=10 --nepoch_no_imprv=6 \
                    --span_emb="boundaries"  \
                    --dim_char=50 --hidden_size_char=50 --hidden_size_lstm=150 \
                    --nn_components=pem_lstm_attention_global \
                    --fast_evaluation=True \
                    --attention_ent_vecs_no_regularization=True --final_score_ffnn=0_0 \
                    --attention_R=10 --attention_K=100 \
                    --global_topk=2 --global_topkthr=0.001 --global_norm_or_mean=norm   \
                    --global_gmask_based_on_localscore=True --global_score_ffnn=0_0\
                    --train_datasets=aida_train \
                    --ed_datasets=aida_dev_z_aida_test_z_aida_train \
                    --ed_val_datasets=0     --improvement_threshold=0.1   --new_all_voters_emb=True      --ablations=True
done
```

### Gerbil evaluation
TODO write instructions


# Creating Entity Vectors
As it is already mentioned, we have created entity vectors for 366255 entities from many
different popular datasets. 

Creating a set with the wikipedia ids of all the candidate entities from all spans for the
datasets: AIDA-TRAINING, AIDA_TEST-A, AIDA-TEST-B, ACE2004, AQUAINT, MSNBC
```
EL_PATH=/home/end2end_neural_el
cd end2end_neural_el/
python -m preprocessing.prepro_util --create_entity_universe=True
```

Additional entities for gerbil evaluation:
Now we create another set of entity vectors (for all the other gerbil datasets). The reason 
that we split it to two set of entities is so that during training we load only the one set in
order to save memory and we load the second set only during evaluation on these datasets.
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

Now create the entity vectors:
-	Follow instructions from https://github.com/dalab/deep-ed
-   the code of this project is under end2end_neural_el/deep-ed/
	There are only some minor changes at the deep-ed-master/entities/relatedness/relatedness.lua
	to read our file of entities (the entities for whom we want to create the entity vectors). 
	Replace lines 277-278 with 280-281 for the extension entities (i.e. the following instructions will be executed twice).
    Option 1: merge the normal set of entities with the extension entities and train them all
    together (this approach is easier and cleaner). Option 2: train the main set of entities and
    the extension entities separately. The advantages of the second approach is that in this
    way we can have a small set of entities (e.g. only from AIDA) for training thus saving
    gpu memory (so possibly increasing batch size) and using the extension entities together
    with the normal set only for the gerbil evaluation (where we process each document separately anyway
    i.e. batch size = 1 due to the way gerbil works). Also this functionality in terms of code it provides
    some flexibility since you can always change the extra entities and use the system to a different dataset.

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

