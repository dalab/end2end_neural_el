# End-to-End Neural Entity Linking (CoNLL 2018, full paper)
### Python source code


1: Setting up the environment.
```
git clone arxiv_url
cd end2end_neural_el
python3 -m pip install virtualenv
python3 -m virtualenv end2end_neural_el_env
source end2end_neural_el_env/bin/activate
pip install -r requirements.txt
```

Download the 'data' folder from [this link](https://drive.google.com/file/d/1OSKvIiXHVVaWUhQ1-fpvePTBQfgMT6Ps/view?usp=sharing), unzip it and place it under end2end_neural_el/ 
These data are enough for running the pretrained models and reproducing the results. 
For reproducing the results you can skip all the next steps and go directly to the 
[gerbil evaluation section](#gerbil-evaluation).

If you want to create new models with different preprocessing 
(or run the preprocessing steps described below)
you should also download the pre-trained Word2Vec vectors 
[GoogleNews-vectors-negative300.bin.gz](https://code.google.com/archive/p/word2vec/). 
Unzip it and place the bin file in the folder end2end_neural_el/data/basic_data/wordEmbeddings/Word2Vec.

If you also want to train your own entity vectors then follow the instructions from [here](https://github.com/dalab/deep-ed).
The 'basic_data' folder this time will be placed under end2end_neural_el/deep-ed/data/

2: Preprocessing stage 1.

Transform the AIDA, ACE2004, AQUAINT, MSNBC, CLUEWEB datasets to a common easy to use format.
Only AIDA train is used by our system for training, and the AIDA-TESTA for hyperparameter tuning.
```
cd code
python -m preprocessing.prepro_aida
python -m preprocessing.prepro_other_datasets
```

3: Preprocessing stage 2 (converting datasets to tfrecords).

This step requires the entity vectors and the word-embeddings to exist.
An essential part of our system are the entity vectors (the equivalent of word-embeddings for entities). 
You can create your entity vectors by following the instructions of the [next chapter](#gerbil-evaluation), otherwise you can use
the provided pretrained ones. We have pretrained  502661 entity vectors. Specifically, 
we have trained entity vectors for all the candidate entities from all possible spans of 
AIDA-TestA, AIDA-TestB, AIDA-Training <sup>1</sup>, ACE2004, AQUAINT, MSNBC, Clueweb, DBpediaSpotlight, Derczynski, 
ERD2014, GERDAQ-Dev, GERDAQ-Test, GERDAQ-TrainingA, GERDAQ-TrainingB, KORE50, 
Microposts2016-Dev, Microposts2016-Test, Microposts2016-Train, N3-RSS-500, N3-Reuters-128,
OKE 2015 Task1, OKE 2016 Task1, and the entity relatedness dataset of (Ceccarelli et al., 2013). In more detail, 
this is done by considering all possible spans
of the document as a candidate span and querying our p(e|m) dictionary for all the candidate entities
for this span (we keep only the top 30 for each candidate span). 

<sup>1</sup> For AIDA-Training 10% of the candidate entities that are detected from the above
 method are missing from the current set of pretrained entities. So in case you want to evaluate
 the algorithm on AIDA-Training for EL this extra entities are needed.

We now encode the words and characters of the datasets to numbers, we find the candidate spans and 
for each one the candidate entities (using the p(e|m) dictionary) and
encode these to numbers as well.
```
python -m preprocessing.prepro_util
```

4: Training the Neural Network.

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
                    --global_thr=0.001 --global_score_ffnn=0_0           
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


Base Model + att + global for ED  (the only difference from the corresponding command
for the EL model is the absence of the argument --all_spans_training=True which means that we train
exactly the same architecture but now only with the gold mentions).
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
                    --fast_evaluation=True  \
                    --attention_ent_vecs_no_regularization=True --final_score_ffnn=0_0 \
                    --attention_R=10 --attention_K=100 \
                    --train_datasets=aida_train \
                    --ed_datasets=aida_dev_z_aida_test_z_aida_train --ed_val_datasets=0 \
                    --global_thr=0.001 --global_score_ffnn=0_0           
done
```

When running multiple experiments with multiple different hyperparameters we want an easy way to evaluate
the different models. The following command goes through the log files of all the models and sorts them
based on the performance on a selected dataset (here the AIDA-TestA).
```
cd evaluation; python summarize_all_experiments.py --macro_or_micro=micro --dev_set=aida_dev --test_set=aida_test
```

# Gerbil evaluation  
On one terminal run [Gerbil](https://github.com/dice-group/gerbil). Execute:
```
cd gerbil/                         
./start.sh
```
Caution: Gerbil might be incompatible with some java versions. With Java 8 it works.

On another terminal execute:
```
cd end2end_neural_el/gerbil-SpotWrapNifWS4Test/
mvn clean -Dmaven.tomcat.port=1235 tomcat:run
```

On a third terminal execute:
```
cd end2end_neural_el/code 
For EL:
python -m gerbil.server --training_name=base_att_global --experiment_name=paper_models   \
           --persons_coreference_merge=True --all_spans_training=True --entity_extension=extension_entities

For ED:
python -m gerbil.server --training_name=base_att_global --experiment_name=paper_models   \
           --persons_coreference_merge=True --ed_mode --entity_extension=extension_entities
```
By changing the training_name parameter you can try the different models 
(base_att_global, base_att, basemodel) and reproduce the results shown in the paper (tables 2, 7, and 8).

Open the url http://localhost:1234/gerbil
- Configure experiment
- In URI field write: http://localhost:1235/gerbil-spotWrapNifWS4Test/myalgorithm
- Name: whatever you wish
- Press add annotator button
- Select the datasets that you want to evaluate the model on
- Run experiment



For **local evaluation and for printing the annotated datasets** run the following command:
(reproduces the results of the table 9 of the paper)
```
For EL:
python -m model.evaluate --training_name=base_att_global  --experiment_name=paper_models --entity_extension=extension_entities  --el_datasets=aida_dev_z_aida_test_z_aida_train_z_ace2004_z_aquaint_z_clueweb_z_msnbc_z_wikipedia    --el_val_datasets=0  --ed_datasets=""  --ed_val_datasets=0   --all_spans_training=True
For ED:
python -m model.evaluate --training_name=base_att_global  --experiment_name=paper_models --entity_extension=extension_entities  --ed_datasets=aida_dev_z_aida_test_z_aida_train_z_ace2004_z_aquaint_z_clueweb_z_msnbc_z_wikipedia    --ed_val_datasets=0  --el_datasets=""  --el_val_datasets=0
```
The local evaluation for EL is almost identical to gerbil scores but for ED it is higher. This should
probably be attributed to some parsing errors and different preprocessing when input is from gerbil in
comparison to local evaluation that is done on the official tokenized datasets.

# Trying the system on random user input text
If you want to try the system on custom input it can be done in multiple ways but the simplest is
following: 
On one terminal run the command (similar to the Gerbil evaluation)
```
cd end2end_neural_el/code 
For EL:
python -m gerbil.server --training_name=base_att_global --experiment_name=paper_models   \
           --persons_coreference_merge=True --all_spans_training=True --entity_extension=extension_entities

For ED:
python -m gerbil.server --training_name=base_att_global --experiment_name=paper_models   \ 
           --persons_coreference_merge=True --ed_mode --entity_extension=extension_entities
```
This launches a server that expects to receive json objects of the following format:
```
{ "text": "Obama will visit Germany and have a meeting with Merkel tomorrow.", 
"spans": [{"start":0,"length":5}, {"start":17,"length":7}, {"start":49,"length":6}]  }
```
For the EL task of course the "spans" field will be an **empty array** e.g.
```
{ "text": "Obama will visit Germany and have a meeting with Merkel tomorrow.", "spans": []  }
```
In another terminal you can submit your query in the following way:
In python:
```
import requests, json
myjson = { "text": "Obama will visit Germany and have a meeting with Merkel tomorrow.", "spans": [{"start":0,"length":5}, {"start":17,"length":7}, {"start":49,"length":6}]  }
myjson = { "text": "Obama will visit Germany and have a meeting with Merkel tomorrow.", "spans": []  }
requests.post("http://localhost:5555", json=myjson)
```
From the terminal directly with curl command:
```
curl -X POST --header 'Content-Type: application/json' --header 'Accept: application/json' -d "{ \"text\": \"Obama will visit Germany and have a meeting with Merkel tomorrow.\", \"spans\": [{\"start\":0,\"length\":5}, {\"start\":17,\"length\":7}, {\"start\":49,\"length\":6}]  }" 'http://localhost:5555'
```
The server's terminal prints the result on the screen e.g.
```
[(17, 7, 'Germany'), (0, 5, 'Barack_Obama'), (49, 6, 'Angela_Merkel')]
```
The third value of the tuple is the wikipedia title. So to obtain a hyperlink add the prefix 
 'https://en.wikipedia.org/wiki/' e.g. https://en.wikipedia.org/wiki/Barack_Obama

Other ways to evaluate your corpus is to publish your documents in the supported formats i.e.
publish them a) in the same format as the files in the data/new_datasets folder b) in the same 
format as the AIDA dataset and then run the prepro_aida.py c) in the format of ace2004, aquaint, msnbc
etc datasets (look folder data/basic_data/test_datasets/wned-datasets)  

# Creating Entity Vectors
Detailed instructions for creating entity vectors can be found [here](create_entity_vectors.md). 

