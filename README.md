# Question Answering on HetioNet

This is a fork of [EmbedKGQA](https://github.com/malllabiisc/EmbedKGQA).

In order to replicate, please download [LibKGE](https://github.com/uma-pi1/kge) and train KG Embeddings on the [HetioNet KG](https://github.com/hetio/hetionet) in tsv format.

Make a pretrained_models folder. Add an embeddings folder to it. In that, copy the hetionet dataset folder (named as HetioNet) that must have been made to run LibKGE on HetioNet in this folder. Also add the checkpoint_best.pt file inside the HetioNet folder from the LibKGE training folder.


## Training

Change to directory ./KGQA/RoBERTa. Following is an example command to run the QA training code

```
python main.py --mode train --relation_dim 50 --do_batch_norm 1 --gpu 0 --freeze 1 --batch_size 32 --validate_every 10 --hops webqsp_half --lr 0.00004 --entdrop 0.0 --reldrop 0.0 --scoredrop 0.0 --decay 1.0 --model ComplEx --patience 20 --ls 0.05 --l3_reg 0.001 --nb_epochs 200 --outfile half_fbwq
```

The best weights will be stored in ./KGQA/RoBERTa/best_score_model.pt

## Changing RoBERTa to BioBERT
Simply replace the model and dataloader files in ./KGQA/RoBERTa with the files of the same name in ./KGQA/RoBERTa/api (The files are the same barring 2-3 lines at the beginning).


## Running the Demo
Make sure to install [spacy](https://spacy.io/usage), and [FastAPI](https://towardsdatascience.com/how-to-deploy-a-machine-learning-model-with-fastapi-docker-and-github-actions-13374cbd638a).

Change to directory ./KGQA/RoBERTa/api. Copy the pretrained_models file in this folder and add the QA trained weights files in ./KGQA/RoBERTa/api/ml in the format modelname_model.pt where modelname will be roberta/biobert

Run this command
```
uvicorn main:app --reload
```
