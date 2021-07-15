import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pickle
from tqdm import tqdm
import argparse
import operator
from torch.nn import functional as F
from dataloader import DatasetMetaQA, DataLoaderMetaQA
from model import RelationExtractor
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
import networkx as nx
import time
import sys
sys.path.append("../..") # Adds higher directory to python modules path.
from kge.model import KgeModel
from kge.util.io import load_checkpoint
from copy import deepcopy
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from enum import Enum

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        return True

def prepare_embeddings(embedding_dict):
    entity2idx = {}
    idx2entity = {}
    i = 0
    embedding_matrix = []
    for key, entity in embedding_dict.items():
        entity2idx[key] = i
        idx2entity[i] = key
        i += 1
        embedding_matrix.append(entity)
    return entity2idx, idx2entity, embedding_matrix

def get_vocab(data):
    word_to_ix = {}
    maxLength = 0
    idx2word = {}
    for d in data:
            sent = d[1]
            for word in sent.split():
                if word not in word_to_ix:
                    idx2word[len(word_to_ix)] = word
                    word_to_ix[word] = len(word_to_ix)
                    
            length = len(sent.split())
            if length > maxLength:
                maxLength = length

    return word_to_ix, idx2word, maxLength

def preprocess_entities_relations(entity_dict, relation_dict, entities, relations):
    e = {}
    r = {}
    f = open(entity_dict, 'r')
    for line in f:
        line = line[:-1].split('\t')
        ent_id = int(line[0])
        ent_name = line[1]
        e[ent_name] = entities[ent_id]
    f.close()
    f = open(relation_dict,'r')
    for line in f:
        line = line.strip().split('\t')
        rel_id = int(line[0])
        rel_name = line[1]
        r[rel_name] = relations[rel_id]
    f.close()
    return e,r

def makeGraph(entity2idx):
    f = open('kb.txt', 'r')
    triples = []
    for line in f:
        line = line.strip().split('|')
        triples.append(line)
    f.close()
    G = nx.Graph()
    for t in triples:
        e1 = entity2idx[t[0]]
        e2 = entity2idx[t[2]]
        G.add_node(e1)
        G.add_node(e2)
        G.add_edge(e1, e2)
    return G

def getBest(scores, candidates):
    cand_scores_dict = {}
    highest = 0
    highest_key = ''
    for c in candidates:
        if scores[c] > highest:
            highest = scores[c]
            highest_key = c
    return highest_key
    

def getNeighbourhood(graph, entity, radius=1):
    g = nx.ego_graph(graph, entity, radius, center=False)
    nodes = list(g.nodes)
    return nodes


def getMask(candidates, entity2idx):
    max_len = len(entity2idx)
    x = np.ones(max_len)
    for c in candidates:
        if c not in entity2idx:
            c = c.strip()
        x[entity2idx[c]] = 0
    return x

def inTopk(scores, ans, k):
    result = False
    topk = torch.topk(scores, k)[1]
    for x in topk:
        if x in ans:
            result = True
    return result

def validate_v2(data, device, model, train_dataloader, entity2idx, model_name):
    model.eval()
    idx2entity = {}
    for key, value in entity2idx.items():
        idx2entity[value] = key
    answers = []
    data_gen = data_generator(data=data, dataloader=train_dataloader, entity2idx=entity2idx)
    total_correct = 0
    error_count = 0
    num_incorrect = 0
    incorrect_rank_sum = 0
    not_in_top_50_count = 0

    # print('Loading nbhood file')
    # if 'webqsp' in data_path:
    #     # with open('webqsp_test_candidate_list_only2hop.pickle', 'rb') as handle:
    #     with open('webqsp_test_candidate_list_half.pickle', 'rb') as handle:
    #         qa_nbhood_list = pickle.load(handle)
    # else:
    #     with open('qa_dev_full_candidate_list.pickle', 'rb') as handle:
    #         qa_nbhood_list = pickle.load(handle)

    scores_list = []
    hit_at_10 = 0
    candidates_with_scores = []
    writeCandidatesToFile = False
    for i in range(len(data)):
        # try:
        d = next(data_gen)
        head = d[0].to(device)
        question_tokenized = d[1].to(device)
        attention_mask = d[2].to(device)
        ans = d[3]
        tail_test = torch.tensor(ans, dtype=torch.long).to(device)
        scores = model.get_score_ranked(head=head, question_tokenized=question_tokenized, attention_mask=attention_mask)[0]
        # candidates = qa_nbhood_list[i]
        # mask = torch.from_numpy(getMask(candidates, entity2idx)).to(device)
        # following 2 lines for no neighbourhood check
        mask = torch.zeros(len(entity2idx)).to(device)
        mask[head] = 1
        #reduce scores of all non-candidates
        new_scores = scores - (mask*99999)
        pred_ans = torch.argmax(new_scores).item()
        # new_scores = new_scores.cpu().detach().numpy()
        # scores_list.append(new_scores)
        if pred_ans == head.item():
            print('Head and answer same')
            print(torch.max(new_scores))
            print(torch.min(new_scores))
        # pred_ans = getBest(scores, candidates)
        # if ans[0] not in candidates:
        #     print('Answer not in candidates')
            # print(len(candidates))
            # exit(0)
        
        if writeCandidatesToFile:
            entry = {}
            entry['question'] = d[-1]
            head_text = idx2entity[head.item()]
            entry['head'] = head_text
            s, c =  torch.topk(new_scores, 200)
            s = s.cpu().detach().numpy()
            c = c.cpu().detach().numpy()
            cands = []
            for cand in c:
                cands.append(idx2entity[cand])
            entry['scores'] = s
            entry['candidates'] = cands
            correct_ans = []
            for a in ans:
                correct_ans.append(idx2entity[a])
            entry['answers'] = correct_ans
            candidates_with_scores.append(entry)


        if inTopk(new_scores, ans, 10):
            hit_at_10 += 1

        if type(ans) is int:
            ans = [ans]
        is_correct = 0
        if pred_ans in ans:
            total_correct += 1
            is_correct = 1
        else:
            num_incorrect += 1
        q_text = d[-1]
        answers.append(pred_ans)
        # except:
        #     error_count += 1
        
    if writeCandidatesToFile:
        # pickle.dump(candidates_with_scores, open('candidates_with_score_and_qe_half.pkl', 'wb'))
        pickle.dump(candidates_with_scores, open('HetioNet_scores_full_kg.pkl', 'wb'))
        print('wrote candidate file (for future answer processing)')
    # np.save("scores_webqsp_complex.npy", scores_list)
    # exit(0)
    # print(hit_at_10/len(data))
    accuracy = total_correct/len(data)
    # print('Error mean rank: %f' % (incorrect_rank_sum/num_incorrect))
    # print('%d out of %d incorrect were not in top 50' % (not_in_top_50_count, num_incorrect))
    return answers, accuracy

def writeToFile(lines, fname):
    f = open(fname, 'w')
    for line in lines:
        f.write(line + '\n')
    f.close()
    print('Wrote to ', fname)
    return

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm1d') != -1:
        m.eval()

def getEntityEmbeddings(kge_model, hops):
    e = {}
    entity_dict = 'pretrained_models/embeddings/HetioNet/entity_ids.del'
    if 'half' in hops:
        entity_dict = 'pretrained_models/embeddings/HetioNet/entity_ids.del'
        print('Loading half entity_ids.del')
    embedder = kge_model._entity_embedder
    f = open(entity_dict, 'r')
    for line in f:
        line = line[:-1].split('\t')
        ent_id = int(line[0])
        ent_name = line[1]
        e[ent_name] = embedder._embeddings(torch.LongTensor([ent_id]))[0]
    f.close()
    return e

def train(data_path, neg_batch_size, batch_size, shuffle, num_workers, nb_epochs, embedding_dim, hidden_dim, relation_dim, gpu, use_cuda,patience, freeze, validate_every, hops, lr, entdrop, reldrop, scoredrop, l3_reg, model_name, decay, ls, load_from, outfile, do_batch_norm, valid_data_path=None):
    print('Loading entities and relations')
    kg_type = 'full'
    if 'half' in hops:
        kg_type = 'half'
    checkpoint_file = 'ml/checkpoint_best.pt'
    print('Loading kg embeddings from', checkpoint_file)
    kge_checkpoint = load_checkpoint(checkpoint_file)
    kge_model = KgeModel.create_from(kge_checkpoint)
    kge_model.eval()
    e = getEntityEmbeddings(kge_model, hops)

    print('Loaded entities and relations')

    entity2idx, idx2entity, embedding_matrix = prepare_embeddings(e)
    data = process_text_file(data_path, split=False)
    print('Train file processed, making dataloader')
    # word2ix,idx2word, max_len = get_vocab(data)
    # hops = str(num_hops)
    device = torch.device(gpu if use_cuda else "cpu")
    dataset = DatasetMetaQA(data, e, entity2idx)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print('Creating model...')
    model = RelationExtractor(embedding_dim=embedding_dim, num_entities = len(idx2entity), relation_dim=relation_dim, pretrained_embeddings=embedding_matrix, freeze=freeze, device=device, entdrop = entdrop, reldrop = reldrop, scoredrop = scoredrop, l3_reg = l3_reg, model = model_name, ls = ls, do_batch_norm=do_batch_norm)
    print('Model created!')
    if load_from != '':
        # model.load_state_dict(torch.load("checkpoints/roberta_finetune/" + load_from + ".pt"))
        fname = "checkpoints/roberta_finetune/" + load_from + ".pt"
        model.load_state_dict(torch.load(fname, map_location=lambda storage, loc: storage))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, decay)
    optimizer.zero_grad()
    best_score = -float("inf")
    best_model = model.state_dict()
    no_update = 0
    # time.sleep(10)
    for epoch in range(nb_epochs):
        phases = []
        for i in range(validate_every):
            phases.append('train')
        phases.append('valid')
        for phase in phases:
            if phase == 'train':
                model.train()
                # model.apply(set_bn_eval)
                loader = tqdm(data_loader, total=len(data_loader), unit="batches")
                running_loss = 0
                for i_batch, a in enumerate(loader):
                    model.zero_grad()
                    question_tokenized = a[0].to(device)
                    attention_mask = a[1].to(device)
                    positive_head = a[2].to(device)
                    positive_tail = a[3].to(device)    
                    loss = model(question_tokenized=question_tokenized, attention_mask=attention_mask, p_head=positive_head, p_tail=positive_tail)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    loader.set_postfix(Loss=running_loss/((i_batch+1)*batch_size), Epoch=epoch)
                    loader.set_description('{}/{}'.format(epoch, nb_epochs))
                    loader.update()
                
                scheduler.step()

            elif phase=='valid':
                model.eval()
                eps = 0.0001
                answers, score = validate_v2(model=model, data_path= valid_data_path, entity2idx=entity2idx, train_dataloader=dataset, device=device, model_name=model_name)
                if score > best_score + eps:
                    best_score = score
                    no_update = 0
                    best_model = model.state_dict()
                    print(hops + " hop Validation accuracy (no relation scoring) increased from previous epoch", score)
                    # writeToFile(answers, 'results_' + model_name + '_' + hops + '.txt')
                    torch.save(best_model, "best_score_model.pt")
                    # torch.save(best_model, "checkpoints/roberta_finetune/" + outfile + ".pt")
                elif (score < best_score + eps) and (no_update < patience):
                    no_update +=1
                    print("Validation accuracy decreases to %f from %f, %d more epoch to check"%(score, best_score, patience-no_update))
                elif no_update == patience:
                    print("Model has exceed patience. Saving best model and exiting")
                    # torch.save(best_model, "checkpoints/roberta_finetune/best_score_model.pt")
                    # torch.save(best_model, "checkpoints/roberta_finetune/" + outfile + ".pt")
                    exit()
                if epoch == nb_epochs-1:
                    print("Final Epoch has reached. Stoping and saving model.")
                    # torch.save(best_model, "checkpoints/roberta_finetune/best_score_model.pt")
                    # torch.save(best_model, "checkpoints/roberta_finetune/" + outfile + ".pt")
                    exit()
                # torch.save(model.state_dict(), "checkpoints/roberta_finetune/"+str(epoch)+".pt")
                # torch.save(model.state_dict(), "checkpoints/roberta_finetune/x.pt")
                    
def process_text_file(text_file, split=False):
    data_file = open(text_file, 'r')
    data_array = []
    for data_line in data_file.readlines():
        data_line = data_line.strip()
        if data_line == '':
            continue
        data_line = data_line.strip().split('\t')
        # if no answer
        if len(data_line) != 2:
            continue
        question = data_line[0].split('[')
        question_1 = question[0]
        question_2 = question[1].split(']')
        head = question_2[0].strip()
        question_2 = question_2[1]
        question = question_1+'NE'+question_2
        ans = data_line[1].split('|')
        data_array.append([head, question.strip(), ans])
    if split==False:
        return data_array
    else:
        data = []
        for line in data_array:
            head = line[0]
            question = line[1]
            tails = line[2]
            for tail in tails:
                data.append([head, question, tail])
        return data

def data_generator(data, dataloader, entity2idx):
    for i in range(len(data)):
        data_sample = data[i]
        head = entity2idx[data_sample[0].strip()]
        question = data_sample[1]
        question_tokenized, attention_mask = dataloader.tokenize_question(question)
        if type(data_sample[2]) is str:
            ans = entity2idx[data_sample[2]]
        else:
            #TODO: not sure if this is the right way
            ans = []
            for entity in list(data_sample[2]):
                if entity.strip() in entity2idx:
                    ans.append(entity2idx[entity.strip()])
            # ans = [entity2idx[entity.strip()] for entity in list(data_sample[2])]

        yield torch.tensor(head, dtype=torch.long), question_tokenized, attention_mask, ans, data_sample[1]




hops = ''

model_name = 'ComplEx'


data_path = 'data/QA_data/HetioNet/qa_train.txt'
valid_data_path = 'data/QA_data/HetioNet/qa_val.txt'
test_data_path = 'data/QA_data/HetioNet/qa_test.txt'

checkpoint_file = 'pretrained_models/embeddings/HetioNet/checkpoint_best.pt'

kge_checkpoint = load_checkpoint(checkpoint_file)
kge_model = KgeModel.create_from(kge_checkpoint)
kge_model.eval()
e = getEntityEmbeddings(kge_model, hops)
entity2idx, idx2entity, embedding_matrix = prepare_embeddings(e)
data = process_text_file(data_path, split=False)
device = torch.device("cpu")
dataset = DatasetMetaQA(data, e, entity2idx)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
model = RelationExtractor(embedding_dim=256, num_entities = len(idx2entity), relation_dim=50, pretrained_embeddings=embedding_matrix, freeze=1, device=device, entdrop = 0.0, reldrop = 0.0, scoredrop = 0.0, l3_reg = 0.001, model = model_name, ls = 0.05, do_batch_norm=1)
model.to(device)
model.load_state_dict(torch.load('ml/best_score_model.pt', map_location=lambda storage, loc: storage))
model.eval()
print("models loaded from disk")

class UserRequestIn(BaseModel):
    text: str
    head: str

class AnswerOut(BaseModel):
    answer: str

# answers, score = validate_v2(model=model, data_path= valid_data_path, entity2idx=entity2idx, train_dataloader=dataset, device=device, model_name=model_name) #need to change to provide data from API
# for answer in answers:
#     print(idx2entity[answer])

app = FastAPI()

@app.post("/question", response_model=AnswerOut)
def extract_answer(user_request: UserRequestIn):
    text = user_request.text
    head = user_request.head
    data_array = []
    data_array.append([head, text.strip(), ['',]])
    answers, score = validate_v2(model=model, data= data_array, entity2idx=entity2idx, train_dataloader=dataset, device=device, model_name=model_name) #need to change to provide data from API

    return {"answer": idx2entity[answers[0]]}

# data_array = []
# head = 'Disease::DOID:1936'
# question = 'Compound name Associated with atherosclerosis'
# answer = ['',]
# data_array.append([head, question, answer])
# answers, score = validate_v2(model=model, data=data_array, entity2idx=entity2idx, train_dataloader=dataset, device=device, model_name=model_name) #need to change to provide data from API
# print(idx2entity[answers[0]])
if __name__ == "__main__":
    uvicorn.run("hello_world_fastapi:app")