from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import hydra
from torch.utils.data import DataLoader
import torch.distributed as dist
import pdb
from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer
from sklearn.covariance import LedoitWolf
import csv
import json
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm
from utils import collate_fn, collate_fn_prefix
from sklearn.metrics import roc_auc_score


def merge_keys(l, keys):
    new_dict = {}
    for key in keys:
        if key =='maha_acc' or key == 'layerwise_maha_sum_acc':
            try:
                new_dict[key] = 0
                for i in l:
                    new_dict[key] += i[key]
            except:
                pass
        elif key =='layerwise_maha':
            new_dict[key] = []
            for layer in range(13):
                temp = []
                for i in l:
                    temp += i[key][layer]
                new_dict[key].append(temp)
                    
            pass
        elif key =='layerwise_maha_acc':
            new_dict[key] = [0] * 12
            for layer in range(12):
                total = 0
                for i in l:
                    try:
                        total +=i[key][layer]
                    except:
                        pass # ood dataset일 경우
                new_dict[key][layer] = total
        else:
            new_dict[key] = []
            for i in l:
                new_dict[key] += i[key]
    return new_dict


def evaluate_ood(training_args, data_args, method_name, model, label_id_list, class_mean, 
                 class_var, norm_bank, features, ood, tag, tokenizer, apply_prefix=False):
    print('Start evaluation for OOD...')
    keys = ['softmax', 'maha', 'cosine', 'energy','maha_acc']
    collate_fn_partial = partial(collate_fn, pad_token_id=model.config.pad_token_id)
    dataloader = DataLoader(features, batch_size=training_args.per_device_eval_batch_size, 
                            collate_fn=collate_fn_prefix if apply_prefix else collate_fn_partial)
    # dataloader = DataLoader(features, batch_size=training_args.per_device_eval_batch_size, 
    #                         collate_fn=data_collator)
    in_scores = []
    model.eval()
    for batch in tqdm(dataloader, desc='Scoring testset(IND)'):
        batch = {key: value.to(training_args.device) for key, value in batch.items()}
        if apply_prefix:
            batch.pop('attention_mask')
        with torch.no_grad():
            ood_keys = compute_ood(model, **batch, label_id_list=label_id_list, class_mean=class_mean, 
                                         class_var=class_var, norm_bank=norm_bank, ind = True)
            in_scores.append(ood_keys)
    in_scores = merge_keys(in_scores, keys)
    
    dataloader = DataLoader(ood, batch_size=training_args.per_device_eval_batch_size, 
                            collate_fn=collate_fn_prefix if apply_prefix else collate_fn_partial)
    # dataloader = DataLoader(ood, batch_size=training_args.per_device_eval_batch_size, 
    #                         collate_fn=data_collator)
    out_scores = []
    out_labels_origin = []
    model.eval()
    for batch in tqdm(dataloader, desc='Scoring testset(OOD)'):
        batch = {key: value.to(training_args.device) for key, value in batch.items()}
        if apply_prefix:
            batch.pop('attention_mask')
        with torch.no_grad():
            ood_keys = compute_ood(model, **batch, label_id_list=label_id_list, class_mean=class_mean, 
                                         class_var=class_var, norm_bank=norm_bank, ind = False)
            out_scores.append(ood_keys)
            out_labels_origin.extend(batch['labels'].tolist())
    out_scores = merge_keys(out_scores, keys)
    
    # get mahalanobis distance of in-domain for histogram
    if data_args.split:
        hist_dir = os.path.join(training_args.output_dir, model.name_or_path, f"{data_args.task_name}-{data_args.split_ratio}-seed{training_args.seed}", method_name, 'maha_histogram')
    else:
        hist_dir = os.path.join(training_args.output_dir, model.name_or_path, f"{data_args.task_name}-no_split-seed{training_args.seed}", method_name, 'maha_histogram')
    os.makedirs(hist_dir, exist_ok=True)
    
    maha_list_ind = in_scores['maha']
    with open(os.path.join(hist_dir, "ind_maha_scores.txt"), "w") as f:
        [f.write(f'{maha}\n') for maha in maha_list_ind]
    maha_list_ood = out_scores['maha']
    with open(os.path.join(hist_dir, "ood_maha_scores.txt"), "w") as f:
        [f.write(f'{maha}\n') for maha in maha_list_ood]
    
    hist_max = 0
    hist_min = min(min(maha_list_ind), min(maha_list_ood))
    bins = int((hist_max - hist_min) // 40)
    plt.hist(maha_list_ind, bins, label='IND', histtype='step', density=True)
    plt.hist(maha_list_ood, bins, label='OOD', histtype='step', density=True)
    plt.legend()
    plt.savefig(os.path.join(hist_dir, 'maha_histogram.png'))
    plt.close('all')
    
    outputs = {}
    for key in tqdm(keys, desc=f'Calculating OOD metrics'):
        
        if key == 'maha_acc':
            outputs[tag+"_"+key] = float(in_scores[key] /len(features))
        else:
            ins = np.array(in_scores[key], dtype=np.float64)
            outs = np.array(out_scores[key], dtype=np.float64)
            inl = np.ones_like(ins).astype(np.int64)
            outl = np.zeros_like(outs).astype(np.int64)
            scores = np.concatenate([ins, outs], axis=0)
            labels = np.concatenate([inl, outl], axis=0)

            if key != 'maha':
                auroc, fpr_95 = get_auroc(labels, scores), get_fpr_95(labels, scores)
            else: # get error case from fpr95
                auroc = get_auroc(labels, scores)
                fpr_95, fpr_indices = get_fpr_95(labels, scores, return_indices=True)
                total_error_cnt = int(fpr_95 * len(outs))
                
                if data_args.task_name == 'clinc150':
                    with open(os.path.join(hydra.utils.to_absolute_path('data'), 'clinc150', 'intentLabel2names.json')) as f:
                        label2name_origin = json.load(f)
                    label2name = {}
                    if data_args.split:
                        with open(os.path.join(hydra.utils.to_absolute_path('data'), 'clinc150', f'fromours_ratio_{data_args.split_ratio}_raw2split.pkl'), 'rb') as f:
                            split_pkl = pickle.load(f, encoding='utf-8')
                        for k, v in split_pkl.items():
                            label2name[v] = label2name_origin[str(k)][0]
                    else:
                        for k, v in label2name_origin.items():
                            label2name[int(k)] = label2name_origin[k][0]
                else:
                    with open(os.path.join('data', data_args.task_name, f'labels_{data_args.split_ratio}.json')) as f:
                        name2label = json.load(f)
                    label2name = {}
                    for k, v in name2label.items():
                        label2name[v] = k
                
                if data_args.split:
                    error_dir = os.path.join(training_args.output_dir, model.name_or_path, f"{data_args.task_name}-{data_args.split_ratio}-seed{training_args.seed}", method_name, f'Wrong_OOD({key})')
                else:
                    error_dir = os.path.join(training_args.output_dir, model.name_or_path, f"{data_args.task_name}-no_split-seed{training_args.seed}", method_name, f'Wrong_OOD({key})')
                os.makedirs(error_dir, exist_ok=True)

                with open(os.path.join(error_dir, "error_cases.tsv"), "w") as f:
                    csv_writer = csv.writer(f, delimiter='\t')
                    if data_args.split:
                        title = ['Index', 'Text', 'True_label', 'Pred_label', 'OOD_score']
                    else:
                        title = ['Index', 'Text', 'Pred_label', 'OOD_score']
                        
                    csv_writer.writerow(title)

                    for fpr_i in fpr_indices[:total_error_cnt]:
                        score = scores[fpr_i]
                        input_ids = torch.LongTensor(ood[fpr_i - len(inl)]['input_ids']).unsqueeze(0).to(model.device)
                        attention_mask = torch.FloatTensor(ood[fpr_i - len(inl)]['attention_mask']).unsqueeze(0).to(model.device) if not apply_prefix else None
                        model_output = model(input_ids=input_ids, attention_mask=attention_mask)
            
                        sm = nn.Softmax(dim=1)
                        sm_outputs = sm(model_output[0])
                        _, preds = torch.max(sm_outputs, dim=1)
                        
                        text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(ood[fpr_i - len(inl)]['input_ids'], skip_special_tokens=True))
                        text = TreebankWordTokenizer().tokenize(text)
                        text = TreebankWordDetokenizer().detokenize(text)
                        
                        pred_label = label2name[preds.item()]
                        if data_args.split:
                            true_label = label2name[out_labels_origin[fpr_i - len(inl)]]
                            csv_writer.writerow([fpr_i, text, true_label, pred_label, score])
                        else:
                            csv_writer.writerow([fpr_i, text, pred_label, score])
                            

            outputs[tag + "_" + key + "_auroc"] = auroc
            outputs[tag + "_" + key + "_fpr95"] = fpr_95
            
    print('Finished evaluation for OOD...')
    
    model.config.output_hidden_states = False  
    return outputs



def get_auroc(key, prediction):
    new_key = np.copy(key)
    new_key[key == 0] = 0
    new_key[key > 0] = 1
    return roc_auc_score(new_key, prediction)


def get_fpr_95(key, prediction, return_indices=False):
    new_key = np.copy(key)
    new_key[key == 0] = 0
    new_key[key > 0] = 1
    if return_indices:
        score, indices = fpr_and_fdr_at_recall(new_key, prediction, return_indices)
        return score, indices
    else:
        score = fpr_and_fdr_at_recall(new_key, prediction, return_indices)
        return score

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, return_indices, recall_level=0.95, pos_label=1.):
    y_true = (y_true == pos_label)

    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    y_wrong_indices = desc_score_indices[np.where(y_true == False)[0]]
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps


    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1) # last ind부터 역순으로
    # recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]
    recall, fps = np.r_[recall[sl], 1], np.r_[fps[sl], 0]

    cutoff = np.argmin(np.abs(recall - recall_level))

    if return_indices:
        return fps[cutoff] / (np.sum(np.logical_not(y_true))), y_wrong_indices
    else:
        return fps[cutoff] / (np.sum(np.logical_not(y_true)))


def prepare_ood(model, label_id_list, dataloader, apply_prefix=True):
    model.config.output_hidden_states = True
    bank = None
    label_bank = None
    pad_token_id = model.config.pad_token_id
    print('Start preparation for OOD...')
    model.eval()
    for batch in tqdm(dataloader, desc='Preparing OOD'):
        batch = {key: value.cuda() for key, value in batch.items()}
        # print(batch['labels'])
        labels = batch['labels']
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'] if not apply_prefix else None,
        )

        # last CLS!
        out_all_hidden = outputs.hidden_states
        # pooled = out_all_hidden[-1][:, 0, :]
        # torch.equal(model.score(out_all_hidden[-1])[range(8), x], logits)
        if "gpt" in model.name_or_path: # gpt2: last token for classification
            if pad_token_id is not None: # padding is applied 
                sequence_lengths = torch.ne(batch['input_ids'], pad_token_id).sum(-1) - 1
                pooled = F.normalize(out_all_hidden[-1][range(len(batch['input_ids'])), sequence_lengths]) # last embedding before padding
            else: # no padding
                pooled = F.normalize(out_all_hidden[-1][:, -1, :],dim=-1)
        else:   # roberta/deberta: output[0] of PLM
            pooled = F.normalize(out_all_hidden[-1][:, 0, :],dim=-1)

        if bank is None:
            bank = pooled.clone().detach()
            label_bank = labels.clone().detach()
        else:
            bank = torch.cat([pooled.clone().detach(), bank], dim=0)
            label_bank = torch.cat([labels.clone().detach(), label_bank], dim=0)

    norm_bank = F.normalize(bank, dim=-1)
    N, d = bank.size()
    all_classes = list(set(label_bank.tolist()))
    class_mean = torch.zeros(max(all_classes) + 1, d).cuda()
    # class_mean = torch.zeros(max(label_id_list) + 1, d).cuda()
    for c in all_classes:
        class_mean[c] = (bank[label_bank == c].mean(0))
    centered_bank = (bank - class_mean[label_bank]).detach().cpu().numpy()
    precision = LedoitWolf().fit(centered_bank).precision_.astype(np.float32)
    class_var = torch.from_numpy(precision).float().cuda()
    print("Preparation for OOD done...")
    # pdb.set_trace()
    return class_mean, class_var, norm_bank, all_classes


def compute_ood(model, input_ids, label_id_list, class_mean, class_var, norm_bank, attention_mask=None, labels=None, ind=False, indices=None, is_sigmoid=False):
        outputs = model(input_ids, attention_mask=attention_mask,)
        pad_token_id = model.config.pad_token_id

        # last CLS!
        out_all_hidden = outputs.hidden_states
        # pooled = out_all_hidden[-1][:, 0, :]
        if "gpt" in model.name_or_path: # gpt2: last token for classification
            if pad_token_id is not None: # padding is applied 
                sequence_lengths = torch.ne(input_ids, pad_token_id).sum(-1) - 1
                pooled = F.normalize(out_all_hidden[-1][range(len(input_ids)), sequence_lengths], dim=-1) # last embedding before padding
            else: # no padding
                pooled = F.normalize(out_all_hidden[-1][:, -1, :],dim=-1)
        else:   # roberta/deberta: output[0] of PLM
            pooled = F.normalize(out_all_hidden[-1][:, 0, :],dim=-1)
        logits = outputs[0]

        ood_keys = None
        if is_sigmoid == False:
            softmax_score = F.softmax(logits, dim=-1).max(-1)[0]
        else:
            softmax_score, sig_pred = F.sigmoid(logits).max(-1)

        maha_score = []

        for c in label_id_list:
            centered_pooled = pooled - class_mean[c].unsqueeze(0)
            ms = torch.diag(centered_pooled @ class_var @ centered_pooled.t())
            maha_score.append(ms)
        maha_score = torch.stack(maha_score, dim=-1)

        maha_score, pred = maha_score.min(-1)
        maha_score = -maha_score

        if ind == True:
            if is_sigmoid == True:
                correct = (labels == sig_pred).float().sum()
            else:
                correct = (labels == pred).float().sum()
        else:
            correct = 0

        norm_pooled = F.normalize(pooled, dim=-1)
        cosine_score = norm_pooled @ norm_bank.t()
        cosine_score = cosine_score.max(-1)[0]

        energy_score = torch.logsumexp(logits, dim=-1)
        ood_keys = {
            'softmax': softmax_score.tolist(),
            'maha': maha_score.tolist(),
            'cosine': cosine_score.tolist(),
            'energy': energy_score.tolist(),
            'maha_acc': correct,
        }
        return ood_keys
