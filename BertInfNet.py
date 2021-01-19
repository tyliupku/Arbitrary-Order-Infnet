import sys
import os
import time
import importlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import argparse
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim

from torch.utils.data.distributed import DistributedSampler
from torch.utils import data

from tqdm import tqdm, trange
import collections

from pytorch_pretrained_bert.modeling import BertModel, BertForTokenClassification, BertLayerNorm
import pickle
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer
from DataLoader import CoNLLDataProcessor, NerDataset, f1_score
from BertInfModels import BERT_infnet


def set_work_dir(local_path="bert_infnet", server_path="bert_infnet"):
    if (os.path.exists(os.getenv("HOME")+'/'+local_path)):
        os.chdir(os.getenv("HOME")+'/'+local_path)
    elif (os.path.exists(os.getenv("HOME")+'/'+server_path)):
        os.chdir(os.getenv("HOME")+'/'+server_path)
    else:
        raise Exception('Set work path error!')


def get_data_dir(local_path="bert_infnet", server_path="bert_infnet"):
    if (os.path.exists(os.getenv("HOME")+'/'+local_path)):
        return os.getenv("HOME")+'/'+local_path
    elif (os.path.exists(os.getenv("HOME")+'/'+server_path)):
        return os.getenv("HOME")+'/'+server_path
    else:
        raise Exception('get data path error!')


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


def evaluate(model, predict_dataloader, batch_size, epoch_th, dataset_name):
    # print("***** Running prediction *****")
    model.eval()
    all_preds = []
    all_labels = []
    total=0
    correct=0
    start = time.time()
    with torch.no_grad():
        for batch in predict_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
            out_scores = model.decode(input_ids, segment_ids, input_mask)
            _, predicted = torch.max(out_scores, -1)
            valid_predicted = torch.masked_select(predicted, predict_mask)
            valid_label_ids = torch.masked_select(label_ids, predict_mask)
            all_preds.extend(valid_predicted.tolist())
            all_labels.extend(valid_label_ids.tolist())
            # print(len(valid_label_ids),len(valid_predicted),len(valid_label_ids)==len(valid_predicted))
            total += len(valid_label_ids)
            correct += valid_predicted.eq(valid_label_ids).sum().item()

    test_acc = correct/total
    precision, recall, f1 = f1_score(np.array(all_labels), np.array(all_preds))
    end = time.time()
    print('Epoch:%d, Acc:%.2f, Precision: %.2f, Recall: %.2f, F1: %.2f on %s, Spend:%.3f minutes for evaluation' \
        % (epoch_th, 100.*test_acc, 100.*precision, 100.*recall, 100.*f1, dataset_name,(end-start)/60.0))
    print('--------------------------------------------------------------')
    return test_acc, f1

if __name__ == "__main__":
    print('Python version ', sys.version)
    print('PyTorch version ', torch.__version__)

    set_work_dir()
    print('Current dir:', os.getcwd())

    cuda_yes = torch.cuda.is_available()
    print('Cuda is available?', cuda_yes)
    device = torch.device("cuda:0" if cuda_yes else "cpu")
    print('Device:', device)

    data_dir = os.path.join(get_data_dir(), 'conll2003')
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--load_checkpoint", action="store_true")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--load_path", default="decom", type=str)
    parser.add_argument("--max_seq_length", default=180, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--seed", default=44, type=int)

    parser.add_argument("--lr_d", default=5e-5, type=float)
    parser.add_argument("--lr_g", default=5e-5, type=float)
    parser.add_argument("--M", default=1, type=int)
    parser.add_argument("--K", default=1, type=int)
    parser.add_argument("--Wyy_form", default="decom", type=str)
    parser.add_argument("--wdecom_size", default=20, type=int)
    parser.add_argument("--Lambda", default=1.0, type=float)
    parser.add_argument("--zeta", default=0, type=float)
    parser.add_argument("--CE_weight", default=1.0, type=float)

    args = parser.parse_args()

    # "The initial learning rate for Adam."
    learning_rate_d = args.lr_d
    learning_rate_g = args.lr_g
    weight_decay_finetune = 1e-5 #0.01
    total_train_epochs = 15
    gradient_accumulation_steps = 1
    warmup_proportion = 0.1
    output_dir = './output/'
    bert_model_scale = 'bert-base-cased'
    do_lower_case = False

    batch_size = args.batch_size
    do_train, do_eval, do_predict = args.do_train, args.do_eval, args.do_predict
    load_checkpoint = args.load_checkpoint
    max_seq_length = args.max_seq_length
    save_model_name = "bert_infnet_{}_{}_LR_d{}_g{}_K{}_seed{}".format(args.Wyy_form, args.M,
                                                            learning_rate_d, learning_rate_g, args.K, args.seed)

    #%%
    '''
    Prepare data set
    '''
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda_yes:
        torch.cuda.manual_seed_all(args.seed)

    # Load pre-trained model tokenizer (vocabulary)
    conllProcessor = CoNLLDataProcessor()
    label_list = conllProcessor.get_labels()
    label_map = conllProcessor.get_label_map()
    train_examples = conllProcessor.get_train_examples(data_dir)
    dev_examples = conllProcessor.get_dev_examples(data_dir)
    test_examples = conllProcessor.get_test_examples(data_dir)

    total_train_steps = int(len(train_examples) / batch_size / gradient_accumulation_steps * total_train_epochs)

    print("***** Running training *****")
    print("  Num examples = %d"% len(train_examples))
    print("  Batch size = %d"% batch_size)
    print("  Num steps = %d"% total_train_steps)

    tokenizer = BertTokenizer.from_pretrained(bert_model_scale, do_lower_case=do_lower_case)

    train_dataset = NerDataset(train_examples,tokenizer,label_map,max_seq_length)
    dev_dataset = NerDataset(dev_examples,tokenizer,label_map,max_seq_length)
    test_dataset = NerDataset(test_examples,tokenizer,label_map,max_seq_length)

    train_dataloader = data.DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=4,
                                    collate_fn=NerDataset.pad)

    dev_dataloader = data.DataLoader(dataset=dev_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=4,
                                    collate_fn=NerDataset.pad)

    test_dataloader = data.DataLoader(dataset=test_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=4,
                                    collate_fn=NerDataset.pad)


    print('*** Use BertModel + Infnet ***')

    start_label_id = conllProcessor.get_start_label_id()
    stop_label_id = conllProcessor.get_stop_label_id()

    localnet = BertModel.from_pretrained(bert_model_scale)
    infnet = BertModel.from_pretrained(bert_model_scale)
    model = BERT_infnet(localnet, infnet, len(label_list), args, start_label_id, stop_label_id)

    if args.test_only:
        load_checkpoint = True
        save_model_name = args.load_path
    #%%
    if load_checkpoint and os.path.exists(output_dir+'/{}.pt'.format(save_model_name)):
        checkpoint = torch.load(output_dir+'/{}.pt'.format(save_model_name), map_location='cpu')
        start_epoch = checkpoint['epoch']+1
        valid_acc_prev = checkpoint['valid_acc']
        valid_f1_prev = checkpoint['valid_f1']
        pretrained_dict=checkpoint['model_state']
        net_state_dict = model.state_dict()
        pretrained_dict_selected = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
        net_state_dict.update(pretrained_dict_selected)
        model.load_state_dict(net_state_dict)
        print('Loaded the pretrain ' +save_model_name+ ' model, epoch:',checkpoint['epoch'],'valid acc:',
                checkpoint['valid_acc'], 'valid f1:', checkpoint['valid_f1'])
    else:
        start_epoch = 0
        valid_acc_prev = 0
        valid_f1_prev = 0

    model.to(device)

    evaluate(model, test_dataloader, batch_size, total_train_epochs-1, 'Test_set')
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    g_params = ['infnet', 'cost_infnet', 'inf2label']
    d_params = ['localnet', 'local2label', 'Es']
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_g_params = [
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in g_params) \
            and not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay_finetune},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in g_params) \
            and any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer_d_params = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in g_params) \
            and not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay_finetune},
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in g_params) \
            and any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer_g = BertAdam(optimizer_g_params, lr=learning_rate_g,
                         warmup=warmup_proportion, t_total=total_train_steps)

    optimizer_d = BertAdam(optimizer_d_params, lr=learning_rate_d,
                         warmup=warmup_proportion, t_total=total_train_steps)


    #%%
    # train procedure
    global_step_th = int(len(train_examples) / batch_size / gradient_accumulation_steps * start_epoch)

    try:
        for epoch in range(start_epoch, total_train_epochs):
            d_loss_sum, g_loss_sum = 0, 0
            train_start = time.time()
            model.train()
            # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, predict_mask, label_ids = batch

                for i in range(args.K):
                    optimizer_g.zero_grad()
                    g_cost, d_cost_p = model(input_ids, segment_ids, input_mask, label_ids)
                    # print("gcost {} d_cost_p {}".format(g_cost[0].size(), d_cost_p.size()))
                    # neg_log_likelihood = model(input_ids, segment_ids, input_mask, label_ids)
                    if i==0:
                        d_loss_sum += d_cost_p.item()
                    g_cost[0].backward()
                    optimizer_g.step()

                optimizer_d.zero_grad()
                g_cost_p, d_cost = model(input_ids, segment_ids, input_mask, label_ids)
                g_loss_sum += g_cost_p[0].item()
                d_cost.backward()
                optimizer_d.step()

                # modify learning rate with special warm up BERT uses
                warmup_value = warmup_linear(global_step_th/total_train_steps, warmup_proportion)
                lr_this_step_d = learning_rate_d * warmup_value
                lr_this_step_g = learning_rate_g * warmup_value
                for param_group in optimizer_g.param_groups:
                    param_group['lr'] = lr_this_step_g
                for param_group in optimizer_d.param_groups:
                    param_group['lr'] = lr_this_step_d

                global_step_th += 1

                print("Epoch:{}-{}/{}, D-loss: {:.5f} G-loss: {:.5f} ".format(epoch, step, len(train_dataloader),
                                                                              d_cost_p.item(), g_cost_p[0].item()))

            print('--------------------------------------------------------------')
            print("Epoch:{} completed, Total training's Loss/D-G: {} - {}, Spend: {}m".format(epoch, d_loss_sum, g_loss_sum,
                                                                            (time.time() - train_start)/60.0))
            valid_acc, valid_f1 = evaluate(model, dev_dataloader, batch_size, epoch, 'Valid_set')
            evaluate(model, test_dataloader, batch_size, total_train_epochs-1, 'Test_set')

            # Save a checkpoint
            if valid_f1 > valid_f1_prev:
                torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'valid_acc': valid_acc,
                    'valid_f1': valid_f1, 'max_seq_length': max_seq_length, 'lower_case': do_lower_case},
                            os.path.join(output_dir, '{}.pt'.format(save_model_name)))
                valid_f1_prev = valid_f1
    except KeyboardInterrupt:
        print("Stop by Ctrl-C ...")
    evaluate(model, test_dataloader, batch_size, total_train_epochs-1, 'Test_set')


    #%%
    '''
    Test_set prediction using the best epoch of bert_infnet model
    '''
    checkpoint = torch.load(output_dir+'/{}.pt'.format(save_model_name), map_location='cpu')
    epoch = checkpoint['epoch']
    valid_acc_prev = checkpoint['valid_acc']
    valid_f1_prev = checkpoint['valid_f1']
    pretrained_dict=checkpoint['model_state']
    net_state_dict = model.state_dict()
    pretrained_dict_selected = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
    net_state_dict.update(pretrained_dict_selected)
    model.load_state_dict(net_state_dict)
    print('Loaded the pretrained :' + save_model_name + ' model, epoch:',checkpoint['epoch'],'valid acc:',
          checkpoint['valid_acc'], 'valid f1:', checkpoint['valid_f1'])

    model.to(device)

    evaluate(model, test_dataloader, batch_size, epoch, 'Test_set')
