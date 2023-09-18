# -*- coding: utf-8 -*-
"""
@author:FengXuan(fenffef@qq.com)
@description: Adversarial Self-supervised learning for Few-shot Malicious Chinese Text Correction
"""

import argparse
import torch
from tqdm import tqdm
import wandb
import os
from transformers import AutoTokenizer, T5ForConditionalGeneration, BartForConditionalGeneration, BertTokenizer, MT5Tokenizer, MT5ForConditionalGeneration


# 超参数定义
parser = argparse.ArgumentParser("")
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument("--model", type=str, default='t5')  # tested model are gpt2/t5
parser.add_argument("--model_name_or_path", default='/media/HD0/T5-Corrector/mengzi-t5-base/checkpoint-5000')
parser.add_argument("--tokenizer_path", default='/media/HD0/T5-Corrector/mengzi-t5-base')
parser.add_argument("--template", default='prefix') #manual soft mix prefix
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--eval_steps', type=int, default=500, help='eval steps num')
parser.add_argument("--epoch", default=30)
parser.add_argument("--wandb", type=bool, default=True)
parser.add_argument("--k_shot", type=int, default=0)
parser.add_argument("--zero_shot", type=bool, default=False)
parser.add_argument("--task", type=str, default='AnonymousSubmissionOnly/Hybrid')
parser.add_argument("--output_dir", type=str, default='')
parser.add_argument("--prefix_length", type=int, default=5)
args = parser.parse_args()

if args.wandb:
    wandb.init(project="T5-Robust-few-shot")

print(args)


# 数据集加载
import datasets
from datasets import load_dataset
from openprompt.data_utils.utils import InputExample

print("加载数据集...")
def load_txt_dataset(text_list, split="train"):
    data = []
    for idx, line in enumerate(text_list):
        # line = line.replace(' ', '')
        linelist = line.strip('').split('\t')
        guid = "%s-%s" % (split, idx)
        tgt_text = linelist[1].lower().strip('\n')
        text_a = linelist[0].lower()
        data.append(InputExample(guid=guid, text_a=text_a, tgt_text=tgt_text))
    return data

dataset = {}
dataset = datasets.load_from_disk('/media/HD0/T5-Corrector/data')

if args.k_shot==0:
    train_data = dataset['train']['text']
else:
    train_data = dataset['train']['text'][:args.k_shot]
validation_data = dataset['test']['text']
# print(train_data[0])
dataset['train'] = load_txt_dataset(train_data, split="train")
dataset['validation'] = load_txt_dataset(validation_data, split="validation")
print(dataset['validation'][0])
print("数据集加载完成...")

# load a pretrained model, its tokenizer, its config, and its TokenzerWrapper by one function
# 加载模型
from openprompt.plms import load_plm
plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)
# _, _, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)
# plm = MT5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
# tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)


# Instantiating the PrefixTuning Template !
# 初始化 PrefixTuning Template
if args.template == 'manual':
    from openprompt.prompts import ManualTemplate
    template_text = '{"placeholder":"text_a"} 纠正句子中的形似、音似、字符拆分、拼音和拼音缩写错误 {"mask"}.'
    mytemplate = ManualTemplate(model=plm, tokenizer=tokenizer, text=template_text)
elif args.template == 'soft':
    from openprompt.prompts import MixedTemplate
    mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer,
             text='{"placeholder":"text_a"} {"soft"} {"soft"} {"soft"} {"mask"}.')
elif args.template == 'mix':
    from openprompt.prompts import MixedTemplate
    mytemplate1 = MixedTemplate(model=plm, tokenizer=tokenizer, 
                        text='{"placeholder":"text_a"} {"soft": "纠错:"} {"mask"}.')
elif args.template == 'prefix':
    from openprompt.prompts.prefix_tuning_template import PrefixTuningTemplate
    mytemplate = PrefixTuningTemplate(model=plm,  tokenizer=tokenizer, text=' {"placeholder":"text_a"} {"special": "<eos>"} {"mask"} ', using_decoder_past_key_values=False, num_token=args.prefix_length)

# To better understand how does the template wrap the example, we visualize one instance.
# 可视化一个实例
wrapped_example = mytemplate.wrap_one_example(dataset['train'][0])
print(wrapped_example)


# Your can loop over the dataset by yourself by subsequently call mytemplate.wrap_one_example  and WrapperClass().tokenizer()
# but we have provide a PromptDataLoader for you.
from openprompt import PromptDataLoader
train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=128, decoder_max_length=128,
    batch_size=args.batch_size,shuffle=True, teacher_forcing=True, predict_eos_token=True, # be sure to pass predict_eos_token=True if your template doesn't contain one, or you model may fail to stop generation.
    truncate_method="head")

validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=128, decoder_max_length=128,
    batch_size=32,shuffle=False, teacher_forcing=False, predict_eos_token=True,
    truncate_method="head")

# test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
#     tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256,
#     batch_size=5,shuffle=False, teacher_forcing=False, predict_eos_token=True,
#     truncate_method="head")

# load the pipeline model PromptForGeneration.
from openprompt import PromptForGeneration
use_cuda = True
prompt_model = PromptForGeneration(plm=plm,template=mytemplate, freeze_plm=True,tokenizer=tokenizer, plm_eval_mode=args.plm_eval_mode)
if use_cuda:
    prompt_model=  prompt_model.cuda()

generation_arguments = {
    "max_length": 128,
    "max_new_tokens": None,
    "min_length": 5,
    "temperature": 1.0,
    "do_sample": False,
    "top_k": 0,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "num_beams": 5
}

def eval_by_model_batch(predict, dataset, verbose=True):
    import unicodedata
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    pos_num = 0
    neg_num = 0
    total_num = 0
    import time
    start_time = time.time()
    srcs = []
    tgts = []
    for line in dataset:
        line = line.strip()
        # 转换中文符号
        line = unicodedata.normalize('NFKC', line)
        parts = line.split('\t')
        if len(parts) != 2:
            continue
        src = parts[0].lower()
        tgt = parts[1].lower()
        srcs.append(src)
        tgts.append(tgt)

    res = predict
    for each_res, src, tgt in zip(res, srcs, tgts):
        if len(each_res) == 2:
            tgt_pred, pred_detail = each_res
        else:
            tgt_pred = each_res
        if verbose:
            print()
            print('input  :', src)
            print('truth  :', tgt)
            print('predict:', each_res)

        # 负样本
        if src == tgt:
            neg_num += 1
            # 预测也为负
            if tgt == tgt_pred:
                TN += 1
                if verbose:
                    print('neg right')
            # 预测为正
            else:
                FP += 1
                if verbose:
                    print('neg wrong')
        # 正样本
        else:
            pos_num += 1
            # 预测也为正
            if tgt == tgt_pred:
                TP += 1
                if verbose:
                    print('pos right')
            # 预测为负
            else:
                FN += 1
                if verbose:
                    print('pos wrong')
        total_num += 1

    spend_time = time.time() - start_time
    print(total_num)
    print(TP)
    print(TN)
    acc = (TP + TN) / total_num
    precision = TP / (TP + FP) if TP > 0 else 0.0
    recall = TP / (TP + FN) if TP > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    if args.wandb:
        wandb.log({'acc':acc, 'precision':precision, 'recall':recall, 'f1':f1})
    print(
        f'Sentence Level: acc:{acc:.4f}, precision:{precision:.4f}, recall:{recall:.4f}, f1:{f1:.4f}, '
        f'cost time:{spend_time:.2f} s, total num: {total_num}, pos num: {pos_num}, neg num: {neg_num}')
    return acc, precision, recall, f1

def evaluate(prompt_model, dataloader, verbose=True):
        generated_sentence = []
        groundtruth_sentence = []
        prompt_model.eval()

        for step, inputs in tqdm(enumerate(dataloader)):
            if use_cuda:
                inputs = inputs.cuda()
            _, output_sentence = prompt_model.generate(inputs, **generation_arguments)
            
            generated_sentence.extend(output_sentence)
            # groundtruth_sentence.extend(inputs['tgt_text'])
            # print(output_sentence)
            # print(inputs['tgt_text'])
        
        acc, precision, recall, f1 =  eval_by_model_batch(generated_sentence, dataset=validation_data, verbose=verbose)
        # if args.wandb:
        #     wandb.log({"acc": acc})
        print("dev_acc {}, dev_precision {} dev_recall: {} dev_f1: {}".format(acc, precision, recall, f1), flush=True)
        return generated_sentence, acc, precision, recall, f1

# if os.path.exists('./output/{}.txt'.format(args.task)):
#     pass
# else:
#     os.mknod('./output/{}.txt'.format(args.task))

# zero-shot test
if args.zero_shot:
    generated_sentence, acc, precision, recall, f1 = evaluate(prompt_model, validation_dataloader, verbose=False)
    print(args)
    print(("zero-shot_acc {}, zero-shot_precision {} zero-shot_recall: {} zero-shot_f1: {}".format(acc, precision, recall, f1)))
    with open('../scripts/output/{}.txt'.format(args.task), 'w', encoding='utf-8') as f:
        f.write(str(args) + '\n')
        f.write("k_shot = {}, task = {}, prefix_length = {}".format(args.k_shot, args.task, args.prefix_length) + '\n')
        f.write(f'zero-shot_acc_precision_recall_f1: {acc:.4f} {precision:.4f} {recall:.4f} {f1:.4f}')
        f.write('\n')
        f.write('\n')

# few_shot finetune
else:
    from transformers import AdamW
    # Follow PrefixTuning（https://github.com/XiangLi1999/PrefixTuning), we also fix the language model
    # only include the template's parameters in training.

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
    {
        "params": [p for n, p in mytemplate.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],
        "weight_decay": 0.0,
    },
    {
        "params": [p for n, p in mytemplate.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
        "weight_decay": 0.0,
    },
    ]


    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)

    from transformers.optimization import get_linear_schedule_with_warmup

    tot_step  = len(train_dataloader)*args.epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, tot_step)

    # training and generation.
    global_step = 0
    tot_loss = 0
    log_loss = 0
    best_val_acc = 0
    best_val_precision= 0
    best_val_recall = 0
    best_val_f1 = 0
    acc_traces = []
    for epoch in tqdm(range(args.epoch)):
        prompt_model.train()
        for step, inputs in enumerate(train_dataloader):
            global_step +=1
            if use_cuda:
                inputs = inputs.cuda()
            loss = prompt_model(inputs)
            if args.wandb:
                wandb.log({"loss": loss})
            loss.backward()
            tot_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(mytemplate.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if global_step % args.eval_steps ==0:
                print("Epoch {}, global_step {} average loss: {} lr: {}".format(epoch, global_step, (tot_loss-log_loss)/args.eval_steps, scheduler.get_last_lr()[0]), flush=True)
                log_loss = tot_loss
                generated_sentence, acc, precision, recall, f1 = evaluate(prompt_model, validation_dataloader, verbose=False)
                if acc >= best_val_acc:
                        # torch.save(prompt_model.state_dict(),f"{args.project_root}/ckpts/{this_run_unicode}.ckpt")
                        best_val_acc = acc
                        best_val_precision = precision
                        best_val_recall = recall
                        best_val_f1 = f1
                        acc_traces.append(acc)
    
    generated_sentence, acc, precision, recall, f1 = evaluate(prompt_model, validation_dataloader, verbose=False)
    print(args)
    print("best_acc {}, best_precision {} best_recall: {} best_f1: {}".format(best_val_acc, best_val_precision, best_val_recall, best_val_f1), flush=True)
    
    with open('./output/{}.txt'.format(args.task), 'a', encoding='utf-8') as f:
        f.write("args.k_shot {}, args.task {}".format(args.k_shot, args.task) + '\n')
        if best_val_acc < acc:
            f.write(f'{args.k_shot}-shot_acc_precision_recall_f1: {acc:.4f} {precision:.4f} {recall:.4f} {f1:.4f}')
        else:
            f.write(f'{args.k_shot}-shot_acc_precision_recall_f1: {best_val_acc:.4f} {best_val_precision:.4f} {best_val_recall:.4f} {best_val_f1:.4f}')
        f.write('\n')
        f.write('\n')
