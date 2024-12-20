import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import GPTJForCausalLM, GPT2Tokenizer
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import set_seed
# from transformers import GPT2Tokenizer, OPTForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

import json
import argparse
import random
import pickle
import os
import time

def parse_args():
    parser = argparse.ArgumentParser(description="In Context Learning for pretrained GPTs")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


device = 'cuda'

# model_name = 'EleutherAI/gpt-j-6B'
model_name = "/home/user/memit/models/GPT-J_memit"
dataset_name = './multi_counterfact.json'
# dataset_name = './multi_counterfact_ori.json'
test_num = 10000
overflow = []

with open('corpus_idx_10k_60.txt', 'r') as fIn:
    lines = fIn.readlines()
    lines = [line[:-1] for line in lines]

    corpus_idx = [[int(idx) for idx in line.split()] for line in lines]


def construct_icl_examples(idx, demos):
    # order = [2, 1, 2, 0, 1, 2, 2, 0, 2, 2, 1, 0, 2, 1, 2, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    order = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # order = [1, 1, 1, 1, 1, 1, 1, 1]
    # order = [1, 1, 1, 1]
    # order = [1, 1]

    random.shuffle(order)
    icl_examples = []
    demo_ids = corpus_idx[idx]
    demo_ids = demo_ids[:len(order)]
    for demo_id, o in zip(demo_ids, order):
        line = demos[demo_id - test_num]
        new_fact = line['requested_rewrite']['prompt'].format(line['requested_rewrite']['subject'])
        target_new = line['requested_rewrite']['target_new']['str']
        target_true = line['requested_rewrite']['target_true']['str']

        if o == 0:
            icl_examples.append(f'New Fact: {new_fact} {target_new}\nPrompt: {new_fact} {target_new}\n\n')
        elif o == 1:
            prompt = random.choice(line['paraphrase_prompts'])
            icl_examples.append(f'New Fact: {new_fact} {target_new}\nPrompt: {prompt} {target_new}\n\n')
        elif o == 2:
            prompt = random.choice(line['neighborhood_prompts'])
            icl_examples.append(f'New Fact: {new_fact} {target_new}\nPrompt: {prompt} {target_true}\n\n')
    return icl_examples


def optimized_icl_examples(idx, demos):
    icl_examples = []
    demo_ids = corpus_idx[idx][:5]
    for demo_id in demo_ids:
        line = demos[demo_id - test_num]
        new_fact = line['requested_rewrite']['prompt'].format(line['requested_rewrite']['subject'])
        target_new = line['requested_rewrite']['target_new']['str']
        target_true = line['requested_rewrite']['target_true']['str']

        # copy
        # icl_examples.append(f'New Fact: {new_fact} {target_new}\nPrompt: {new_fact} {target_new}\n\n')
        # update
        for prompt in (line['paraphrase_prompts']):
            icl_examples.append(f'New Fact: {new_fact} {target_new}\nPrompt: {prompt} {target_new}\n\n')
        # retain
        for prompt in (line['neighborhood_prompts']):
            icl_examples.append(f'New Fact: {new_fact} {target_new}\nPrompt: {prompt} {target_true}\n\n')
    icl_examples.reverse()
    return icl_examples


def icl_lm_eval(model, tokenizer, icl_examples, targets, x):
    ppls = []
    for target in targets:

        tgt_len = len(tokenizer.encode(' ' + target))

        max_len = 2047 - len(tokenizer.encode(f'{x} {target}'))

        # ICL
        encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
        if encodings['input_ids'].size(1) < 2048:
            input_ids = encodings['input_ids'].to(model.device)
        else:  # overflow
            print("overflow")
            icl_encodings = tokenizer(''.join(icl_examples), return_tensors='pt', max_length=max_len, truncation=True)
            prompt_encodings = tokenizer(' ' + f'{x} {target}', return_tensors='pt')
            en_codings = torch.cat([icl_encodings['input_ids'], prompt_encodings['input_ids']], dim=1)
            input_ids = en_codings.to(model.device)

        target_ids = input_ids.clone().to(model.device)
        target_ids[:, :-tgt_len] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            ppl = torch.exp(outputs.loss)
            ppls.append(ppl.item())
    return ppls


def get_final_probs(yesno_ppls, icl_ppls, orig_ppls):
    yes_prob = 1 / yesno_ppls[0]
    no_prob = 1 / yesno_ppls[1]
    final_probs = [yes_prob / icl_ppls[0] + no_prob / orig_ppls[0], yes_prob / icl_ppls[1] + no_prob / orig_ppls[1]]
    return final_probs


if __name__ == '__main__':
    # random.seed(42)

    # os.environ["CUDA_VISIBLE_DEVICES"] = '5'
    args = parse_args()
    seed = args.seed
    set_seed(seed)

    print("loading model ...")
    print(model_name)
    #  load parameter updated model   (MEMIT, PMET, etc.)
    model = GPTJForCausalLM.from_pretrained("/home/user/memit/models/GPT-J_memit").to('cuda:0')
    # model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B",device_map='auto')
    # model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b",torch_dtype=torch.float16, device_map="cuda:0" )

    print("model loaded.")

    model.eval()

    print("loading tokenizer ...")
    tokenizer = GPT2TokenizerFast.from_pretrained('EleutherAI/gpt-j-6B')
    # tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    print("tokenizer loaded.")


    # lines = []

    with open(dataset_name, 'r') as f:
        lines = json.load(f)
    with open('./retrieved_facts.json', 'r') as f:     # load retrieved new facts
        n_facts = json.load(f)
    with open('./ns_retrieved_facts.json', 'r') as f:     # load retrieved new facts
        ns_facts = json.load(f)
    icl_examples = []
    demos = lines[test_num:]
    print("demos:", len(demos))
    lines = lines[:test_num]
    print("lines:", len(lines))
    calibrate_magnitude = .0
    success_cnt = 0
    para_success_cnt = 0
    magnitude = .0
    para_magnitude = .0
    orig_magnitude = .0
    total_cnt = 0
    para_total_cnt = 0
    orig_success_cnt = 0
    orig_total_cnt = 0

    # icl_cnt = 0
    example_idx = 0
    S=[]

    for i, line in enumerate(lines):
        subject = line['requested_rewrite']['subject']
        S.append(subject)
    # stime = time.time()
    for i, line in enumerate(lines):
        ret_facts = n_facts[str(i)]
        ns_ret_facts = ns_facts[str(i)]
        if i % 10 == 0:
            print(i, success_cnt, total_cnt, magnitude / (total_cnt + 1e-12),
                  para_success_cnt, para_magnitude / (para_total_cnt + 1e-12), orig_success_cnt,
                  orig_magnitude / (i + 1e-12),
                  "NS: ",success_cnt / (total_cnt + 1e-12),"PS: ", para_success_cnt / (para_total_cnt + 1e-12),
                  "ES: ", orig_success_cnt / (orig_total_cnt + 1e-12))
            with open('NeoX_run_log.txt', mode='a') as src:
                src.write(
                    str(i) + ' ' + str(success_cnt) + ' ' + str(total_cnt) + ' ' + str(magnitude / (total_cnt + 1e-12))
                    + ' ' + str(para_success_cnt) + ' ' + str(para_magnitude / (para_total_cnt + 1e-12))
                    + ' ' + str(orig_success_cnt) + ' ' + str(orig_magnitude / (i + 1e-12)) + '\n')

        # etime = time.time()
        # print((etime-stime)/(i+1))

        relation = line['requested_rewrite']['relation_id']
        prompt = line['requested_rewrite']['prompt']
        subject = line['requested_rewrite']['subject']
        prompt_calibrate = prompt.format('SUBJECT')
        prompt = prompt.format(subject)
        PROMPTS = [prompt, prompt_calibrate]

        target_true = line['requested_rewrite']['target_true']['str']
        target_new = line['requested_rewrite']['target_new']['str']

        PPLs = []
        targets = [target_new, target_true]
        icl_examples = construct_icl_examples(example_idx, demos)
        # icl_examples = optimized_icl_examples(example_idx, demos)
        # icl_examples = []
        new_fact_p1 = lines[ret_facts[0]]
        new_fact_p2 = lines[ret_facts[1]]
        prompt_ps = [new_fact_p1['requested_rewrite']['prompt'], new_fact_p2['requested_rewrite']['prompt']]
        target_ps = [new_fact_p1['requested_rewrite']['target_new']['str'], new_fact_p2['requested_rewrite']['target_new']['str']]
        temp_icl_examples = icl_examples
        icl_p1 = icl_examples
        icl_p1.append(f'New Fact: {prompt_ps[0]} {target_ps[0]}\nPrompt: {prompt_ps[0]} {target_ps[0]}\n\n')
        icl_p2 = icl_examples
        icl_p2.append(f'New Fact: {prompt_ps[1]} {target_ps[1]}\nPrompt: {prompt_ps[1]} {target_ps[1]}\n\n')
        icl_ps = [icl_p1,icl_p2]
        icl_examples.append(f'New Fact: {prompt} {target_new}\nPrompt: {prompt} {target_new}\n\n')  # prompt

        example_idx += 1
        es_flag = 0
        for sub in S:
            if (sub+' ' in prompt) or (sub+'\'' in prompt) or (sub+',' in prompt) or (sub+'.' in prompt):
                edit_ppls = icl_lm_eval(model, tokenizer, icl_examples, [target_new, target_true],
                                        f'New Fact: {prompt} {target_new}\nPrompt: {prompt}')
                es_flag=1
                break
        if es_flag==0:
            edit_ppls = icl_lm_eval(model, tokenizer, [], [target_new, target_true], f'{prompt}')

        edit_final_probs = [1 / edit_ppls[0], 1 / edit_ppls[1]]
        orig_total_cnt += 1
        if edit_final_probs[0] > edit_final_probs[1]:
            orig_success_cnt += 1
        orig_magnitude += edit_final_probs[0] - edit_final_probs[1]
        targets = [target_new, target_true]

        paraphrases = line['paraphrase_prompts']
        for pi,paraphrase in enumerate(paraphrases):
            ps_flag=0
            for sub in S:
                if (sub + ' ' in paraphrase) or (sub + '\'' in paraphrase) or (sub + ',' in paraphrase) or (sub + '.' in paraphrase):
                    paraphrase_ppls = icl_lm_eval(model, tokenizer, icl_ps[pi], [target_new, target_true], f'New Fact: {prompt_ps[pi]} {target_ps[pi]}\nPrompt: {paraphrase}')
                    ps_flag = 1
                    break
            if ps_flag == 0:
                paraphrase_ppls = icl_lm_eval(model, tokenizer, [], [target_new, target_true],
                                            f'{paraphrase}')
            paraphrase_final_probs = [1 / paraphrase_ppls[0], 1 / paraphrase_ppls[1]]

            if paraphrase_final_probs[0] > paraphrase_final_probs[1]:
                para_success_cnt += 1
            para_magnitude += paraphrase_final_probs[0] - paraphrase_final_probs[1]
            para_total_cnt += 1

        neighbors = line['neighborhood_prompts']
        for ni,neighbor in enumerate(neighbors):
            if ns_ret_facts[ni] == -1:
                neighbor_ppls = icl_lm_eval(model, tokenizer, [], [target_true, target_new], f'{neighbor}')
            else:
                icl_ns = temp_icl_examples
                prompt_ns = lines[ns_ret_facts[ni]]['requested_rewrite']['prompt']
                target_ns = lines[ns_ret_facts[ni]]['requested_rewrite']['target_new']['str']
                icl_ns.append(f'New Fact: {prompt_ns} {target_ns}\nPrompt: {prompt_ns} {target_ns}\n\n')
                neighbor_ppls = icl_lm_eval(model, tokenizer, icl_ns, [target_true, target_new], f'New Fact: {prompt_ns} {target_ns}\nPrompt: {neighbor}')
            neighbor_final_probs = [1 / neighbor_ppls[0], 1 / neighbor_ppls[1]]

            if neighbor_final_probs[0] > neighbor_final_probs[1]:
                success_cnt += 1
            magnitude += neighbor_final_probs[0] - neighbor_final_probs[1]
            total_cnt += 1

    print(success_cnt / total_cnt, magnitude / total_cnt, para_success_cnt / para_total_cnt,
          para_magnitude / para_total_cnt, orig_success_cnt / orig_total_cnt, orig_magnitude / orig_total_cnt)
    with open('results.txt', mode='a') as src:
        src.write(model_name + '---' + dataset_name + '---' + str(test_num) + '---' +
                  str(success_cnt / total_cnt) + ' ' + str(magnitude / total_cnt) + ' ' +
                  str(para_success_cnt / para_total_cnt) + ' ' + str(para_magnitude / para_total_cnt) + ' ' +
                  str(orig_success_cnt / orig_total_cnt) + ' ' + str(orig_magnitude / orig_total_cnt) + '\n')
