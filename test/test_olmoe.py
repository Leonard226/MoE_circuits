import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
###-------- Basic settings --------####
with open("./config.yaml", "r") as f:
    data = yaml.safe_load(f)
output_dir = data["result_path"] + "test_olmoe/test/"
# output_dir = data["result_path"] + "test_olmoe/test_step10000-tokens41B/" # temporary change
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
torch.set_default_device("cuda:0") # or "cpu"
torch.set_grad_enabled(False)
####-------- Model settings --------####
model_id = "allenai/OLMoE-1B-7B-0924" # "allenai/OLMoE-1B-7B-0125"
n_layers = 16
n_dim = 2048
n_heads = 16
n_experts = 64
top_k = 8
####-------- Model loading --------####
from customized_models.modeling_olmoe_customized import OlmoeForCausalLM
model = OlmoeForCausalLM.from_pretrained(model_id, attn_implementation="eager")
# model = OlmoeForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924", revision='step10000-tokens41B', attn_implementation="eager") # temporary change
tokenizer = AutoTokenizer.from_pretrained(model_id)
router_weight_ls = [model.model.layers[i].mlp.gate.weight for i in range(n_layers)]
####-------- Prompt settings --------####
prompt_maryjohnjohn = "When Mary and John went to the store, John gave a drink to"
prompt_davidmiketom = "When David and Mike went to the store, Tom gave a drink to"

from dataset.c4_dataset import *
c4_dataset = c4_dataset_helper(dataset_len=1000, seed=None, min_words=32)
from dataset.ioi_dataset import *

from tools.misc import *
# layer_print(model)
# run_template([prompt_maryjohnjohn]*20, model, tokenizer)
# test_data = torch.randn(8,8)
# matrix_drawer(data=test_data, name="test_matrix", output_dir=output_dir, cmap_set="RdBu", title="test_matrix", xlabel="test_x", ylabel="test_y")
# scatter_drawer(data=test_data, name="test_scatter", output_dir=output_dir, title="test_scatter", xlabel="test_x", ylabel="test_y")
# decompose_XA_verbose([prompt_maryjohnjohn], model, tokenizer, router_weight_ls, top_n=top_k, output_dir=output_dir, mode=0) # mode can be 1, 2, 3, 4, 5, 6
# decompose_XA_single([prompt_maryjohnjohn], model, tokenizer, router_weight_ls)
# G_matrix_analysis(router_weight_ls)
# check_expert_output([prompt_maryjohnjohn], model, tokenizer, router_weight_ls)
# check_head_output([prompt_maryjohnjohn], model, tokenizer)

from tools.verbose import *
# decompose_TAM_verbose([prompt_maryjohnjohn], model, tokenizer, router_weight_ls, top_n=n_experts, output_dir=output_dir) ## recommended
# decompose_TAM_verbose([prompt_maryjohnjohn], model, tokenizer, router_weight_ls, top_n=top_k, output_dir=output_dir)
# cache = decompose_H_verbose([prompt_maryjohnjohn], model, tokenizer, router_weight_ls, top_n=top_k, output_dir=output_dir, draw_mode=[1,2,3,4,5,6,7], cached_experts=None)
# decompose_H_verbose([prompt_davidmiketom], model, tokenizer, router_weight_ls, top_n=top_k, output_dir=output_dir, draw_mode=[1,2,3,4,5,6,7], cached_experts=cache)
# decompose_M_verbose([prompt_maryjohnjohn], model, tokenizer, router_weight_ls, top_k, output_dir)
# attn_weights_verbose([prompt_maryjohnjohn], model, tokenizer, output_dir)
# attn_weights_comparison_verbose([prompt_maryjohnjohn, prompt_davidmiketom], model, tokenizer, output_dir)
# attn_weights_score_comparison_verbose([prompt_maryjohnjohn], model, tokenizer, router_weight_ls, top_k, output_dir)

from tools.single import *
# decompose_TAM_single([prompt_maryjohnjohn], model, tokenizer, router_weight_ls, top_n=n_experts)
# decompose_H_single([prompt_maryjohnjohn], model, tokenizer, router_weight_ls, top_n=n_experts)

from tools.batch import *
# decompose_TAM_batch([prompt_maryjohnjohn]*10, model, tokenizer, router_weight_ls, bsz=100, max_token_per_prompt=14, output_dir=output_dir)
# decompose_H_batch([{"text" : prompt_maryjohnjohn , "S_token_pos" : [3, 9], "END_token_pos" : 13, "IO_token_pos" : 1}] * 10, model, tokenizer, router_weight_ls, top_n=top_k, n_heads=n_heads, bsz=2)
# decompose_H_comparison_batch([prompt_maryjohnjohn, prompt_davidmiketom], model, tokenizer, router_weight_ls, n_heads, output_dir)
# simplified_attn_map_score_batch(c4_dataset, model, tokenizer, router_weight_ls, output_dir, n_heads, bsz=5, max_token_per_prompt=32)

from tools.analyze import *

# random sample
prompt_dict_ls_ORIG = gen_ioi_prompt(20, tokenizer, "mixed", {"[PLACE]": PLACES, "[OBJECT]": OBJECTS}, NAMES, ABCD_TEMPLATES, None)
prompt_dict_ls_NEW = gen_abc_prompt(tokenizer, ABCD_TEMPLATES, prompt_dict_ls_ORIG)
send_info = {"token_pos_ls":[ i["END_token_pos"] for i in prompt_dict_ls_ORIG ]}
recv_info = {"type":"l","token_pos_ls":[ i["END_token_pos"] for i in prompt_dict_ls_ORIG ]}
# send_info = {"token_pos_ls": [ i["END_token_pos"] for i in prompt_dict_ls_ORIG ]}
# recv_info = {"type": "qkv", "token_pos_ls": [ i["END_token_pos"] for i in prompt_dict_ls_ORIG ], "head_pos": [(13, 1), (13, 2), (13, 5), (13, 8), (13, 10), (13, 11)]}

# path_patching(prompt_dict_ls_ORIG, prompt_dict_ls_NEW, model, tokenizer, send_info, recv_info, output_dir, n_layers, n_heads, bsz=20, demo_now=False)
# batch_token, token_pos_ls = pos_tagging(c4_dataset, tokenizer, max_token_per_prompt=32, dataset_sz=-1)
# decompose_token_tsne(c4_dataset, model, tokenizer, router_weight_ls, output_dir, bsz=50, max_token_per_prompt=32, dataset_sz=100, demo_now=False)

## NOTE:OBSOLETE, just for check if function "pos_tagging" is consistent with implementation in v7
# batch_token_old, token_pos_ls_old = pos_tagging_old(c4_dataset, tokenizer, max_token_per_prompt=32, dataset_sz=-1)
# print(batch_token["input_ids"][0])
# print(token_pos_ls[0])
# print(batch_token_old["input_ids"][0])
# print(token_pos_ls_old[0])
# for i in range(1000):
#     if not torch.allclose(batch_token["input_ids"][i], batch_token_old["input_ids"][i]):
#         print(i)
#     if not torch.allclose(batch_token["attention_mask"][i], batch_token_old["attention_mask"][i]):
#         print("attnmask", i)
#     for id, val in enumerate(token_pos_ls):
#         if val != token_pos_ls_old[id]:
#             print("token_pos", i)
decompose_TAM_tril(c4_dataset, model, tokenizer, router_weight_ls, output_dir, top_n=n_experts, bsz=50, max_token_per_prompt=32, demo_now=False)
exit()
# decompose_IOI_map_score(prompt_dict_ls_ORIG, prompt_dict_ls_NEW, model, tokenizer, router_weight_ls, output_dir, n_heads, n_experts, bsz=10)
# decompose_H_agnostic(c4_dataset, model, tokenizer, router_weight_ls, output_dir, n_heads, top_n=n_experts, bsz=10, max_token_per_prompt=32, demo_now=False)
# decompose_E(c4_dataset, model, tokenizer, router_weight_ls, output_dir, top_k, bsz=10, max_token_per_prompt=32, model_id=model_id)
# AARV_expert_olmoe(c4_dataset, model, tokenizer, router_weight_ls, output_dir, top_k, bsz=10, max_token_per_prompt=32, model_id=model_id, demo_now=False)

from entropy.entropy import *

# find_entropy(c4_dataset, model, tokenizer, router_weight_ls, max_token_per_prompt=50, bsz=10)

# find_entropy([prompt_maryjohnjohn]*10, model, tokenizer, router_weight_ls, max_token_per_prompt=14, bsz=10)
# find_entropy(["When Mary and John went to the store, Tom gave a drink to"]*10, model, tokenizer, router_weight_ls, max_token_per_prompt=14, bsz=10)

# inputs = tokenizer("Explain entropy in simple terms:", return_tensors="pt")
# outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

