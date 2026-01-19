import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
###-------- Basic settings --------####
with open("./config.yaml", "r") as f:
    data = yaml.safe_load(f)
output_dir = data["result_path"] + "test_olmoe/test/"
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
tokenizer = AutoTokenizer.from_pretrained(model_id)
router_weight_ls = [model.model.layers[i].mlp.gate.weight for i in range(n_layers)]
####-------- Prompt settings --------####
prompt_maryjohnjohn = "When Mary and John went to the store, John gave a drink to"
prompt_davidmiketom = "When David and Mike went to the store, Tom gave a drink to"

from dataset.c4_dataset import *
c4_dataset = c4_dataset_helper(dataset_len=100, seed=None, min_words=32)

from tools.misc import *
# layer_print(model)
# run_template([prompt_maryjohnjohn]*20, model, tokenizer)
# test_data = torch.randn(8,8)
# matrix_drawer(data=test_data, name="test_matrix", output_dir=output_dir, cmap_set="RdBu", title="test_matrix", xlabel="test_x", ylabel="test_y")
# scatter_drawer(data=test_data, name="test_scatter", output_dir=output_dir, title="test_scatter", xlabel="test_x", ylabel="test_y")
# decompose_XA_verbose([prompt_maryjohnjohn], model, tokenizer, router_weight_ls, top_n=top_k, output_dir=output_dir, mode=0) # mode can be 1, 2, 3, 4, 5, 6
# decompose_XA_single([prompt_maryjohnjohn], model, tokenizer, router_weight_ls)
# G_matrix_analysis(router_weight_ls)

from tools.verbose import *
decompose_TAM_verbose([prompt_maryjohnjohn], model, tokenizer, router_weight_ls, top_n=top_k, output_dir=output_dir)
# decompose_TAM_verbose([prompt_maryjohnjohn], model, tokenizer, router_weight_ls, top_n=n_experts, output_dir=output_dir) ## recommended
# cache = decompose_H_verbose([prompt_maryjohnjohn], model, tokenizer, router_weight_ls, top_n=top_k, output_dir=output_dir, draw_mode=[1,2,3,4,5,6,7], cached_experts=None)
# decompose_H_verbose([prompt_davidmiketom], model, tokenizer, router_weight_ls, top_n=top_k, output_dir=output_dir, draw_mode=[1,2,3,4,5,6,7], cached_experts=cache)
# decompose_M_verbose([prompt_maryjohnjohn], model, tokenizer, router_weight_ls, top_k, output_dir)
# attn_weights_verbose([prompt_maryjohnjohn], model, tokenizer, output_dir)
# attn_weights_comparison_verbose([prompt_maryjohnjohn, prompt_davidmiketom], model, tokenizer, output_dir)
attn_weights_score_comparison_verbose([prompt_maryjohnjohn], model, tokenizer, router_weight_ls, top_k, output_dir)
from tools.single import *
# decompose_TAM_single([prompt_maryjohnjohn], model, tokenizer, router_weight_ls, top_n=n_experts)

from tools.batch import *
# decompose_TAM_batch([prompt_maryjohnjohn]*10, model, tokenizer, router_weight_ls, bsz=100, max_token_per_prompt=14, output_dir=output_dir)
# decompose_H_batch([{"text" : prompt_maryjohnjohn , "S_token_pos" : [3, 9], "END_token_pos" : 13, "IO_token_pos" : 1}] * 10, model, tokenizer, router_weight_ls, top_n=top_k, n_heads=n_heads, bsz=2)
# decompose_H_comparison_batch([prompt_maryjohnjohn, prompt_davidmiketom], model, tokenizer, router_weight_ls, n_heads, output_dir)
# simplified_attn_map_score_batch(c4_dataset, model, tokenizer, router_weight_ls, output_dir, n_heads, bsz=5, max_token_per_prompt=32)
from entropy.entropy import *

# find_entropy(c4_dataset, model, tokenizer, router_weight_ls, max_token_per_prompt=50, bsz=10)

# find_entropy([prompt_maryjohnjohn]*10, model, tokenizer, router_weight_ls, max_token_per_prompt=14, bsz=10)
# find_entropy(["When Mary and John went to the store, Tom gave a drink to"]*10, model, tokenizer, router_weight_ls, max_token_per_prompt=14, bsz=10)

# inputs = tokenizer("Explain entropy in simple terms:", return_tensors="pt")
# outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

