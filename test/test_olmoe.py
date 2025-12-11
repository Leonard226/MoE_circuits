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

from tools.misc import *
layer_print(model)