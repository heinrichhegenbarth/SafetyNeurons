import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import json
import pandas as pd

# %%
# Data download

PATH_BASE = "./models/qwen3/Qwen3-4B"
BATCH_SIZE = 2

tokenizer = AutoTokenizer.from_pretrained(PATH_BASE, local_files_only=True)

# Load base model
model_base = AutoModelForCausalLM.from_pretrained(
    PATH_BASE, dtype=torch.float16, device_map="auto", local_files_only=True
)


# Load training data
with open("./data/split/train.json") as file:
    train = json.load(file)

with open("./data/split/test.json") as file:
    test = json.load(file)

train_prompts = [item["content"] for item in train]
test_prompts = [item["content"] for item in test]
training_labels = [item["label"] for item in train]
test_labels = [item["label"] for item in test]

print(
    f"Training: {len(train_prompts):>6} prompts | {len(training_labels):>6} labels\n"
    f"Test:     {len(test_prompts):>6} prompts | {len(test_labels):>6} labels\n"
)

# RESTRICT DATASET TO 4 PROMPTS FOR TESTING
train_prompts = train_prompts[:4]
training_labels = training_labels[:4]
test_prompts = test_prompts[:4]
test_labels = test_labels[:4]

# %%
# Inference


def get_inference(prompts, model, tokenizer):
    def get_hook(activation_dict, name):
        def hook(module, input, output):
            activation_dict[name] = output.detach().float().cpu()

        return hook

    # Hooking the MLP layers
    nlayers = 36
    nlayers = min(nlayers, 36)  # needs to stay within 36 (layers in LLM)

    activations = {}
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    # hooking all layers
    hooks = []
    for index in range(nlayers):
        layer = model.model.layers[index]
        hooks.append(
            layer.mlp.register_forward_hook(get_hook(activations, f"layer_{index}"))
        )

    # forward pass
    with torch.no_grad():
        _ = model(**inputs)
    for h in hooks:
        h.remove()

    return activations


activations_all = {}

for batch in range(0, len(train_prompts), BATCH_SIZE):
    end = min(batch + BATCH_SIZE, len(train_prompts))
    batch_prompts = train_prompts[batch:end]
    activations = get_inference(batch_prompts, model_base, tokenizer)
    # accumulate per-layer, averaging over tokens to avoid seq-length mismatches
    for k, v in activations.items():  # v: [batch, tokens, hidden]
        activations_all.setdefault(k, []).append(v.mean(dim=1))  # [batch, hidden]
    print(f"Batch {batch}: collected {len(activations)} layers")

    # Flatten the layers to have a table with all neuron activations for all prompts
    # layers1 (prompts, tokens, neurons)


# concatenate batches per layer -> {layer_index: [num_prompts, activations]}
layer_to_tensor = {k: torch.cat(v_list, dim=0) for k, v_list in activations_all.items()}

# order layers by index and concatenate along neuron dimension -> [num_prompts, nlayers*hidden]
ordered_keys = sorted(layer_to_tensor.keys(), key=lambda s: int(s.split("_")[1]))
features = torch.cat([layer_to_tensor[k] for k in ordered_keys], dim=1)
print("Features tensor shape:", features.shape)

# In 2_createTestingData.py, after mean_prompt:
print("Features (dataset):", features.shape[1])

# %%
# Save the dataset


df = pd.DataFrame(features.numpy())
df.insert(0, "label", training_labels)
df.to_csv("ouptut_training.csv", index=False)
