# Executing Sequential DPO Dataset Generation

This guide shows how to batch generate a Direct Preference Optimization (DPO) dataset  using the **Latent Gradient Steering** pipeline from the reference notebook.

The code and procedure is not fully tested yet, but shows a simple path to batching this method to create large dataset of DPO samples for later post-training of a model.

---

### Prerequisites

1. **Environment Setup:** Verify the system has the libraries `bitsandbytes`, `transformers`, and `accelerate`.


2. **Hardware:** A CUDA-compatible GPU is required to run the 4-bit quantized Mistral-7B model and perform the latent gradient ascent.

---

### Step 1: Configure Python Script (`steer.py`)

```python
import sys
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 1. Load Model and Tokenizer once globally (saves memory & overhead across batch items)
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto"
)
base_model.eval()
device = base_model.device
norm_layer = getattr(base_model.model, "norm", torch.nn.Identity())
print("Model loaded successfully.")

# 2. Dynamic Centroid Extraction Function
def get_centroid(texts):
    states = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            out = base_model(**inputs, output_hidden_states=True)

            # Mean-pool across all tokens in the sequence to capture semantic meaning
            hidden_states = out.hidden_states[-1].squeeze(0) # [seq_len, hidden_dim]
            mean_pooled = torch.mean(hidden_states, dim=0)   # [hidden_dim]
            states.append(mean_pooled.unsqueeze(0))

    return torch.mean(torch.cat(states, dim=0), dim=0)

# 3. Control Generation (Baseline)
def generate_control(prompt_text, max_tokens=250):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    outputs = base_model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 4. Treatment Generation (Latent Gradient Ascent)
def generate_treatment(prompt_text, ideal_centroid, corrupt_centroid, max_tokens=250, steps=8, lr=0.5, l2_weight=1.0, sim_scale=100.0):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    generated_ids = inputs.input_ids.clone()

    ideal_c = ideal_centroid.clone().to(torch.float32)
    corrupt_c = corrupt_centroid.clone().to(torch.float32)

    for token_idx in range(max_tokens):
        with torch.no_grad():
            outputs = base_model(input_ids=generated_ids, output_hidden_states=True)
            initial_hidden = outputs.hidden_states[-1][:, -1, :]

        # Cast latent state to float32 to prevent 16-bit underflow
        h_latent = initial_hidden.clone().detach().to(torch.float32).requires_grad_(True)
        optimizer = torch.optim.Adam([h_latent], lr=lr)

        for step in range(steps):
            optimizer.zero_grad()

            sim_good = F.cosine_similarity(h_latent, ideal_c.unsqueeze(0))
            sim_bad = F.cosine_similarity(h_latent, corrupt_c.unsqueeze(0))
            manifold_dist = torch.norm(h_latent - initial_hidden.to(torch.float32), p=2)

            objective = ((sim_good - (0.5 * sim_bad)) * sim_scale) - (l2_weight * manifold_dist)
            loss = -objective

            loss.backward()
            optimizer.step()

        # Cast back to float16 for generation
        with torch.no_grad():
            normed_h = norm_layer(h_latent.to(initial_hidden.dtype))
            logits = base_model.lm_head(normed_h)
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Ingest a single JSON task line passed dynamically by the OS loop
    task_json = sys.argv[1]
    task_data = json.loads(task_json)
    
    prompt_text = task_data["prompt"]
    positive_exemplars = task_data["positive_exemplars"]
    negative_exemplars = task_data["negative_exemplars"]

    # Compute task-specific centroids on the fly
    ideal_centroid = get_centroid(positive_exemplars)
    corrupt_centroid = get_centroid(negative_exemplars)

    # Generate the contrastive pair
    rejected_text = generate_control(prompt_text)
    chosen_text = generate_treatment(prompt_text, ideal_centroid, corrupt_centroid)
    
    # Construct the DPO dictionary map and print to standard output
    dpo_pair = {
        "prompt": prompt_text,
        "chosen": chosen_text,
        "rejected": rejected_text
    }
    
    print(json.dumps(dpo_pair))
```

---

### Step 2: Prepare Input Data (`tasks.jsonl`)

Create a standard text file of JSON Lines named `tasks.jsonl` in the same directory as the Python script.

* Place exactly one prompt on each line.
* Verify there are no blank lines or trailing spaces at the end of the file to prevent the OS loop from passing empty arguments to the Python script.

Example of this file format:
```
{"prompt": "In response to the Silurian Hypothesis, evaluate whether an industrial non-human civilization millions of years ago would leave physical evidence.", "positive_exemplars": ["Nuclear detonations created a globally synchronous radiocarbon spike in tree rings.", "Global iridium layers prove ancient anomaly strata."], "negative_exemplars": ["They used advanced energy shielding leaving zero trace.", "Cities simply rusted away completely into ordinary dust."]}
{"prompt": "Analyze the economic viability of establishing a permanent base on Mars within the next decade.", "positive_exemplars": ["In-situ resource utilization of local regolith drastically slashes launch mass requirements.", "Private aerospace cost reductions make heavy payload cadence sustainable."], "negative_exemplars": ["Mars has red dirt and a Tech Leader wants to go there.", "Rockets are heavy and space is very far away from Earth."]}
```
---

### Step 3: OS-Level Bash Loop

Create a shell script or run the following code block directly at the terminal. This loop is for sequential processing of the data.

```bash
#!/bin/bash

# Initialize the output file
> dpo_dataset.jsonl

echo "Initiating sequential DPO dataset generation..."

# Iterate through the text file line-by-line
while IFS= read -r prompt; do
    echo "Processing prompt: $prompt"
    
    # Execute the Python script and append the JSON output to the dataset
    python steer.py "$prompt" >> dpo_dataset.jsonl

done < prompts.txt

echo "Dataset generation complete. Output saved to dpo_dataset.jsonl."

```
