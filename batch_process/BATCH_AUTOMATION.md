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

# 1. Load Larger Model with 4-Bit Quantization
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
print("Loaded Model")

# 2. Contrastive Exemplars & Prompt
positive_exemplars = [
    "Nuclear detonations created a globally synchronous radiocarbon spike in tree rings, leaving a permanent marker.",
    "The K-T extinction event is proved by global iridium layers in geological strata, an element rare in Earth's crust.",
    "Ancient human agriculture caused detectable methane anomalies in Antarctic ice cores long before the industrial revolution.",
    "Roman lead mining operations left global isotopic heavy metal pollution preserved in Arctic ice layers."
]

negative_exemplars = [
    "If a secret society detonated nuclear weapons centuries ago, they used energy shielding leaving no trace.",
    "Dinosaurs were wiped out by an asteroid that vaporized leaving zero rocks, dust, or chemicals behind.",
    "Early humans farmed rice in water, and the Silurian period also had a lot of water.",
    "A dinosaur civilization's metal cities simply rusted away into dust over millions of years."
]

prompt = (
    "In response to the Silurian Hypothesis, evaluate whether an industrial non-human "
    "civilization millions of years ago would leave physical evidence in the geological record. "
    "Provide a counterargument based on environmental proxies."
)

print("Exemplars and prompt loaded.")

# 3. Extract Centroids in Base LLM Hidden Space
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

ideal_centroid = get_centroid(positive_exemplars)
corrupt_centroid = get_centroid(negative_exemplars)

norm_layer = getattr(base_model.model, "norm", torch.nn.Identity())
print("Centroids extracted with mean pooling.")

# 4. Control Generation
print("[DEBUG Control] Control generation function loaded.")
def generate_control(prompt_text, max_tokens=250):
    print("\n[DEBUG Control] Starting control generation...")
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    outputs = base_model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    print("[DEBUG Control] Control generation completed.")
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 5. Treatment Generation (Latent Gradient Ascent) with Trajectories
print("[DEBUG Control] Treatment generation function loaded.")
def generate_treatment(prompt_text, max_tokens=250, steps=8, lr=0.5, l2_weight=1.0, sim_scale=100.0, track_trajectories=True):
    print(f"\n[DEBUG Treatment] Starting treatment generation (Max tokens: {max_tokens}, Steps/token: {steps})...")
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    generated_ids = inputs.input_ids.clone()

    all_trajectories = []

    print("[DEBUG Treatment] Cloning centroids to float32...")
    # Centroids are in float32 for stable gradient calculations
    ideal_c = ideal_centroid.clone().to(torch.float32)
    corrupt_c = corrupt_centroid.clone().to(torch.float32)

    for token_idx in range(max_tokens):
        print(f"\n[DEBUG Treatment] --- Processing Token {token_idx + 1}/{max_tokens} ---")

        with torch.no_grad():
            outputs = base_model(input_ids=generated_ids, output_hidden_states=True)
            initial_hidden = outputs.hidden_states[-1][:, -1, :]
        print(f"[DEBUG Treatment] Forward pass complete. Initial hidden state shape: {initial_hidden.shape}")

        # Cast latent state to float32 to prevent 16-bit underflow of micro-gradients
        h_latent = initial_hidden.clone().detach().to(torch.float32).requires_grad_(True)
        optimizer = torch.optim.Adam([h_latent], lr=lr)

        token_trajectory = []

        print(f"[DEBUG Treatment] Beginning {steps} gradient ascent steps...")
        for step in range(steps):
            optimizer.zero_grad()

            # Calculate similarities and distances in float32
            sim_good = F.cosine_similarity(h_latent, ideal_c.unsqueeze(0))
            sim_bad = F.cosine_similarity(h_latent, corrupt_c.unsqueeze(0))
            manifold_dist = torch.norm(h_latent - initial_hidden.to(torch.float32), p=2)

            # Apply sim_scale to balance similarity gradients against the L2 norm vector
            objective = ((sim_good - (0.5 * sim_bad)) * sim_scale) - (l2_weight * manifold_dist)
            loss = -objective

            if track_trajectories:
                token_trajectory.append({
                    "step": step + 1,
                    "loss": loss.item(),
                    "sim_good": sim_good.item(),
                    "sim_bad": sim_bad.item(),
                    "manifold_dist": manifold_dist.item()
                })

            loss.backward()
            optimizer.step()

            # Print first, middle, and last step to avoid too much spam, but verify progress
            if step == 0 or step == steps//2 or step == steps - 1:
                print(f"    [Step {step + 1}/{steps}] Loss: {loss.item():.4f} | Sim_Good: {sim_good.item():.4f} | Manifold_Dist: {manifold_dist.item():.4f}")

        print(f"[DEBUG Treatment] Gradient steps finished for Token {token_idx + 1}.")

        if track_trajectories:
            all_trajectories.append(token_trajectory)

        # Cast back to the model's original dtype (float16) before generation
        with torch.no_grad():
            normed_h = norm_layer(h_latent.to(initial_hidden.dtype))
            logits = base_model.lm_head(normed_h)
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

            # Decode the selected token to show user the exact word being produced
            new_word = tokenizer.decode(next_token_id[0])
            print(f"[DEBUG Treatment] Generated token ID: {next_token_id.item()} -> Word: '{new_word}'")

        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

        if next_token_id.item() == tokenizer.eos_token_id:
            print(f"[DEBUG Treatment] EOS token reached at token {token_idx + 1}. Breaking loop.")
            break

    print("\n[DEBUG Treatment] Treatment generation completed successfully.")
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True), all_trajectories

if __name__ == "__main__":
    # 1. Ingest a single prompt string passed by the OS
    prompt_text = sys.argv[1]
    
    # 2. Generate responses at the default max_tokens (250) limit
    rejected_text = generate_control(prompt_text)
    chosen_text, _ = generate_treatment(prompt_text, track_trajectories=False)
    
    # 3. Construct the DPO dictionary map and print to standard output
    dpo_pair = {
        "prompt": prompt_text,
        "chosen": chosen_text,
        "rejected": rejected_text
    }
    
    print(json.dumps(dpo_pair))

```

---

### Step 2: Prepare Input Data (`prompts.txt`)

Create a standard text file named `prompts.txt` in the same directory as the Python script.

* Place exactly one prompt on each line.
* Verify there are no blank lines or trailing spaces at the end of the file to prevent the OS loop from passing empty arguments to the Python script.

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
