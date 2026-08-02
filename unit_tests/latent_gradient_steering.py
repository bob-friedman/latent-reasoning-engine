"""
latent_gradient_steering.py

This module contains the core logic for Latent Gradient Steering.
It implements continuous gradient-based optimization on the pre-unembedding hidden states
of a language model to steer its autoregressive generation.
"""

import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_norm_layer(model):
    """
    Retrieves the final normalization layer of the language model dynamically.
    Supports Mistral/Llama-like models and GPT2-like models.
    """
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    return torch.nn.Identity()

def get_centroid(texts, tokenizer, model, device=None):
    """
    Extracts the mean-pooled hidden state centroid across a list of text exemplars.
    """
    if device is None:
        device = next(model.parameters()).device

    states = []
    # Ensure model is in evaluation mode
    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            out = model(**inputs, output_hidden_states=True)

            # Mean-pool across all tokens in the sequence to capture semantic meaning
            hidden_states = out.hidden_states[-1].squeeze(0)  # [seq_len, hidden_dim]
            mean_pooled = torch.mean(hidden_states, dim=0)    # [hidden_dim]
            states.append(mean_pooled.unsqueeze(0))

    return torch.mean(torch.cat(states, dim=0), dim=0)

def generate_control(prompt_text, tokenizer, model, max_tokens=250, device=None):
    """
    Generates a baseline (unsteered) response from the language model using greedy decoding.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    print("\n[DEBUG Control] Starting control generation...")
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    print("[DEBUG Control] Control generation completed.")
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_treatment(
    prompt_text,
    tokenizer,
    model,
    ideal_centroid,
    corrupt_centroid,
    norm_layer=None,
    max_tokens=250,
    steps=8,
    lr=0.5,
    l2_weight=1.0,
    sim_scale=100.0,
    track_trajectories=True,
    device=None
):
    """
    Generates a steered (treatment) response from the language model by performing
    gradient ascent on the pre-unembedding hidden states at each generation step.
    """
    if device is None:
        device = next(model.parameters()).device

    if norm_layer is None:
        norm_layer = get_norm_layer(model)

    model.eval()
    print(f"\n[DEBUG Treatment] Starting treatment generation (Max tokens: {max_tokens}, Steps/token: {steps})...")
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    generated_ids = inputs.input_ids.clone()

    all_trajectories = []

    print("[DEBUG Treatment] Cloning centroids to float32...")
    ideal_c = ideal_centroid.clone().to(torch.float32).to(device)
    corrupt_c = corrupt_centroid.clone().to(torch.float32).to(device)

    for token_idx in range(max_tokens):
        print(f"\n[DEBUG Treatment] --- Processing Token {token_idx + 1}/{max_tokens} ---")

        with torch.no_grad():
            outputs = model(input_ids=generated_ids, output_hidden_states=True)
            initial_hidden = outputs.hidden_states[-1][:, -1, :]
        print(f"[DEBUG Treatment] Forward pass complete. Initial hidden state shape: {initial_hidden.shape}")

        # Cast latent state to float32 to prevent 16-bit underflow of micro-gradients
        h_latent = initial_hidden.clone().detach().to(torch.float32).to(device).requires_grad_(True)
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

            # Print progress for verification
            if step == 0 or step == steps // 2 or step == steps - 1:
                print(f"    [Step {step + 1}/{steps}] Loss: {loss.item():.4f} | Sim_Good: {sim_good.item():.4f} | Manifold_Dist: {manifold_dist.item():.4f}")

        print(f"[DEBUG Treatment] Gradient steps finished for Token {token_idx + 1}.")

        if track_trajectories:
            all_trajectories.append(token_trajectory)

        # Cast back to the model's original dtype before generation
        with torch.no_grad():
            normed_h = norm_layer(h_latent.to(initial_hidden.dtype))
            logits = model.lm_head(normed_h)
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

            # Decode the selected token
            new_word = tokenizer.decode(next_token_id[0])
            print(f"[DEBUG Treatment] Generated token ID: {next_token_id.item()} -> Word: '{new_word}'")

        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

        if next_token_id.item() == tokenizer.eos_token_id:
            print(f"[DEBUG Treatment] EOS token reached at token {token_idx + 1}. Breaking loop.")
            break

    print("\n[DEBUG Treatment] Treatment generation completed successfully.")
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True), all_trajectories

def generate_dpo_dataset(
    prompts,
    tokenizer,
    model,
    ideal_centroid,
    corrupt_centroid,
    output_file="dpo_dataset.jsonl",
    max_tokens=150,
    steps=8,
    lr=0.5,
    l2_weight=1.0,
    sim_scale=100.0,
    device=None
):
    """
    Generates a DPO dataset containing prompt, chosen (steered), and rejected (unsteered) pairs.
    """
    dpo_data = []

    for idx, prompt_text in enumerate(prompts):
        print(f"\nProcessing prompt {idx + 1}/{len(prompts)} for DPO dataset...")

        # 1. Generate the Rejected response (Default/Unsteered)
        rejected_text = generate_control(
            prompt_text,
            tokenizer,
            model,
            max_tokens=max_tokens,
            device=device
        )

        # 2. Generate the Chosen response (Steered/Optimized)
        chosen_text, _ = generate_treatment(
            prompt_text,
            tokenizer,
            model,
            ideal_centroid,
            corrupt_centroid,
            max_tokens=max_tokens,
            steps=steps,
            lr=lr,
            l2_weight=l2_weight,
            sim_scale=sim_scale,
            track_trajectories=False,  # Turn off logging to speed up generation
            device=device
        )

        # 3. Append to DPO structure
        dpo_data.append({
            "prompt": prompt_text,
            "chosen": chosen_text,
            "rejected": rejected_text
        })

    # Export to JSONL
    with open(output_file, 'w') as f:
        for item in dpo_data:
            f.write(json.dumps(item) + '\n')

    print(f"Successfully saved {len(prompts)} preference pairs to {output_file}")
    return dpo_data
