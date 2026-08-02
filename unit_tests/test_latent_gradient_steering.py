import os
import pytest
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from latent_gradient_steering import (
    get_centroid,
    generate_control,
    generate_treatment,
    generate_dpo_dataset,
    get_norm_layer
)

@pytest.fixture(scope="module")
def setup_model_and_tokenizer():
    model_name = "hf-internal-testing/tiny-random-GPT2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = "cpu"
    model = model.to(device)
    return model, tokenizer, device

def test_get_norm_layer(setup_model_and_tokenizer):
    model, _, _ = setup_model_and_tokenizer
    norm_layer = get_norm_layer(model)
    # Since GPT2 has ln_f as its final LayerNorm layer, we verify it is returned
    assert isinstance(norm_layer, torch.nn.LayerNorm)

def test_get_centroid(setup_model_and_tokenizer):
    model, tokenizer, device = setup_model_and_tokenizer
    texts = ["This is a test exemplar.", "Another short example text."]

    centroid = get_centroid(texts, tokenizer, model, device=device)

    # Assert centroid is a PyTorch tensor with the correct hidden dimension (32 for tiny-random-GPT2)
    assert isinstance(centroid, torch.Tensor)
    assert centroid.dim() == 1
    assert centroid.shape[0] == 32

def test_generate_control(setup_model_and_tokenizer):
    model, tokenizer, device = setup_model_and_tokenizer
    prompt = "Predict the next word"

    output_text = generate_control(prompt, tokenizer, model, max_tokens=5, device=device)

    assert isinstance(output_text, str)
    assert len(output_text) > len(prompt)

def test_generate_treatment_and_optimization_validation(setup_model_and_tokenizer):
    model, tokenizer, device = setup_model_and_tokenizer
    prompt = "Analyze the evidence"

    # Construct distinct dummy exemplars for positive/negative targets
    pos_exemplars = ["The experiment confirmed the physical isotopic anomalies directly.", "Empirical heavy metal residues were found."]
    neg_exemplars = ["Alien structures vanished leaving zero rocks or evidence behind.", "Magic shields protected them leaving no residue."]

    ideal_centroid = get_centroid(pos_exemplars, tokenizer, model, device=device)
    corrupt_centroid = get_centroid(neg_exemplars, tokenizer, model, device=device)

    # Run the treatment generation with trajectory tracking enabled
    # We use a higher steps value (e.g. 8) to verify that loss decreases during optimization.
    output_text, trajectories = generate_treatment(
        prompt_text=prompt,
        tokenizer=tokenizer,
        model=model,
        ideal_centroid=ideal_centroid,
        corrupt_centroid=corrupt_centroid,
        max_tokens=3,
        steps=8,
        lr=0.5,
        l2_weight=1.0,
        sim_scale=100.0,
        track_trajectories=True,
        device=device
    )

    # Verify outputs
    assert isinstance(output_text, str)
    assert len(output_text) > len(prompt)
    assert len(trajectories) > 0

    # Validate optimization correctness: check that gradient ascent successfully optimized the objective
    # This means the loss should overall decrease from the first step to the last step for the generated tokens.
    for token_idx, token_traj in enumerate(trajectories):
        assert len(token_traj) == 8
        first_step = token_traj[0]
        last_step = token_traj[-1]

        # In generate_treatment, loss = -objective, where we want to maximize objective.
        # So we expect loss at the last step to be lower than or equal to loss at the first step.
        print(f"Token {token_idx} Loss: {first_step['loss']:.4f} -> {last_step['loss']:.4f}")
        assert last_step['loss'] <= first_step['loss']

        # Verify that sim_good overall increases, or sim_bad decreases, or objective improves
        first_obj = (first_step['sim_good'] - 0.5 * first_step['sim_bad']) * 100.0 - first_step['manifold_dist']
        last_obj = (last_step['sim_good'] - 0.5 * last_step['sim_bad']) * 100.0 - last_step['manifold_dist']
        assert last_obj >= first_obj

def test_generate_dpo_dataset(setup_model_and_tokenizer, tmp_path):
    model, tokenizer, device = setup_model_and_tokenizer
    prompts = [
        "What is the Silurian hypothesis?",
        "Explain the Fermi paradox."
    ]

    pos_exemplars = ["Isotopic anomalies leave clear traces.", "Chemical layers verify the event."]
    neg_exemplars = ["Fictional stories with zero evidence.", "Hand-waving theory with no traces."]

    ideal_centroid = get_centroid(pos_exemplars, tokenizer, model, device=device)
    corrupt_centroid = get_centroid(neg_exemplars, tokenizer, model, device=device)

    output_file = tmp_path / "test_dpo.jsonl"

    dataset = generate_dpo_dataset(
        prompts=prompts,
        tokenizer=tokenizer,
        model=model,
        ideal_centroid=ideal_centroid,
        corrupt_centroid=corrupt_centroid,
        output_file=str(output_file),
        max_tokens=3,
        steps=3,
        lr=0.1,
        l2_weight=1.0,
        sim_scale=10.0,
        device=device
    )

    # Verify the structure and length of the generated dataset
    assert len(dataset) == len(prompts)
    assert os.path.exists(output_file)

    with open(output_file, 'r') as f:
        lines = f.readlines()
        assert len(lines) == len(prompts)
        for line in lines:
            data = json.loads(line)
            assert "prompt" in data
            assert "chosen" in data
            assert "rejected" in data
            assert isinstance(data["prompt"], str)
            assert isinstance(data["chosen"], str)
            assert isinstance(data["rejected"], str)
