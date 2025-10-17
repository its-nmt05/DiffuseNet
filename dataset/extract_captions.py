import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import numpy as np
import glob
import os
from tqdm import tqdm
import csv
import argparse


# load VLM and processor from HF
def get_model_and_processor(hf_cache_dir):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        dtype=torch.float16,
        cache_dir=hf_cache_dir,
        device_map='auto'
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", cache_dir=hf_cache_dir)
    return model, processor


# running inference for batched images 
def inference_batch(frames_batch, prompt, model, processor, device='cuda'):
    messages_batch = [
        [{
            "role": "user",
            "content": [
                {"type": "image", "image": frame},
                {"type": "text", "text": prompt},
            ],
        }]
        for frame in frames_batch
    ]

    # Prepare the input for the processor
    inputs = processor.apply_chat_template(
        messages_batch, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_tensors="pt",
        padding=True,
        return_dict=True
    )
    inputs = inputs.to(device)

    # Perform batch inference for all images
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_texts


# process the entire dataset
def generate_captions(hf_cache_dir, npz_file, output_file, prompt, batch_size, device='cuda'):
    data = np.load(npz_file, mmap_mode='r')
    frames = data["frames"]
    model, processor = get_model_and_processor(hf_cache_dir)

    captions = []   # placeholder for captions 
    
    print(f"Generating captions for {len(frames)} frames in \"{npz_file}\"")

    # process batches
    for i in tqdm(range(0, len(frames), batch_size), desc="Processing batches"):
        batch_frames = frames[i:i + batch_size]
        batch_captions = inference_batch(batch_frames, prompt, model, processor,  device)

        for idx, cap in enumerate(batch_captions):
            frame_idx = i + idx
            captions.append([frame_idx, cap])

    # write paths and captions as CSV rows
    with open(output_file, "w", newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["idx", "caption"])  # write header
        writer.writerows(captions)   

    print(f"Output captions saved to {output_file}")


if __name__ == '__main__':
    SYSTEM_PROMPT = """
    You generate text prompts for a Pokémon gameplay image dataset to train a diffusion model (DiT).

    Rules:
    1. Avoid Pokémon-specific terms, names, or move names (no "Pikachu", "Thunderbolt", "Poké Ball", etc.).
    2. Use generic but vivid descriptions understandable by CLIP (e.g., "small yellow creature emitting electricity").
    3. Clearly describe subject, action, environment, and mood.
    4. Use the template:
    [Subject/Character], [Action/Interaction], [Setting/Environment], [Lighting/Time/Style]
    5. Each prompt must be 15-30 words long.
    6. Focus on general concepts like “creature,” “battlefield,” “forest,” “energy blast,” “trainer,” “arena,” etc.
    7. Avoid vague adjectives like “nice” or “cool.”

    Examples:
    Small yellow creature releasing electric sparks toward a green reptilian monster, rocky arena, cinematic style, night lighting, sparks illuminating the scene.  
    Young trainer throwing a red-and-white sphere toward a glowing beast, grassy field, anime-inspired art style, bright daylight, energetic action shot.
    """

    parser = argparse.ArgumentParser(description="Generate image captions using Qwen2.5-VL")
    parser.add_argument("--hf_cache_dir", type=str, required=True)
    parser.add_argument("--npz_file", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=SYSTEM_PROMPT)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # create captions.csv inside dataset dir
    output_file = os.path.join(os.path.dirname(args.npz_file), "captions.csv")

    generate_captions(
        hf_cache_dir=args.hf_cache_dir,
        npz_file=args.npz_file,
        output_file=output_file,
        prompt=args.prompt,
        batch_size=args.batch_size,
        device=args.device
    )