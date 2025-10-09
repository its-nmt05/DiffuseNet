import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
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
def inference_batch(image_paths, prompt, model, processor, device='cuda'):
    messages_batch = [
        [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }]
        for image_path in image_paths
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
def generate_captions(hf_cache_dir, frames_dir, output_file, prompt, batch_size, device='cuda'):
    image_paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    model, processor = get_model_and_processor(hf_cache_dir)

    captions = []   # placeholder for captions 
    
    print(f"Generating captions for {len(image_paths)} images in \"{frames_dir}\"")

    # process batches
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        batch_paths = image_paths[i:i + batch_size]
        # prompts = [prompt] * len(batch_paths)
        batch_captions = inference_batch(batch_paths, prompt, model, processor,  device)

        for path, cap in zip(batch_paths, batch_captions):
            captions.append([os.path.basename(path), cap])

    # write paths and captions as CSV rows
    with open(output_file, "w", newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["image", "caption"])  # write header
        writer.writerows(captions)   

    print(f"Output captions saved to {output_file}")


if __name__ == '__main__':
    SYSTEM_PROMPT = """
    You generate text prompts for a Pokemon gameplay image dataset to train a diffusion model (DiT).  

    Rules:
    1. Describe the scene, subject, and context clearly.
    2. Include action, pose, perspective, and interaction if applicable.
    3. Use template:
    [Subject/Character], [Action/Interaction], [Setting/Environment], [Lighting/Time]
    4. Avoid vague terms like “nice” or “good-looking”
    5. Each prompt must be 15-30 words long

    Examples:
    Pikachu using thunderbolt against Bulbasaur in a rocky arena, retro pixel-art style, night setting, side view, glowing sparks filling the scene.  
    Ash holding a Poké Ball in a grassy battlefield, cel-shaded anime style, bright sunlight, medium shot, crowd cheering with excitement.
    """

    parser = argparse.ArgumentParser(description="Generate image captions using Qwen2.5-VL")
    parser.add_argument("--hf_cache_dir", type=str, required=True,
                        help="Path to Hugging Face cache directory")
    parser.add_argument("--frames_dir", type=str, required=True,
                        help="Path to dataset directory containing frames")
    parser.add_argument("--prompt", type=str, default=SYSTEM_PROMPT,
                        help="Prompt instruction for caption generation")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference (e.g., 'cuda' or 'cpu')")

    args = parser.parse_args()

    # create captions.csv inside dataset dir
    output_file = os.path.join(os.path.dirname(args.frames_dir), "captions.csv")

    generate_captions(
        hf_cache_dir=args.hf_cache_dir,
        frames_dir=args.frames_dir,
        output_file=output_file,
        prompt=args.prompt,
        batch_size=args.batch_size,
        device=args.device
    )