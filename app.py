import json
import os
import glob
import sys
import time
from pathlib import Path
from typing import Tuple

from huggingface_hub import hf_hub_download
from PIL import Image
import gradio as gr
import torch
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import LLaMA, ModelArgs, Tokenizer, Transformer, VisionModel

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def setup_model_parallel() -> Tuple[int, int]:
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MP'] = '1'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '2223'
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt0_path: str,
    ckpt1_path: str,
    param_path: str,
    tokenizer_path: str,
    instruct_adapter_path: str,
    caption_adapter_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    print("Loading")
    instruct_adapter_checkpoint = torch.load(
        instruct_adapter_path, map_location="cpu")
    caption_adapter_checkpoint = torch.load(
        caption_adapter_path, map_location="cpu")
    with open(param_path, "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    model_args.adapter_layer = int(
        instruct_adapter_checkpoint['adapter_query.weight'].shape[0] / model_args.adapter_len)
    model_args.cap_adapter_layer = int(
        caption_adapter_checkpoint['cap_adapter_query.weight'].shape[0] / model_args.cap_adapter_len)

    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)

    # To reduce memory usuage
    ckpt0 = torch.load(ckpt0_path, map_location='cuda')
    model.load_state_dict(ckpt0, strict=False)
    del ckpt0
    torch.cuda.empty_cache()

    ckpt1 = torch.load(ckpt1_path, map_location='cuda')
    model.load_state_dict(ckpt1, strict=False)
    del ckpt1
    torch.cuda.empty_cache()

    vision_model = VisionModel(model_args)

    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(instruct_adapter_checkpoint, strict=False)
    model.load_state_dict(caption_adapter_checkpoint, strict=False)
    vision_model.load_state_dict(caption_adapter_checkpoint, strict=False)

    generator = LLaMA(model, tokenizer, vision_model)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def instruct_generate(
    instruct: str,
    input: str = 'none',
    max_gen_len=512,
    temperature: float = 0.1,
    top_p: float = 0.75,
):
    if input == 'none':
        prompt = PROMPT_DICT['prompt_no_input'].format_map(
            {'instruction': instruct, 'input': ''})
    else:
        prompt = PROMPT_DICT['prompt_input'].format_map(
            {'instruction': instruct, 'input': input})

    results = generator.generate(
        [prompt], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p
    )
    result = results[0].strip()
    print(result)
    return result


def caption_generate(
    img: str,
    max_gen_len=512,
    temperature: float = 0.1,
    top_p: float = 0.75,
):
    imgs = [Image.open(img).convert('RGB')]
    prompts = ["Generate caption of this image :",] * len(imgs)

    results = generator.generate(
        prompts, imgs=imgs, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p
    )
    result = results[0].strip()
    print(result)
    return result


def download_llama_adapter(instruct_adapter_path, caption_adapter_path):
    if not os.path.exists(instruct_adapter_path):
        os.system(
            f"wget -q -O {instruct_adapter_path} https://github.com/ZrrSkywalker/LLaMA-Adapter/releases/download/v.1.0.0/llama_adapter_len10_layer30_release.pth")

    if not os.path.exists(caption_adapter_path):
        os.system(
            f"wget -q -O {caption_adapter_path} https://github.com/ZrrSkywalker/LLaMA-Adapter/releases/download/v.1.0.0/llama_adapter_len10_layer30_caption_vit_l.pth")


# ckpt_path = "/data1/llma/7B/consolidated.00.pth"
# param_path = "/data1/llma/7B/params.json"
# tokenizer_path = "/data1/llma/tokenizer.model"
ckpt0_path = hf_hub_download(
    repo_id="csuhan/llama_storage", filename="consolidated.00_part0.pth")
ckpt1_path = hf_hub_download(
    repo_id="csuhan/llama_storage", filename="consolidated.00_part1.pth")
param_path = hf_hub_download(
    repo_id="nyanko7/LLaMA-7B", filename="params.json")
tokenizer_path = hf_hub_download(
    repo_id="nyanko7/LLaMA-7B", filename="tokenizer.model")
instruct_adapter_path = "llama_adapter_len10_layer30_release.pth"
caption_adapter_path = "llama_adapter_len10_layer30_caption_vit_l.pth"
max_seq_len = 512
max_batch_size = 1

# download models
# download_llama_adapter(instruct_adapter_path, caption_adapter_path)

local_rank, world_size = setup_model_parallel()
if local_rank > 0:
    sys.stdout = open(os.devnull, "w")

generator = load(
    ckpt0_path, ckpt1_path, param_path, tokenizer_path, instruct_adapter_path, caption_adapter_path, local_rank, world_size, max_seq_len, max_batch_size
)


def create_instruct_demo():
    with gr.Blocks() as instruct_demo:
        with gr.Row():
            with gr.Column():
                instruction = gr.Textbox(lines=2, label="Instruction")
                input = gr.Textbox(
                    lines=2, label="Context input", placeholder='none')
                max_len = gr.Slider(minimum=1, maximum=512,
                                    value=128, label="Max length")
                with gr.Accordion(label='Advanced options', open=False):
                    temp = gr.Slider(minimum=0, maximum=1,
                                     value=0.1, label="Temperature")
                    top_p = gr.Slider(minimum=0, maximum=1,
                                      value=0.75, label="Top p")

                run_botton = gr.Button("Run")

            with gr.Column():
                outputs = gr.Textbox(lines=10, label="Output")

        inputs = [instruction, input, max_len, temp, top_p]

        examples = [
            "Tell me about alpacas.",
            "Write a Python program that prints the first 10 Fibonacci numbers.",
            "Write a conversation between the sun and pluto.",
            "Write a theory to explain why cat never existed",
        ]
        examples = [
            [x, "none", 128, 0.1, 0.75]
            for x in examples]

        gr.Examples(
            examples=examples,
            inputs=inputs,
            outputs=outputs,
            fn=instruct_generate,
            cache_examples=os.getenv('SYSTEM') == 'spaces'
        )
        run_botton.click(fn=instruct_generate, inputs=inputs, outputs=outputs)
    return instruct_demo


def create_caption_demo():
    with gr.Blocks() as instruct_demo:
        with gr.Row():
            with gr.Column():
                img = gr.Image(label='Input', type='filepath')
                max_len = gr.Slider(minimum=1, maximum=512,
                                    value=64, label="Max length")
                with gr.Accordion(label='Advanced options', open=False):
                    temp = gr.Slider(minimum=0, maximum=1,
                                     value=0.1, label="Temperature")
                    top_p = gr.Slider(minimum=0, maximum=1,
                                      value=0.75, label="Top p")

                run_botton = gr.Button("Run")

            with gr.Column():
                outputs = gr.Textbox(lines=10, label="Output")

        inputs = [img, max_len, temp, top_p]

        examples = glob.glob("caption_demo/*.jpg")
        examples = [
            [x, 64, 0.1, 0.75]
            for x in examples]

        gr.Examples(
            examples=examples,
            inputs=inputs,
            outputs=outputs,
            fn=caption_generate,
            cache_examples=os.getenv('SYSTEM') == 'spaces'
        )
        run_botton.click(fn=caption_generate, inputs=inputs, outputs=outputs)
    return instruct_demo


description = """
# LLaMA-Adapter🚀
The official demo for **LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention**.
Please refer to our [arXiv paper](https://arxiv.org/abs/2303.16199) and [github](https://github.com/ZrrSkywalker/LLaMA-Adapter) for more details.
"""

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(description)
    with gr.TabItem("Instruction-Following"):
        create_instruct_demo()
    with gr.TabItem("Image Captioning"):
        create_caption_demo()

demo.queue(api_open=True, concurrency_count=1).launch()
