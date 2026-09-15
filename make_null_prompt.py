"""
Precompute the one UMT5 null-prompt embedding the SkyReels world-model row needs.

SkyReels-V2 DF 1.3B is a text-to-video transformer: its only non-timestep conditioning is
cross-attention over a UMT5-XXL sequence. We never vary the text, so we run the encoder exactly
once, here, and save the tensor. `video_wm.SkyReelsWorldModel` loads it as a NON-persistent
buffer, so training and evaluation never instantiate the 22.7 GB text encoder and the tensor
never enters a checkpoint.

The embedding is built the way `SkyReelsV2DiffusionForcingPipeline._get_t5_prompt_embeds`
builds it (pipeline lines 176-215): tokenize with padding to `max_sequence_length`, run the
encoder with the attention mask, then keep the real tokens and ZERO-pad the rest back to full
length. The pipeline's `__call__` defaults to `max_sequence_length=512`, so that is the default
here. Note the zero rows are not inert: the transformer's `PixArtAlphaTextProjection` has
biases, so a shorter tensor is a different conditioning, not a cheaper equivalent.

Memory: the encoder is UMT5-XXL, 5,680,910,336 parameters, stored fp32 across five shards
(22,723,641,344 bytes = 21.2 GiB, from text_encoder/model.safetensors.index.json).
    --dtype bfloat16 (default)  10.6 GiB of weights plus roughly 1 GiB of workspace, so it fits
                                on one A6000 or in CPU RAM on Spiderman.
    --dtype float32             21.2 GiB of weights; use --device cpu unless the card is empty.
Shards are streamed (low_cpu_mem_usage is the default), so peak RSS stays near the final weight
size rather than double it. The 22.7 GB download dominates the wall clock; the encode itself is
one forward pass over 512 tokens. Needed once per prompt string, then never again.

    python make_null_prompt.py --out weights/skyreels_null_prompt.pt
    python make_null_prompt.py --prompt "" --device cuda --dtype bfloat16
"""
import argparse
import os

import torch


def main(args):
    from transformers import AutoTokenizer, UMT5EncoderModel

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.skyreels_path, subfolder="tokenizer", cache_dir=args.hf_cache)
    # `torch_dtype` is the spelling transformers 4.x wants; 5.x renamed it to `dtype` but still
    # accepts this one with a deprecation warning, so one line covers both server environments.
    enc = UMT5EncoderModel.from_pretrained(args.skyreels_path, subfolder="text_encoder", cache_dir=args.hf_cache,
                                           torch_dtype=dtype, low_cpu_mem_usage=True)
    enc = enc.to(args.device).eval()

    # `prompt_clean` in the pipeline only collapses whitespace and unescapes HTML; for the empty
    # prompt and for plain ASCII captions it is the identity, so we tokenize the string as given.
    batch = tok([args.prompt], padding="max_length", max_length=args.max_sequence_length,
                truncation=True, add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
    ids, mask = batch.input_ids, batch.attention_mask
    seq_len = int(mask.gt(0).sum())

    with torch.no_grad():
        hidden = enc(ids.to(args.device), mask.to(args.device)).last_hidden_state[0].float().cpu()
    embed = torch.zeros(1, args.max_sequence_length, hidden.shape[-1], dtype=torch.float32)
    embed[0, :seq_len] = hidden[:seq_len]        # real tokens kept, the tail zeroed (pipeline 205-208)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"embed": embed, "prompt": args.prompt, "seq_len": seq_len,
                "max_sequence_length": args.max_sequence_length, "model": args.skyreels_path,
                "dtype": args.dtype}, args.out)
    print(f"wrote {args.out}: shape {tuple(embed.shape)}, {seq_len} real tokens, "
          f"norm {embed.norm():.4f}, {os.path.getsize(args.out) / 2**20:.1f} MiB")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--skyreels-path", default="Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers")
    p.add_argument("--prompt", default="", help="the text to encode; empty is the null prompt")
    p.add_argument("--out", default="weights/skyreels_null_prompt.pt")
    p.add_argument("--max-sequence-length", type=int, default=512,
                   help="the DF pipeline's __call__ default; keep it unless you retrain the row")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--hf-cache", default=None)
    main(p.parse_args())
