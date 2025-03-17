import time
from original_speculative_decoding import SpeculativeDecoder as originalDecoder
from speculative_decoding import SpeculativeDecoderExp
from typing import Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
import numpy as np
import sys
import os


from iree import compiler as ireec
import iree.turbine.aot as aot

import subprocess


# from transformers.onnx import export

def benchmark(
    main_model_name: str,
    draft_model_name: str,
    top_p, # Feel free to pick either top_p or top_k.
    top_k,
    max_tokens) -> Tuple[float, float]:
    """ Runs speculative decoding and compare it against greedy decoding with
    the main model.

    Returns:
      A tuple of (spec_decoding_runtime, greedy_decoding_runtime)
    """
    # TODO: Implement.
    decoder = SpeculativeDecoder(main_model_name, draft_model_name)
    
    prompt = "Once upon a time,"
    
    start_time = time.time()
    greedy_output = decoder.target_generate_greedy(
        prompt,
        max_new_tokens=max_tokens,
    )
    greedy_decoding_runtime = time.time() - start_time

    start_time = time.time()
    spec_output = decoder.generate(
        prompt,
        temperature=1.0,
        top_k=top_k,
        top_p=top_p,
        gamma=5,
        max_new_tokens=max_tokens
    )
    spec_decoding_runtime = time.time() - start_time
    
    print("Speculative Decoding Output:")
    print(spec_output)
    print("\nGreedy Decoding Output:")
    print(greedy_output)
    
    return (spec_decoding_runtime, greedy_decoding_runtime)

def calculate_tokens_per_second(num_tokens: int, runtime: float) -> float:
    """
    Calculate the number of tokens generated per second.

    Args:
        num_tokens (int): The number of tokens generated.
        runtime (float): The time taken to generate the tokens, in seconds.

    Returns:
        float: The number of tokens generated per second.
    """
    return num_tokens / runtime if runtime > 0 else 0

def benchmark_v2(
    main_model_name: str,
    draft_model_name: str,
    top_p: float,
    top_k: int,
    max_tokens: int,
    num_runs: int = 5
) -> Tuple[float, float, float, float]:
    """
    Runs speculative decoding and compares it against greedy decoding with the main model.

    Args:
        main_model_name (str): Name of the main (larger) model.
        draft_model_name (str): Name of the draft (smaller) model.
        top_p (float): The cumulative probability threshold for top-p sampling.
        top_k (int): The number of top tokens to consider for top-k sampling.
        max_tokens (int): The maximum number of new tokens to generate.
        num_runs (int): Number of runs to average over. Defaults to 5.

    Returns:
        A tuple of (avg_spec_decoding_runtime, avg_greedy_decoding_runtime, 
                    avg_spec_tokens_per_second, avg_greedy_tokens_per_second)
    """
    decoder = SpeculativeDecoderExp(main_model_name, draft_model_name)
    prompt = "Once upon a time"
    # prompt = "Say something about the letter A"
    
    tokenizer = AutoTokenizer.from_pretrained(main_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to('cpu')
    len_encoded_prompt = len(tokenizer.encode(prompt))       

    example_encoding = tokenizer(prompt, return_tensors="pt")
    example_input = example_encoding["input_ids"].cpu()
    example_args = (example_input,)

    if True:
        generated_ids = decoder(input_ids)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print("-- Check our modifications for exporting:")
        print("Output from modified forward function: ")
        print(generated_text)
        print('\n')
        tester = originalDecoder(main_model_name, draft_model_name)
        spec_output = tester.generate(
            prompt,
            temperature=1.0,
            top_k=top_k,
            top_p=top_p,
            gamma=5,
            max_new_tokens=max_tokens
        )
        print("Output from original generate function: ")
        print(spec_output)
        print("-- End of sanity check")

   
    mlir_file_name = 'decoder_compiled.mlir'
    exec_file_name = 'decoder.vmfb'
    sample_input_file_name = 'sample_input.npy'

    if not os.path.isfile(exec_file_name):
        print('Exporting')
        decoder_compiled = aot.export(decoder, *example_args)
        decoder_compiled.import_to("TORCH_IR")
        print('Compiling')
        original = sys.stdout
        with open(mlir_file_name, 'w') as sys.stdout:
            decoder_compiled.mlir_module.print()
        sys.stdout = original

        # TODO: input size gets 'hard-coded# in to the mlir. Need dynamic shape here
        compiled_flatbuffer = ireec.tools.compile_file(
            mlir_file_name,
            output_file=exec_file_name,
            target_backends=["llvm-cpu"],
            extra_args=[
                        # "--iree-llvmcpu-debug-symbols=false",
                        "--iree-hal-executable-debug-level=3",
                        "--iree-llvmcpu-target-cpu=host",
                        "--iree-llvmcpu-target-cpu-features=host"
                        ]
        )
        print('Compiled')
        # run with:
        # iree-run-module --module=decoder.vmfb --input=@sample_input.npy --device=local-task
    
    print('Dumping sample input (for iree runtime) to file and exiting')
    if os.path.exists("sample_input.npy"):
        os.remove("sample_input.npy")
    if os.path.exists("sample_output.npy"):
        os.remove("sample_output.npy")
    np.save(sample_input_file_name,example_args[0].to())
    print("Running compiled module on IREE")
    print(subprocess.run(["iree-run-module --module=decoder.vmfb --input=@sample_input.npy --device=local-task --output=@sample_output.npy"], 
                     shell=True, capture_output=True))
    iree_data = np.load('sample_output.npy')
    iree_text = tokenizer.decode(iree_data[0], skip_special_tokens=True)
    print("Text output from IREE:")
    print(iree_text)
    print("\n")

    print("Running original benchmark stuff...:")
    spec_runtimes = []
    greedy_runtimes = []
    spec_tokens_per_second = []
    greedy_tokens_per_second = []

    for _ in range(num_runs):
        # Benchmark speculative decoding
        start_time = time.time()
        spec_output = decoder.generate(
            # prompt,
            input_ids,
            tokenizer.pad_token_id,
            len_encoded_prompt,
            temperature=1.0,
            top_k=top_k,
            top_p=top_p,
            gamma=5,
            max_new_tokens=max_tokens
        )
        spec_runtime = time.time() - start_time
        spec_runtimes.append(spec_runtime)
        
        spec_output = tokenizer.decode(spec_output, skip_special_tokens=True)
        spec_tokens = len(tokenizer.encode(spec_output)) - len(tokenizer.encode(prompt))
        spec_tokens_per_second.append(calculate_tokens_per_second(spec_tokens, spec_runtime))

        # Benchmark target greedy decoding
        # start_time = time.time()
        # greedy_output = decoder.target_generate_greedy(
        #     prompt,
        #     max_new_tokens=max_tokens,
        # )
        # greedy_runtime = time.time() - start_time
        greedy_runtime = 0
        greedy_runtimes.append(greedy_runtime)
        
        # greedy_tokens = len(tokenizer.encode(greedy_output)) - len(tokenizer.encode(prompt))
        # greedy_tokens_per_second.append(calculate_tokens_per_second(greedy_tokens, greedy_runtime))

    avg_spec_runtime = sum(spec_runtimes) / num_runs
    avg_greedy_runtime = sum(greedy_runtimes) / num_runs
    avg_spec_tokens_per_second = sum(spec_tokens_per_second) / num_runs
    # avg_greedy_tokens_per_second = sum(greedy_tokens_per_second) / num_runs
    avg_greedy_tokens_per_second = 0

    print(f"\nAverage over {num_runs} runs:")
    print(f"Speculative Decoding Runtime: {avg_spec_runtime:.2f} seconds")
    print(f"Greedy Decoding Runtime: {avg_greedy_runtime:.2f} seconds")
    print(f"Speedup: {avg_greedy_runtime / avg_spec_runtime:.2f}x")
    print(f"Speculative Decoding Tokens/second: {avg_spec_tokens_per_second:.2f}")
    # print(f"Greedy Decoding Tokens/second: {avg_greedy_tokens_per_second:.2f}")

    # Print sample outputs for verification
    print("\nSample Speculative Decoding Output:")
    print(spec_output)
    # print("\nSample Greedy Decoding Output:")
    # print(greedy_output)

    return (avg_spec_runtime, avg_greedy_runtime, avg_spec_tokens_per_second, avg_greedy_tokens_per_second)
    
if __name__ == '__main__':
    
    main_model = "gpt2-medium"
    draft_model = "distilgpt2"
    top_p = 1
    top_k = 0
    max_tokens = 100

    # spec_time, greedy_time = benchmark(main_model, draft_model, top_p, top_k, max_tokens)
    # print(f"\nSpeculative Decoding Runtime: {spec_time:.2f} seconds")
    # print(f"Greedy Decoding Runtime: {greedy_time:.2f} seconds")
    # print(f"Speedup: {greedy_time / spec_time:.2f}x")
    # to get avg over multiple runs, un-comment below line
    benchmark_v2(main_model, draft_model, top_p, top_k, max_tokens)

    