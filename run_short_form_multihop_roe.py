#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
RoE-Enhanced version of run_short_form_multihop.py

Usage:
python run_short_form_multihop_roe.py \
    --model_name shayekh/openrag_llama2_7b_8x135m \
    --dataset shayekh/openrag_bench \
    --task hotpotqa \
    --mode adaptive_retrieval \
    --max_new_tokens 100 \
    --threshold 0.0 \
    --metric hotpotem \
    --ndocs 3 \
    --use_groundness \
    --use_utility \
    --use_seqscore \
    --use_roe \
    --roe_k 8 \
    --roe_tau 0.05 \
    --output_file ./eval_roe/hotpotqa.jsonl
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import torch
import numpy as np
from tqdm import tqdm
import json
import argparse
import sys
import os

# Add roe_implementation to path
roe_impl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'roe_implementation')
if roe_impl_path not in sys.path:
    sys.path.insert(0, roe_impl_path)

from roe_openrag import generate_with_roe, create_default_tau_map

from openrag.utils import (
    PROMPT_DICT,
    TASK_INST,
    load_jsonlines,
    control_tokens,
    load_special_tokens,
)
from datasets import load_dataset
from openrag.metrics import match, accuracy, hotpot_exact_match_score, hotpot_f1_score


TASK_INSTRUCTION = (
  f"You are a question answering agent. Given a context and a question, your task is to answer the question based on the context. "
  f"Instead of a full sentence, your answer must be the shortest word or phrase or named entity. "
  f"Some example outputs 'answer' are: yes; no; Ibn Sina; Doha, Qatar; 2,132 seats, Los Angeles, California etc." 
)

PROMPT_DICT["prompt_no_input"] = \
TASK_INSTRUCTION + "### Instruction:\n{instruction}\n\n### Response:\n"

seed = 633

torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def postprocess_answer_option_conditioned(answer):
    for token in control_tokens:
        answer = answer.replace(token, "")

    if "</s>" in answer:
        answer = answer.replace("</s>", "")
    if "\n" in answer:
        answer = answer.replace("\n", "")

    if "<|endoftext|>" in answer:
        answer = answer.replace("<|endoftext|>", "")

    return answer


from transformers import StoppingCriteria, StoppingCriteriaList


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, tokenizer, stops = [], encounters=1, device="cuda"):
        super().__init__()
        self.tokenizer = tokenizer
        self.stops = [stop.to(device) for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_token = input_ids[0][-1]
        for stop in self.stops:
            if stop == last_token:
                return True
        return False


def call_model_roe(
    tokenizer,
    prompt,
    model,
    max_new_tokens=15,
    use_roe=True,
    roe_k=8,
    tau_map=None,
    device="cuda",
):
    """
    Generate using RoE or baseline.
    
    Args:
        tokenizer: Tokenizer
        prompt: Input prompt
        model: Model
        max_new_tokens: Max tokens to generate
        use_roe: Whether to use RoE
        roe_k: Number of RoE samples
        tau_map: Temperature map for RoE
        device: Device
    
    Returns:
        Generated text
    """
    if use_roe:
        # Use RoE generation
        generated_text = generate_with_roe(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            K=roe_k,
            layer_tau_map=tau_map,
            device=device,
            use_greedy=True
        )
        return generated_text
    else:
        # Baseline generation
        stop_words = ["</s>"]
        stop_words_ids = [tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze() 
                          for stop_word in stop_words]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(tokenizer, stops=stop_words_ids)])
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        preds = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False,
            top_p=1.0,
            stopping_criteria=stopping_criteria,
        )
        
        pred_text = tokenizer.batch_decode(
            preds.sequences[:, inputs.input_ids.shape[1]:], 
            skip_special_tokens=True, 
        )[0]
        
        return pred_text


def call_model_rerank_w_scores_batch_roe(
    tokenizer,
    prompt,
    evidences,
    model,
    max_new_tokens=15,
    ret_tokens=None,
    rel_tokens=None,
    grd_tokens=None,
    ut_tokens=None,
    use_seqscore=False,
    threshold=0.5,
    w_rel=1.0,
    w_sup=1.0,
    w_use=0.5,
    mode="adaptive_retrieval",
    closed=False,
    use_roe=True,
    roe_k=8,
    tau_map=None,
    device="cuda",
):
    """
    RoE-enhanced version of call_model_rerank_w_scores_batch.
    
    Main changes:
    1. Uses call_model_roe() instead of model.generate()
    2. Simplified scoring (no detailed log probs extraction)
    3. Focuses on final predictions
    """
    stop_words = ["</s>"]
    stop_words_ids = [tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze() 
                      for stop_word in stop_words]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(tokenizer, stops=stop_words_ids)])

    detailed_init_score = {
        "logproba_retrieval_thresh": 0,
        "proba_retrieval_thresh": 0,
        "pred_retrieval_decision": "",
        "do_retrieve": None,
        "proba_r": "",
        "logr": "",
        "lognr": "",
        "proba_nr": "",
        "seq_logprob": ""
    }
    
    results = {}
    
    # Step 1: Decide whether to retrieve
    if mode != "always_retrieve":
        # Generate initial response to check for [Retrieval] token
        pred_text = call_model_roe(
            tokenizer=tokenizer,
            prompt=prompt,
            model=model,
            max_new_tokens=max_new_tokens,
            use_roe=use_roe,
            roe_k=roe_k,
            tau_map=tau_map,
            device=device
        )
        
        results["no_retrieval"] = pred_text
        pred = pred_text
        
        # Simple heuristic: check if [Retrieval] appears in output
        # (In RoE mode, we simplify decision logic since detailed logprobs are harder to extract)
        if threshold is not None:
            # Use heuristic: if no_retrieval prediction is very short or contains special tokens
            do_retrieve = len(pred_text.split()) < 3 or "[Retrieval]" in pred_text
        else:
            do_retrieve = "[Retrieval]" in pred_text
    
    if mode == "always_retrieve":
        do_retrieve = True
    elif mode == "no_retrieval":
        do_retrieve = False
    
    # Step 2: If retrieve, generate with each evidence
    if do_retrieve:
        evidence_predictions = []
        
        for p_idx, evidence in enumerate(evidences):
            augmented_prompt = prompt + "[Retrieval]<paragraph>{0}</paragraph>".format(evidence["text"])
            
            # Generate with RoE
            pred_text = call_model_roe(
                tokenizer=tokenizer,
                prompt=augmented_prompt,
                model=model,
                max_new_tokens=max_new_tokens,
                use_roe=use_roe,
                roe_k=roe_k,
                tau_map=tau_map,
                device=device
            )
            
            evidence_predictions.append(pred_text)
            
            # Simple scoring: length and coherence heuristic
            # RoE should produce more coherent responses, so we use length as proxy
            score = len(pred_text.split()) / max_new_tokens
            
            results[f"retrieval_{p_idx}"] = {
                "pred": pred_text,
                "score": score if pred_text != "</s>" else -1,
                "ctx": evidence,
            }
        
        # Select best prediction (longest coherent answer)
        if closed:
            # For closed tasks, aggregate by answer
            answer2score = {}
            for key, result in results.items():
                if key == "no_retrieval":
                    continue
                answer = postprocess_answer_option_conditioned(result["pred"])
                score = result["score"]
                answer2score.setdefault(answer, 0)
                answer2score[answer] += score
            
            sorted_answers = sorted(answer2score.items(), key=lambda x: x[1], reverse=True)
            best_option = sorted_answers[0][0]
        else:
            # For open tasks, pick highest scoring path
            path2score = {
                key: item["score"]
                for key, item in results.items()
                if key != "no_retrieval"
            }
            best_path = sorted(path2score.items(), key=lambda x: x[1], reverse=True)[0][0]
            best_option = results[best_path]["pred"]
        
        return best_option, results, do_retrieve, detailed_init_score
    
    else:
        # No retrieval: use initial prediction
        prompt += "[No Retrieval]"
        pred = call_model_roe(
            tokenizer=tokenizer,
            prompt=prompt,
            model=model,
            max_new_tokens=max_new_tokens,
            use_roe=use_roe,
            roe_k=roe_k,
            tau_map=tau_map,
            device=device
        )
        
        postprocessed_pred = postprocess_answer_option_conditioned(pred)
        return postprocessed_pred, results, do_retrieve, detailed_init_score


def process_data_evidences(demonstration, top_n):
    ctx_key = "ctxs" if "ctxs" in demonstration else "top_contexts"
    prompt = PROMPT_DICT["prompt_no_input"].format_map(demonstration)
    evidences = demonstration[ctx_key][:top_n]
    return prompt, evidences


def preprocess_input_data(dataset, task=None):
    new_data = []
    if task in TASK_INST:
        instruction = TASK_INST[task]
    else:
        instruction = None
    for item in dataset:
        if task == "arc_c":
            choices = item["choices"]
            answer_labels = {}
            for i in range(len(choices["label"])):
                answer_key = choices["label"][i]
                text = choices["text"][i]
                if answer_key == "1":
                    answer_labels["A"] = text
                if answer_key == "2":
                    answer_labels["B"] = text
                if answer_key == "3":
                    answer_labels["C"] = text
                if answer_key == "4":
                    answer_labels["D"] = text
                if answer_key in ["A", "B", "C", "D"]:
                    answer_labels[answer_key] = text

            if "D" not in answer_labels:
                answer_labels["D"] = ""
            choices = "\nA: {0}\nB: {1}\nC: {2}\nD: {3}".format(
                answer_labels["A"],
                answer_labels["B"],
                answer_labels["C"],
                answer_labels["D"],
            )
            if "E" in answer_labels:
                choices += "\nE: {}".format(answer_labels["E"])
            item["instruction"] = (
                instruction + "\n\n### Input:\n" + item["question"] + choices
            )
            item["answers"] = [item["answerKey"]]
        else:
            prompt = (
                instruction + "\n\n## Input:\n\n" + item["question"]
                if instruction is not None
                else item["question"]
            )
            item["instruction"] = prompt
        new_data.append(item)

    return new_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--input_file", type=str, default="None")
    parser.add_argument("--dataset", type=str, default="shayekh/openrag_bench")
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=15)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument(
        "--download_dir",
        type=str,
        help="specify vllm model download dir",
        default=".cache",
    )
    parser.add_argument(
        "--ndocs",
        type=int,
        default=10,
        help="Number of documents to retrieve per questions",
    )
    parser.add_argument(
        "--world_size", type=int, default=1, help="world size to use multiple GPUs."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="half",
        help="We use bfloat16 for training. If you run inference on GPUs that do not support BF16, please set this to be `half`.",
    )
    # Decoding hyperparams
    parser.add_argument(
        "--threshold", type=float, default=None, help="Adaptive threshold."
    )
    parser.add_argument("--use_seqscore", action="store_true")
    parser.add_argument(
        "--use_groundness", action="store_true", help="use ground score"
    )
    parser.add_argument("--use_utility", action="store_true", help="tree search")
    parser.add_argument("--beam_width", type=int, default=2, help="beam search width")
    parser.add_argument("--max_depth", type=int, default=2, help="tree depth width")
    parser.add_argument(
        "--w_rel", type=float, default=1.0, help="reward weight for document relevance"
    )
    parser.add_argument(
        "--w_sup",
        type=float,
        default=1.0,
        help="reward weight for generation support (attribution)",
    )
    parser.add_argument(
        "--w_use",
        type=float,
        default=1.0,
        help="reward weight for overall completeness / utility.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="mode to control retrieval.",
        default="default",
        choices=["adaptive_retrieval", "no_retrieval", "always_retrieve"],
    )
    parser.add_argument(
        "--metric", type=str, help="metric to be used during evaluation"
    )
    
    # RoE-specific arguments
    parser.add_argument(
        "--use_roe", 
        action="store_true", 
        help="Use RoE (Repetition of Experts) for generation"
    )
    parser.add_argument(
        "--roe_k", 
        type=int, 
        default=8, 
        help="Number of parallel RoE samples (default: 8)"
    )
    parser.add_argument(
        "--roe_tau", 
        type=float, 
        default=0.05, 
        help="Temperature for middle layers in RoE (default: 0.05, recommended: 0.01-0.1)"
    )
    
    args = parser.parse_args()
    
    # Print RoE configuration
    if args.use_roe:
        print("="*80)
        print("RoE (Repetition of Experts) ENABLED")
        print("="*80)
        print(f"RoE K (parallel samples): {args.roe_k}")
        print(f"RoE tau (temperature): {args.roe_tau}")
        print(f"Expected slowdown: ~{args.roe_k}x")
        print("="*80 + "\n")
    
    gpt = args.model_name
    input_path = args.input_file
    if input_path.endswith(".json"):
        input_data = json.load(open(input_path))
    elif input_path.endswith(".jsonl"):
        input_data = load_jsonlines(input_path)
    else:
        dataset = load_dataset(args.dataset, args.task)
        input_data = dataset['dev'].to_list()
        

    input_data = preprocess_input_data(input_data, task=args.task)
    tokenizer = AutoTokenizer.from_pretrained(gpt)
    model_args = {}
    if args.dtype == "half":
        model_args["torch_dtype"] = torch.float16
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        gpt, 
        device_map="cuda:0", 
        trust_remote_code=True,
        resume_download=True,
        **model_args,
    ).eval()
    
    print("Model loaded successfully!\n")
    
    # Create tau map for RoE
    tau_map = None
    if args.use_roe:
        num_layers = model.config.num_hidden_layers
        tau_map = create_default_tau_map(num_layers, tau_middle=args.roe_tau)
        print(f"Created tau map for {num_layers} layers")
        print(f"  First layer: tau={tau_map[0]}")
        print(f"  Middle layers: tau={tau_map[1]}")
        print(f"  Last layer: tau={tau_map[num_layers-1]}\n")
    
    ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
        tokenizer, use_grounding=args.use_groundness, use_utility=args.use_utility
    )

    def generate(tokenizer, prompt, evidences, max_new_tokens):
        return call_model_rerank_w_scores_batch_roe(
            tokenizer,
            prompt,
            evidences=evidences,
            model=model,
            max_new_tokens=max_new_tokens,
            rel_tokens=rel_tokens,
            ret_tokens=ret_tokens,
            grd_tokens=grd_tokens,
            ut_tokens=ut_tokens,
            threshold=args.threshold,
            use_seqscore=args.use_seqscore,
            w_rel=args.w_rel,
            w_sup=args.w_sup,
            w_use=args.w_use,
            mode=args.mode,
            closed=args.task in ["fever", "arc_c"],
            use_roe=args.use_roe,
            roe_k=args.roe_k,
            tau_map=tau_map,
            device=args.device
        )

    preds = []
    prompts = []
    golds = []
    metric_results = []
    scores = []
    all_results = []
    detailed_init_scores = []
    count = 0
    f1_list, precision_list, recall_list = [], [], []
    
    print(f"Starting evaluation on {len(input_data)} examples...")
    print(f"Output file: {args.output_file}\n")
    
    for i, row in tqdm(enumerate(input_data), total=len(input_data)):
        results = {}
        prompt = PROMPT_DICT["prompt_no_input"].format_map(row)
        _, evidences = process_data_evidences(row, top_n=args.ndocs)
        pred, results, do_retrieve, detailed_init_score = generate(
            tokenizer,
            prompt,
            evidences,
            max_new_tokens=args.max_new_tokens,
        )
        
        if type(pred) is str and len(pred) > 1 and (pred[0] == "#" or pred[0] == ":"):
            pred = pred[1:]
            
        prompts.append(prompt)
        preds.append(pred)
        all_results.append(results)
        detailed_init_scores.append(detailed_init_score)
        if do_retrieve is True:
            count += 1
        if "answers" not in row and "answer" in row:
            row["answers"] = (
                [row["answer"]] if type(row["answer"]) is str else row["answer"]
            )
        if args.metric == "accuracy":
            metric_result = accuracy(pred, row["output"])
        elif args.metric == "hotpotem":
            em = hotpot_exact_match_score(pred, row["answers"][0])
            f1, precision, recall = hotpot_f1_score(pred, row["answers"][0])
            metric_result = em
            f1_list.append(f1)
            precision_list.append(precision)
            recall_list.append(recall)

        elif args.metric == "match":
            if "SUPPORTS" in pred:
                pred = "true"
            elif "REFUTES" in pred:
                pred = "false"
            metric_result = match(pred, row["answers"])
        else:
            raise NotImplementedError

        metric_results.append(metric_result)
        if i % 10 == 0 and i > 0:
            print(f"\nProgress: {i}/{len(input_data)}")
            print("Current average: {}".format(np.mean(metric_results)))
            if args.metric == "hotpotem":
                print("Average em: {}, f1: {}, precision: {}, recall: {}".format(
                    np.mean(metric_results), np.mean(f1_list), np.mean(precision_list), np.mean(recall_list)))
            final_results = {
                "preds": preds,
                "prompts": prompts,
                "metric_results": metric_results,
                "all_results": all_results,
                "golds": golds,
                "metric": args.metric,
                "metric_mean": np.mean(metric_results),
                "scores": scores,
                
                "F1": f1_list,
                "EM": metric_results,
                "Precision": precision_list,
                "Recall": recall_list,
                
                # RoE metadata
                "use_roe": args.use_roe,
                "roe_k": args.roe_k if args.use_roe else None,
                "roe_tau": args.roe_tau if args.use_roe else None,
            }
            with open(args.output_file + "_tmp", "w") as outfile:
                json.dump(final_results, outfile)

    final_results = {
        "preds": preds,
        "prompts": prompts,
        "metric_results": metric_results,
        "all_results": all_results,
        "golds": golds,
        "metric": args.metric,
        "metric_mean": np.mean(metric_results),
        "scores": scores,
        
        "F1": f1_list,
        "EM": metric_results,
        "Precision": precision_list,
        "Recall": recall_list,

        "detailed_init_scores": detailed_init_scores,
        
        # RoE metadata
        "use_roe": args.use_roe,
        "roe_k": args.roe_k if args.use_roe else None,
        "roe_tau": args.roe_tau if args.use_roe else None,
    }
    with open(args.output_file, "w") as outfile:
        json.dump(final_results, outfile)

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print("Final result: {0}".format(np.mean(metric_results)))
    print("Retrieval Frequencies: {0}".format(count / len(input_data)))
    if args.metric == "hotpotem":
        print("Average em: {}, f1: {}, precision: {}, recall: {}".format(
                np.mean(metric_results), np.mean(f1_list), np.mean(precision_list), np.mean(recall_list)))
    
    if args.use_roe:
        print("\nRoE Configuration:")
        print(f"  K (samples): {args.roe_k}")
        print(f"  tau (temperature): {args.roe_tau}")
    
    print(f"\nResults saved to: {args.output_file}")
    print("="*80)


if __name__ == "__main__":
    main()
