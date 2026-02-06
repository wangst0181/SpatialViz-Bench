import os, re
import time
import random
import torch
import json
from argparse import ArgumentParser
from tqdm import tqdm
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from copy import deepcopy

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_jsonl(file_path):
    results = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            results.append(data)
    return results

OPTION_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
REGEX_FLAGS = re.IGNORECASE | re.DOTALL

PROMPT_LIBRARY = {
    "choice_reasoning_tagged": {
        "name": "choice_reasoning_tagged",
        "template": (
            "You should first provide a reasoning process, then provide a single option "
            "(A, B, C or D) as the final answer. "
            "The reasoning process and the answer are enclosed within <think></think> and "
            "<answer></answer> tags respectively."
        ),
        "answer_type": "option",
        "question_field": "Question",
        "requires_choices": True,
    },
    "choice_reasoning_boxed": {
        "name": "choice_reasoning_boxed",
        "template": (
            "Answer with the option’s letter from the given choices and put the letter "
            "in one '\\boxed{}'. Please solve the problem step by step."
        ),
        "answer_type": "option",
        "question_field": "Question",
        "requires_choices": True,
        "custom_answer_patterns": [
            r"\\{1,2}boxed\{(?:\\text\{)?(?P<value>[A-D])"
        ],
    },
    "direct_reasoning_tagged": {
        "name": "direct_reasoning_tagged",
        "template": (
            "You should first provide a reasoning process, then provide the number as the "
            "final answer. The reasoning process and the answer are enclosed within "
            "<think></think> and <answer></answer> tags respectively."
        ),
        "answer_type": "number",
        "question_field": "Question_direct",
        "requires_choices": False,
    },
    "choice_answer_only": {
        "name": "choice_answer_only",
        "template": (
            "Answer with a single option letter (A, B, C, or D), enclosed within the "
            "<answer></answer> tag. For example: <answer>A</answer>. Ensure that your "
            "output contains only the final answer, without any intermediate reasoning "
            "or additional content."
        ),
        "answer_type": "option",
        "question_field": "Question",
        "requires_choices": True,
    },
}


def _compile_patterns(patterns):
    compiled = []
    for pattern in patterns:
        if isinstance(pattern, re.Pattern):
            compiled.append(pattern)
        else:
            compiled.append(re.compile(pattern, REGEX_FLAGS))
    return compiled


OPTION_PATTERNS = _compile_patterns([
    r"<answer>\s*(?P<value>.*?)\s*</answer>",
    r"<answer>\s*option\s+(?P<value>[A-D])(?=answer>)",
    r"</answer>\s*(?P<value>[A-D])\b",
    r"(?:final|correct\s+)?answer\s*(?:is|:)\s*(?:option\s*)?(?P<value>[A-D])\b",
    r"correct\s+path\s*(?:is|:)?\s*(?:option\s*)?(?P<value>[A-D])\b",
    r"correct\s+choice\s*(?:is|:)?\s*(?:option\s*)?(?P<value>[A-D])\b",
    r"option\s+(?P<value>[A-D])\b",
    r"choose\s+(?P<value>[A-D])\b",
    r"\\{1,2}boxed\{(?:\\text\{)?(?P<value>[A-D])",
])


LEGACY_FINAL_ANSWER_PATTERNS = [
    "<answer>",
    "Answer:",
    "Final answer",
    "final answer",
    "Final Answer",
    "the answer is",
    "The answer is",
    "correct answer",
    "Correct answer",
    "Correct Answer",
    "correct path",
]


NUMBER_PATTERNS = _compile_patterns([
    r"<answer>\s*(?P<value>-?\d+(?:\.\d+)?)\s*</answer>",
    r"(?:final|correct\s+)?answer\s*(?:is|:)\s*(?P<value>(?<![A-Za-z])-?\d+(?:\.\d+)?)\b",
    r"result\s*(?:is|:)\s*(?P<value>(?<![A-Za-z])-?\d+(?:\.\d+)?)\b",
    r"\\{1,2}boxed\{(?:\\text\{)?(?P<value>-?\d+(?:\.\d+)?)",
])


def prepare_prompt_spec(spec_key, fallback_key, *, default_answer_type, question_field, requires_choices):
    """
    Returns a normalized prompt specification.
    spec_key can be a key in PROMPT_LIBRARY or a raw prompt string.
    """
    if spec_key and spec_key in PROMPT_LIBRARY:
        spec = deepcopy(PROMPT_LIBRARY[spec_key])
    elif spec_key:
        spec = {
            "name": "custom_prompt",
            "template": spec_key,
            "answer_type": default_answer_type,
            "question_field": question_field,
            "requires_choices": requires_choices,
        }
    else:
        spec = deepcopy(PROMPT_LIBRARY[fallback_key])

    spec.setdefault("name", fallback_key)
    spec.setdefault("answer_type", default_answer_type)
    spec.setdefault("question_field", question_field)
    spec.setdefault("requires_choices", requires_choices)
    spec.setdefault("custom_answer_patterns", [])
    spec["custom_answer_patterns"] = _compile_patterns(spec["custom_answer_patterns"])
    return spec


def format_input_text(item, prompt_spec, ops):
    question_field = prompt_spec.get("question_field", "Question")
    question = item.get(question_field)
    if not question:
        raise ValueError(f"Question field '{question_field}' missing for item {get_data_id(item)}")

    text = prompt_spec["template"].strip() + "\nQuestion: " + question.strip()
    if prompt_spec.get("requires_choices", False):
        choices = item.get("Choices")
        if not choices:
            raise ValueError("Choices missing while prompt requires them.")
        choice_lines = []
        for idx, choice in enumerate(choices):
            label = ops[idx] if idx < len(ops) else OPTION_LABELS[idx]
            choice_lines.append(f"{label}. {choice}")
        text += "\nChoices: " + "\n".join(choice_lines)
    return text


def extract_thinking(response):
    if not response:
        return None
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    return think_match.group(1).strip() if think_match else None


def _legacy_extract_option_answer(response):
    """
    Replicates supplementary/code/evaluation/evaluate_old.py matching logic.
    Prioritizes a simple pattern split before newer regex heuristics.
    """
    if not response:
        return None

    text = response.strip()
    if not text:
        return None

    matches = []
    if len(text) == 1:
        matches = re.findall(r"[A-D]", text)
    else:
        for marker in LEGACY_FINAL_ANSWER_PATTERNS:
            if marker in text:
                matches = re.findall(r"\b([A-D])\b", text.split(marker)[-1].strip().split(".")[0])
                if matches:
                    break

    if not matches:
        return None

    unique = {m for m in matches}
    if len(unique) == 1:
        return unique.pop()
    return None


def _cleanup_answer(value, answer_type):
    """
    Clean extracted raw value and enforce the unique-answer rule.
    """
    if value is None:
        return None

    value = value.strip()

    if answer_type == "option":
        matches = re.findall(r"\b([A-D])\b", value)
        if len(matches) == 1:
            return matches[0]
        return None

    if answer_type == "number":
        matches = re.findall(r"(?<![A-Za-z])(-?\d+(?:\.\d+)?)", value)
        if len(matches) == 1:
            return matches[0].strip()
        return None

    return None


def fallback_option_answer(response):
    tail = "\n".join(response.strip().splitlines()[-3:])
    matches = re.findall(r"\b([A-D])\b", tail, re.IGNORECASE)
    return matches[-1].upper() if matches else None


def fallback_number_answer(response):
    tail = "\n".join(response.strip().splitlines()[-3:])
    matches = re.findall(r"-?\d+(?:\.\d+)?", tail)
    return matches[-1] if matches else None


def _find_answers_for_patterns(patterns_list, response, answer_type):
    """
    Helper to find answers from a pattern list.
    - For "option": collect all matches for a pattern, enforce uniqueness per pattern.
    - For "number": return the first valid match.
    """
    found_answers = set()

    if answer_type == "number":
        for pattern in patterns_list:
            match = pattern.search(response)
            if match:
                try:
                    raw_value = match.group("value")
                except (IndexError, KeyError):
                    raw_value = match.group(1) if match.groups() else match.group(0)
                cleaned = _cleanup_answer(raw_value, answer_type)
                if cleaned:
                    found_answers.add(cleaned)
                    return found_answers
        return found_answers

    for pattern in patterns_list:
        pattern_specific_answers = set()
        matches = pattern.finditer(response)
        for match in matches:
            try:
                raw_value = match.group("value")
            except (IndexError, KeyError):
                raw_value = match.group(1) if match.groups() else match.group(0)
            cleaned = _cleanup_answer(raw_value, answer_type)
            if cleaned:
                pattern_specific_answers.add(cleaned)

        if pattern_specific_answers:
            found_answers.update(pattern_specific_answers)
            break

    return found_answers


def extract_answer(response, answer_type, extra_patterns=None, allow_tail_fallback=True):
    """
    Extract answer with priority: legacy -> extra patterns -> standard patterns -> tail fallback.
    """
    if not response:
        return None

    legacy_answer = _legacy_extract_option_answer(response)
    if legacy_answer:
        return legacy_answer

    if extra_patterns:
        extra_answers = _find_answers_for_patterns(extra_patterns, response, answer_type)
        if len(extra_answers) == 1:
            return extra_answers.pop()
        if len(extra_answers) > 1:
            return None

    std_patterns = OPTION_PATTERNS if answer_type == "option" else NUMBER_PATTERNS
    std_answers = _find_answers_for_patterns(std_patterns, response, answer_type)

    if len(std_answers) == 1:
        return std_answers.pop()
    if len(std_answers) > 1:
        return None

    if not allow_tail_fallback:
        return None

    if answer_type == "option":
        return fallback_option_answer(response)
    return fallback_number_answer(response)


def resolve_prompt_for_item(item, prompt_specs, prefer_direct):
    order = ["direct", "choice"] if prefer_direct else ["choice", "direct"]
    for key in order:
        spec = prompt_specs.get(key)
        if not spec:
            continue
        try:
            text = format_input_text(item, spec, OPTION_LABELS)
            return spec, text
        except ValueError:
            continue
    raise ValueError(f"No valid prompt configuration found for item {get_data_id(item)}")


def has_valid_answer(record, spec):
    if not record:
        return False
    if record.get("Response") == "Failed!!!":
        return False
    final_answer = record.get("FinalAnswer")
    if not final_answer:
        return False
    if spec["answer_type"] == "option":
        return bool(re.fullmatch(r"[A-D]", final_answer.strip().upper()))
    return bool(re.search(r"-?\d+(?:\.\d+)?", final_answer))


def build_image_path(item, benchmark_root):
    if not benchmark_root:
        return None
    category = item.get("Category")
    task = item.get("Task")
    level = item.get("Level")
    image_id = item.get("Image_id")
    if not all([category, task, level, image_id]):
        return None
    return os.path.join(
        benchmark_root,
        category,
        task,
        level,
        f"{image_id}.png",
    )


def sample_data_per_level(data, sample_size, seed=None):
    if not sample_size or sample_size <= 0:
        return data
    rng = random.Random(seed) if seed is not None else None
    buckets = {}
    order = []
    for item in data:
        key = (item.get("Category"), item.get("Task"), item.get("Level"))
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(item)

    sampled = []
    for key in order:
        items = buckets[key]
        if len(items) <= sample_size:
            sampled.extend(items)
        else:
            chooser = rng.sample if rng else random.sample
            sampled.extend(chooser(items, sample_size))

    print(f"[Sampling] Sampled {len(sampled)} / {len(data)} items ({sample_size} per difficulty).")
    return sampled


def _prepare_prompt_specs(choice_prompt, direct_prompt):
    return {
        "choice": prepare_prompt_spec(
            choice_prompt,
            "choice_reasoning_tagged",
            default_answer_type="option",
            question_field="Question",
            requires_choices=True,
        ),
        "direct": prepare_prompt_spec(
            direct_prompt,
            "direct_reasoning_tagged",
            default_answer_type="number",
            question_field="Question_direct",
            requires_choices=False,
        ),
    }

def get_response(model, processor, image_path, text, *, include_image=True):
    if not include_image or not image_path:
        raise ValueError("Image input is disabled, but Kimi-VL requires images.")
    image = Image.open(image_path)
    messages = [
        {"role": "user", 
         "content": [{"type": "image", "image": image_path}, 
                     {"type": "text", "text": text}]}
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    inputs = processor(images=image, text=text, return_tensors="pt", padding=True, truncation=True).to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response


def get_data_id(item):
    category = item["Category"]
    task = item["Task"]
    level = item["Level"]
    image_id = item['Image_id']
    return f"{category}-{task}-{level}-{image_id}"    

def run_model_for_item(
    item,
    text,
    spec,
    image_path,
    *,
    enable_image_input,
    max_retries,
    request_interval,
    allow_tail_fallback,
    model,
    processor,
):
    attempts = 0
    last_error = None
    include_image = bool(enable_image_input and image_path)
    while attempts < max_retries:
        try:
            response = get_response(
                model,
                processor,
                image_path,
                text,
                include_image=include_image,
            )
            result = item.copy()
            result.update({
                "InputText": text,
                "PromptName": spec["name"],
                "AnswerType": spec["answer_type"],
            })

            thinking = extract_thinking(response)
            has_reasoning_prompt = "reasoning" in spec["name"].lower()
            if thinking:
                result["ThinkingProcess"] = thinking

            final_answer = extract_answer(
                response,
                spec["answer_type"],
                spec.get("custom_answer_patterns"),
                allow_tail_fallback=allow_tail_fallback,
            )
            if final_answer:
                result["FinalAnswer"] = final_answer
            else:
                result["Response"] = response

            if has_reasoning_prompt and not thinking:
                result["Response"] = response

            time.sleep(request_interval)
            return result
        except Exception as exc:
            last_error = str(exc)
            attempts += 1
            time.sleep(request_interval)

    failed = item.copy()
    failed.update({
        "InputText": text,
        "PromptName": spec["name"],
        "AnswerType": spec["answer_type"],
        "Response": "Failed!!!",
        "ErrorMessage": last_error,
    })
    return failed


def evaluate(args, model_path, model, processor):
    data = load_json(f"{args.benchmark_test_path}/{args.data_file}")
    if args.enable_sampling:
        data = sample_data_per_level(
            data,
            args.sample_per_level,
            seed=args.sample_seed,
        )

    prompt_specs = _prepare_prompt_specs(args.choice_prompt, args.direct_prompt)
    use_images = not args.text_only
    save_path = f"{args.results_dir}/results_{model_path.split(os.sep)[-1]}.jsonl"
    print(save_path)
    result_file = open(save_path, "w", encoding='utf-8')
    
    for item in tqdm(data, desc=f"Evaluating {model_path.split(os.sep)[-1]}"):
        try:
            spec, text = resolve_prompt_for_item(item, prompt_specs, args.use_direct_answer)
        except ValueError as exc:
            failure = item.copy()
            failure.update({
                "InputText": "",
                "PromptName": "unresolved",
                "AnswerType": None,
                "Response": "Failed!!!",
                "ErrorMessage": str(exc),
            })
            result_file.write(json.dumps(failure, ensure_ascii=False) + "\n")
            continue

        image_path = build_image_path(item, args.benchmark_test_path) if use_images else None
        result = run_model_for_item(
            item,
            text,
            spec,
            image_path,
            enable_image_input=use_images,
            max_retries=args.max_retries,
            request_interval=args.request_interval,
            allow_tail_fallback=args.enable_tail_fallback,
            model=model,
            processor=processor,
        )
        result_file.write(json.dumps(result, ensure_ascii=False) + "\n")
        result_file.flush()

def modify(args):
    data = load_json(f"{args.benchmark_test_path}/{args.data_file}")
    if args.enable_sampling:
        data = sample_data_per_level(
            data,
            args.sample_per_level,
            seed=args.sample_seed,
        )

    prompt_specs = _prepare_prompt_specs(args.choice_prompt, args.direct_prompt)
    use_images = not args.text_only

    for model_path in args.model_paths:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        save_name = model_path.split('/')[-1]
        result_file_path = f"{args.results_dir}/results_{save_name}.jsonl"
        if not os.path.exists(result_file_path):
            print(f"No base file found at {result_file_path}. Running fresh evaluation first.")
            evaluate(args, model_path, model, processor)
            return True

        existing_results = load_jsonl(result_file_path)
        result_map = {get_data_id(record): record for record in existing_results}

        save_path = f"{args.results_dir}/results_{save_name}_modify.jsonl"
        print(save_path)
        new_result_file = open(save_path, "w", encoding='utf-8')

        for item in tqdm(data, desc=f"Modify {save_name}"):
            try:
                spec, text = resolve_prompt_for_item(item, prompt_specs, args.use_direct_answer)
            except ValueError as exc:
                failure = item.copy()
                failure.update({
                    "InputText": "",
                    "PromptName": "unresolved",
                    "AnswerType": None,
                    "Response": "Failed!!!",
                    "ErrorMessage": str(exc),
                })
                new_result_file.write(json.dumps(failure, ensure_ascii=False) + "\n")
                continue

            data_id = get_data_id(item)
            cached = result_map.get(data_id)
            if cached and has_valid_answer(cached, spec):
                new_result_file.write(json.dumps(cached, ensure_ascii=False) + "\n")
                continue

            image_path = build_image_path(item, args.benchmark_test_path) if use_images else None
            result = run_model_for_item(
                item,
                text,
                spec,
                image_path,
                enable_image_input=use_images,
                max_retries=args.max_retries,
                request_interval=args.request_interval,
                allow_tail_fallback=args.enable_tail_fallback,
                model=model,
                processor=processor,
            )
            new_result_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            new_result_file.flush()

        del model
        del processor
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_paths", type=str, nargs='+', required=True,
                        help="List of model names or local paths to the models.")
    parser.add_argument('--benchmark_test_path', type=str, required=True,
                        help="Root directory of the benchmark data.")
    parser.add_argument('--results_dir', type=str, required=True,
                        help="Directory to save results.")
    parser.add_argument('--data_file', type=str, required=True,
                        help="File name inside benchmark_test_path to load as dataset.")
    parser.add_argument('--run_mode', choices=["evaluate", "modify"], default="modify",
                        help="Choose whether to run a fresh evaluation or fix previous results.")
    parser.add_argument('--use_direct_answer', action='store_true',
                        help="Prefer direct-answer prompts when datasets provide Question_direct.")
    parser.add_argument('--choice_prompt', type=str, default="choice_reasoning_tagged",
                        help="Prompt key or raw text for multiple-choice questions.")
    parser.add_argument('--direct_prompt', type=str, default="direct_reasoning_tagged",
                        help="Prompt key or raw text for direct-answer questions.")
    parser.add_argument('--max_retries', type=int, default=3,
                        help="Maximum times to retry a failed model call per instance.")
    parser.add_argument('--request_interval', type=float, default=1.0,
                        help="Seconds to sleep between model attempts.")
    parser.add_argument('--text_only', action='store_true',
                        help="If set, skip attaching images and evaluate as text-only prompts.")
    parser.add_argument('--enable_sampling', action='store_true',
                        help="Enable random sampling per (Category, Task, Level) before evaluation.")
    parser.add_argument('--sample_per_level', type=int, default=10,
                        help="Number of samples per (Category, Task, Level) group when sampling is enabled.")
    parser.add_argument('--sample_seed', type=int, default=42,
                        help="Random seed used for sampling when --enable_sampling is set.")
    parser.add_argument('--enable_tail_fallback', action='store_true',
                        help="Enable low-accuracy tail fallback extraction when tags are missing.")
    args = parser.parse_args()
    print("Parsed args:", args)
    if args.run_mode == "modify":
        modify(args)
    else:
        for model_path in args.model_paths:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2"
            )
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            evaluate(args, model_path, model, processor)
            del model
            del processor
            torch.cuda.empty_cache()
