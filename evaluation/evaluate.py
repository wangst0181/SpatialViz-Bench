import json
import os
import re
import base64
import math
import time
import random
from tqdm import tqdm
from openai import OpenAI
from copy import deepcopy
from argparse import ArgumentParser

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

def dump_json(save_path, data):
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
        
def dump_jsonl(save_path, data):
    with open(save_path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_client(openai_api_key, openai_api_base):
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    return client

def get_response(key, url, model_name_or_path, image_path, text, logprobs=False, top_logprobs=6, include_image=True):
    client = create_client(key, url)
    user_content = []
    use_image = bool(include_image and image_path)
    if use_image:
        if not os.path.exists(image_path):
            print(f"[Warning] Image path not found, fallback to text only: {image_path}")
            use_image = False
        else:
            encoded_image = encode_image(image_path)
            image_type = image_path.split('.')[-1]
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_type};base64,{encoded_image}",
                    },
                }
            )
    user_content.append({"type": "text", "text": text})
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": user_content,
        },
    ]
    if 'omni' not in model_name_or_path:
        if logprobs:
            chat_response = client.chat.completions.create(
                model=model_name_or_path, 
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                messages=messages,
            )
        else:
            if "qwen3" not in model_name_or_path:
                chat_response = client.chat.completions.create(
                    model=model_name_or_path, 
                    messages=messages,
                )
            else:
                chat_response = client.chat.completions.create(
                    model=model_name_or_path, 
                    messages=messages,
                    extra_body={"enable_thinking":True},
                )
                

        if chat_response.choices is None:
            return chat_response, None
        elif logprobs and hasattr(chat_response.choices[0], 'logprobs') and chat_response.choices[0].logprobs is not None:
            return chat_response.choices[0].message.content, chat_response.choices[0].logprobs.content
        elif hasattr(chat_response.choices[0].message, 'reasoning_content') and chat_response.choices[0].message.reasoning_content is not None:
            return f"<think>{chat_response.choices[0].message.reasoning_content}</think>" + chat_response.choices[0].message.content, None
        else:
            return chat_response.choices[0].message.content, None
    else:
        chat_response = client.chat.completions.create(
            model=model_name_or_path,
            messages=messages,
            modalities=["text"],
            stream=True,
            stream_options={"include_usage": True}
        )
        response = ""
        for chunk in chat_response:
            if chunk.choices:
                if hasattr(chunk.choices[0].delta, 'content')  and chunk.choices[0].delta.content is not None:
                    response += chunk.choices[0].delta.content
                if hasattr(chunk.choices[0].delta, 'reasoning_content')  and chunk.choices[0].delta.reasoning_content is not None:
                    response += chunk.choices[0].delta.reasoning_content
        return response, None


def _require_api_key(key_name, key_value):
    if not key_value:
        raise ValueError(f"Missing API key for {key_name}. Provide it via command-line arguments.")
    return key_value


def get_response_closed_models(
    model_name,
    image_path,
    text,
    logprobs=False,
    top_logprobs=6,
    include_image=True,
    *,
    qwen_api_key=None,
    doubao_api_key=None,
    openai_api_key=None,
    gemini_api_key=None,
    openrouter_api_key=None,
):
    if model_name in ["qvq-72b-preview", "qwen2.5-vl-72b-instruct", "qwen2.5-vl-32b-instruct", "qwen2.5-vl-7b-instruct", "qwen2.5-vl-3b-instruct", "qwen-vl-max-latest", "qwen2.5-omni-7b", "qwen-omni-turbo", "qwen3-32b", "qwen3-max", "qwen2.5-72b-instruct"]:
        return get_response(
            key=_require_api_key("Qwen", qwen_api_key),
            url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model_name_or_path=model_name,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            image_path=image_path,
            text=text,
            include_image=include_image,
        )
    elif model_name in ["doubao-1-5-vision-pro-32k-250115"]:
        return get_response(
            key=_require_api_key("Doubao", doubao_api_key),
            url="https://ark.cn-beijing.volces.com/api/v3",
            model_name_or_path=model_name,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            image_path=image_path,
            text=text,
            include_image=include_image,
        )
    elif model_name in ["gemini-2.5-pro-preview-03-25", "gemini-1.5-pro", "gemini-2.5-flash-preview-04-17"]:
        return get_response(
            key=_require_api_key("Gemini", gemini_api_key),
            url="https://generativelanguage.googleapis.com/v1beta/openai/",
            model_name_or_path=model_name,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            image_path=image_path,
            text=text,
            include_image=include_image,
        )
    elif model_name in ["moonshotai/moonlight-16b-a3b-instruct:free", "moonshotai/kimi-vl-a3b-thinking:free", "meta-llama/llama-4-maverick", "meta-llama/llama-4-scout", "anthropic/claude-3.5-sonnet", "openai/o3", "openai/o4-mini", "openai/o1", "openai/gpt-4o-2024-08-06", "openai/gpt-4o", "google/gemini-2.5-pro", "qwen/qwen3-32b"]:
        return get_response(
            key=_require_api_key("OpenRouter", openrouter_api_key),
            url="https://openrouter.ai/api/v1",
            model_name_or_path=model_name,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            image_path=image_path,
            text=text,
            include_image=include_image,
        )
    elif model_name in ["gpt-4o", "o1", "o4-mini", "o3"]:
        return get_response(
            key=_require_api_key("OpenAI", openai_api_key),
            url="https://api.openai.com/v1",
            model_name_or_path=model_name,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            image_path=image_path,
            text=text,
            include_image=include_image,
        )


def get_prob_for_choice(log_probs):
    ops = {"A":-math.inf, "B":-math.inf, "C":-math.inf, "D":-math.inf}
    flag = 0
    for item in log_probs:
        if flag == 0 and "answer" in item.top_logprobs[0].token:
            flag = 1
        if flag == 1 and item.top_logprobs[0].token in [">", ">A", ">B", ">C", ">D", "A", "B", "C", "D", "A.", "B.", "C.", "D."]:
            flag = 2
        if flag == 2:
            for top in item.top_logprobs:
                token = top.token
                prob = top.logprob
                op = re.findall(r'[A-D]', token)
                if len(op) != 0:
                    if prob > ops[op[0]]:
                        ops[op[0]] = prob
    exp_sum = sum(math.exp(x) for x in ops.values())
    if any(v != -math.inf for v in ops.values()):
        softmax_probs = {
            opt: math.exp(logprob) / exp_sum for opt, logprob in ops.items()
        }
    else:
        ops = None
        softmax_probs = None
    return ops, softmax_probs


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
    # 1. Match standard <answer>...</answer>
    r"<answer>\s*(?P<value>.*?)\s*</answer>",
    r"<answer>\s*option\s+(?P<value>[A-D])(?=answer>)",

    # 2. Match malformed </answer>B</think> style (Index 989)
    r"</answer>\s*(?P<value>[A-D])\b",
    
    # 3. Match formats like "Answer: D</think>" (Index 4; \\b is key)
    r"(?:final|correct\s+)?answer\s*(?:is|:)\s*(?:option\s*)?(?P<value>[A-D])\b",

    # 3.5 Match answers describing "correct path"
    r"correct\s+path\s*(?:is|:)?\s*(?:option\s*)?(?P<value>[A-D])\b",
    r"correct\s+choice\s*(?:is|:)?\s*(?:option\s*)?(?P<value>[A-D])\b",
    
    # 4. Other standard matches
    r"option\s+(?P<value>[A-D])\b",
    r"choose\s+(?P<value>[A-D])\b",
    r"\\{1,2}boxed\{(?:\\text\{)?(?P<value>[A-D])",
])
# --- (end of update) ---

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
    (Updated to satisfy requests 1 and 2)
    Clean the raw extracted string 'value' and strictly enforce the "unique answer" rule.

    1. (Request 2) Find *all* matches (A-D or number) within the extracted 'value',
       rather than matching the entire 'value' string.
    2. (Request 1) Return the match only when *exactly one* is found.
       If 0 or 2+ matches are found, return None (invalid extraction).
    """
    if value is None:
        return None
    
    value = value.strip()
    
    if answer_type == "option":
        # (Request 2) Find all standalone A-D letters (case-insensitive; use \\b for boundaries)
        # Example: "A." or "is A" matches, but "Apple" does not match "A"
        matches = re.findall(r"\b([A-D])\b", value)
        
        # (Request 1) Enforce uniqueness strictly
        if len(matches) == 1:
            return matches[0] # Return the unique answer
        else:
            # 0 or 2+ matches (e.g., "A and B") are invalid
            return None 

    elif answer_type == "number":
        # (Request 2) Find all numbers (allow negatives and decimals)
        # Use logic similar to NUMBER_PATTERNS to avoid matching letters+digits
        matches = re.findall(r"(?<![A-Za-z])(-?\d+(?:\.\d+)?)", value)
        
        # (Request 1) Enforce uniqueness strictly
        if len(matches) == 1:
            return matches[0].strip() # Return the unique number
        else:
            # 0 or 2+ numbers are invalid
            return None
            
    # Default
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
    Helper: find valid answers from a list of patterns.
    - For "option", collect matches and return a set of unique answers.
    - For "number", return the first match (as a set).
    """
    found_answers = set()
    
    # For "number", only the first valid match is needed
    if answer_type == "number":
        for pattern in patterns_list:
            match = pattern.search(response)
            if match:
                try:
                    raw_value = match.group("value")
                except (IndexError, KeyError):
                    raw_value = match.group(1) if match.groups() else match.group(0)
                
                # Call the updated _cleanup_answer
                cleaned = _cleanup_answer(raw_value, answer_type)
                if cleaned:
                    found_answers.add(cleaned)
                    return found_answers # Return set with the first number
        return found_answers # Return empty set

    for pattern in patterns_list:
        # 1. Create a temporary set for this pattern
        pattern_specific_answers = set()
        
        # 2. Find all matches for this pattern
        matches = pattern.finditer(response) 
        for match in matches:
            try:
                raw_value = match.group("value")
            except (IndexError, KeyError):
                raw_value = match.group(1) if match.groups() else match.group(0)
            
            # 3. Clean and collect valid answers for this pattern
            cleaned = _cleanup_answer(raw_value, answer_type)
            if cleaned:
                pattern_specific_answers.add(cleaned)
        
        # 4. (Key) Stop once this pattern finds any answer
        if pattern_specific_answers:
            found_answers.update(pattern_specific_answers)
            break       
        
    return found_answers


def extract_answer(response, answer_type, extra_patterns=None, allow_tail_fallback=True):
    """
    (Rewritten)
    Extract answers by priority (Request 1) and check multiple answers (Request 2).
    """
    if not response:
        return None

    # 0. Apply legacy logic first to match evaluate_old.py behavior
    legacy_answer = _legacy_extract_option_answer(response)

    if legacy_answer:
        return legacy_answer

    # 1. (Request 1) Prefer extra_patterns
    if extra_patterns:
        extra_answers = _find_answers_for_patterns(extra_patterns, response, answer_type)

        if len(extra_answers) == 1:
            return extra_answers.pop() # Unique preferred answer
        if len(extra_answers) > 1:
            return None 
    
    # 2. If extra_patterns is empty or yields nothing, use standard patterns
    std_patterns = OPTION_PATTERNS if answer_type == "option" else NUMBER_PATTERNS
    std_answers = _find_answers_for_patterns(std_patterns, response, answer_type)
    
    # 3. (Request 2) Resolve standard pattern results
    if len(std_answers) == 1:
        return std_answers.pop() # Unique standard answer
    if len(std_answers) > 1:
        return None 

    # If no answer yet, apply tail fallback (disabled by default)
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


def run_model_for_item(
    model,
    item,
    text,
    spec,
    image_path,
    *,
    enable_image_input,
    logprobs,
    top_logprobs,
    max_retries,
    request_interval,
    allow_tail_fallback,
    qwen_api_key=None,
    doubao_api_key=None,
    openai_api_key=None,
    gemini_api_key=None,
    openrouter_api_key=None,
):
    attempts = 0
    last_error = None
    include_image = bool(enable_image_input and image_path)
    while attempts < max_retries:
        try:
            response, log_probs = get_response_closed_models(
                model,
                image_path,
                text,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                include_image=include_image,
                qwen_api_key=qwen_api_key,
                doubao_api_key=doubao_api_key,
                openai_api_key=openai_api_key,
                gemini_api_key=gemini_api_key,
                openrouter_api_key=openrouter_api_key,
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

            # Call the rewritten extract_answer with the updated _cleanup_answer logic
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

            if logprobs and log_probs is not None:
                log_probs_pre, softmax_probs = get_prob_for_choice(log_probs)
                if log_probs_pre is not None and softmax_probs is not None:
                    result.update({"log_probs": log_probs_pre, "softmax_probs": softmax_probs})
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
    

def get_data_id(item):
    category = item.get("Category", "UnknownCategory")
    task = item.get("Task", "UnknownTask")
    level = item.get("Level", "UnknownLevel")
    image_id = item.get("Image_id", "NoImage")
    return f"{category}-{task}-{level}-{image_id}"


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


def evaluate(
    model_list,
    data,
    benchmark_test_path,
    save_dir,
    *,
    use_images=True,
    logprobs=False,
    top_logprobs=6,
    use_direct_answer=False,
    choice_prompt=None,
    direct_prompt=None,
    max_retries=3,
    request_interval=1.0,
    tail_fallback=True, 
    qwen_api_key=None,
    doubao_api_key=None,
    openai_api_key=None,
    gemini_api_key=None,
    openrouter_api_key=None,
):
    """
    Evaluate models on the provided dataset.
    """
    os.makedirs(save_dir, exist_ok=True)
    prompt_specs = _prepare_prompt_specs(choice_prompt, direct_prompt)

    for model in model_list:
        print(f"[Evaluate] model={model}")
        save_name = model.split('/')[-1]
        save_path = f"{save_dir}/results_{save_name}.jsonl"

        if os.path.exists(save_path):
            print(f"Results already exist at {save_path}. Run modify() if you want to resume.")
            continue

        with open(save_path, "w", encoding="utf-8") as result_file:
            for item in tqdm(data, desc=f"Evaluating {save_name}"):
                try:
                    spec, text = resolve_prompt_for_item(item, prompt_specs, use_direct_answer)
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

                image_path = build_image_path(item, benchmark_test_path) if use_images else None

                result = run_model_for_item(
                    model,
                    item,
                    text,
                    spec,
                    image_path,
                    enable_image_input=use_images,
                    logprobs=logprobs,
                    top_logprobs=top_logprobs,
                    max_retries=max_retries,
                    request_interval=request_interval,
                    allow_tail_fallback=tail_fallback, 
                    qwen_api_key=qwen_api_key,
                    doubao_api_key=doubao_api_key,
                    openai_api_key=openai_api_key,
                    gemini_api_key=gemini_api_key,
                    openrouter_api_key=openrouter_api_key,
                )
                result_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                result_file.flush()


def modify(
    model_list,
    data,
    benchmark_test_path,
    save_dir,
    *,
    use_images=True,
    logprobs=False,
    top_logprobs=6,
    use_direct_answer=False,
    choice_prompt=None,
    direct_prompt=None,
    max_retries=3,
    request_interval=1.0,
    tail_fallback=True, 
    qwen_api_key=None,
    doubao_api_key=None,
    openai_api_key=None,
    gemini_api_key=None,
    openrouter_api_key=None,
):
    """
    Re-run failed or missing samples for existing result files.
    Copies successful entries directly to the new file.
    """
    os.makedirs(save_dir, exist_ok=True)
    prompt_specs = _prepare_prompt_specs(choice_prompt, direct_prompt)

    for model in model_list:
        print(f"[Modify] model={model}")
        save_name = model.split('/')[-1]

        result_file_path = f"{save_dir}/results_{save_name}.jsonl"
        if not os.path.exists(result_file_path):
            print(f"No base file found at {result_file_path}. Running fresh evaluation first.")
            evaluate(
                [model],
                data,
                benchmark_test_path,
                save_dir,
                use_images=use_images,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                use_direct_answer=use_direct_answer,
                choice_prompt=choice_prompt,
                direct_prompt=direct_prompt,
                max_retries=max_retries,
                request_interval=request_interval,
                tail_fallback=tail_fallback, 
                qwen_api_key=qwen_api_key,
                doubao_api_key=doubao_api_key,
                openai_api_key=openai_api_key,
                gemini_api_key=gemini_api_key,
                openrouter_api_key=openrouter_api_key,
            )
            continue

        existing_results = load_jsonl(result_file_path)
        result_map = {get_data_id(record): record for record in existing_results}

        save_path = f"{save_dir}/results_{save_name}_modify.jsonl"
        with open(save_path, "w", encoding="utf-8") as new_file:
            for item in tqdm(data, desc=f"Modify {save_name}"):
                try:
                    spec, text = resolve_prompt_for_item(item, prompt_specs, use_direct_answer)
                except ValueError as exc:
                    failure = item.copy()
                    failure.update({
                        "InputText": "",
                        "PromptName": "unresolved",
                        "AnswerType": None,
                        "Response": "Failed!!!",
                        "ErrorMessage": str(exc),
                    })
                    new_file.write(json.dumps(failure, ensure_ascii=False) + "\n")
                    continue

                data_id = get_data_id(item)
                cached = result_map.get(data_id)
                if cached and has_valid_answer(cached, spec):
                    new_file.write(json.dumps(cached, ensure_ascii=False) + "\n")
                    continue

                image_path = build_image_path(item, benchmark_test_path) if use_images else None

                result = run_model_for_item(
                    model,
                    item,
                    text,
                    spec,
                    image_path,
                    enable_image_input=use_images,
                    logprobs=logprobs,
                    top_logprobs=top_logprobs,
                    max_retries=max_retries,
                    request_interval=request_interval,
                    allow_tail_fallback=tail_fallback, 
                    qwen_api_key=qwen_api_key,
                    doubao_api_key=doubao_api_key,
                    openai_api_key=openai_api_key,
                    gemini_api_key=gemini_api_key,
                    openrouter_api_key=openrouter_api_key,
                )
                new_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                new_file.flush()

def random_answer(benchmark_test_path, save_dir):
    data = load_json(f"{benchmark_test_path}/data.json")
    save_path = f"{save_dir}/results_random.jsonl"
    print(save_path)
    new_result_file = open(save_path, "w", encoding='utf-8')
    for item in tqdm(data):
        result = item.copy()
        answer = random.choice(["A", "B", "C", "D"])
        result.update({"FinalAnswer": answer})
        new_result_file.write(json.dumps(result, ensure_ascii=False) + "\n")
        new_result_file.flush()


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


def _attach_accuracy_stats(stat_entry):
    total = stat_entry.get("total_num", 0)
    if total <= 0:
        stat_entry.update({
            "acc": "0.00",
            "se": "0.00",
            "ci_95": "0.00 ± 0.00",
            "ci_95_range": [0.0, 0.0],
        })
        return

    correct = stat_entry.get("correct_num", 0)
    proportion = correct / total
    se = math.sqrt(proportion * (1 - proportion) / total)
    ci_half = 1.96 * se
    lower = max(0.0, proportion - ci_half)
    upper = min(1.0, proportion + ci_half)

    stat_entry["acc"] = "{:.2f}".format(proportion * 100)
    stat_entry["se"] = "{:.2f}".format(se * 100)
    stat_entry["ci_95"] = "{:.2f} ± {:.2f}".format(proportion * 100, ci_half * 100)
    stat_entry["ci_95_range"] = [round(lower * 100, 2), round(upper * 100, 2)]


def _normalize_patterns_from_spec(spec_input):
    """
    Normalize a prompt specification (string key or dict) and return
    (answer_type, compiled_custom_patterns).
    """
    if spec_input is None:
        return None, []

    if isinstance(spec_input, str):
        if spec_input not in PROMPT_LIBRARY:
            raise ValueError(f"Prompt '{spec_input}' not found in PROMPT_LIBRARY.")
        spec = deepcopy(PROMPT_LIBRARY[spec_input])
    elif isinstance(spec_input, dict):
        spec = deepcopy(spec_input)
    else:
        raise TypeError("prompt_spec must be a str key or a dict specification.")

    spec.setdefault("answer_type", "option")
    spec.setdefault("custom_answer_patterns", [])
    spec["custom_answer_patterns"] = _compile_patterns(spec["custom_answer_patterns"])
    return spec["answer_type"], spec["custom_answer_patterns"]


def _build_extra_pattern_map(prompt_spec):
    """
    Accepts None, a single spec (str/dict), or a mapping like
    {"option": spec_a, "number": spec_b}. Returns a tuple:
    ({answer_type: pattern_list}, forced_answer_type or None).
    """
    if not prompt_spec:
        return {}, None

    if (
        isinstance(prompt_spec, dict)
        and prompt_spec
        and set(prompt_spec.keys()).issubset({"option", "number"})
        and all(isinstance(v, (str, dict)) for v in prompt_spec.values())
    ):
        pattern_map = {}
        for ans_type, spec in prompt_spec.items():
            _, patterns = _normalize_patterns_from_spec(spec)
            if patterns:
                pattern_map[ans_type] = patterns
        return pattern_map, None

    answer_type, patterns = _normalize_patterns_from_spec(prompt_spec)
    pattern_map = {answer_type: patterns} if patterns else {}
    return pattern_map, answer_type


def get_answer(file_path, save_dir, tail_fallback=True, prompt_spec=None):
    """
    Processes a JSONL file to evaluate model answers which are expected to be numbers.
    """
    print(file_path)
    results = load_jsonl(file_path)
    extra_pattern_map, forced_answer_type = _build_extra_pattern_map(prompt_spec)
    
    counting = {}
    counting["total"] = {"correct_num": 0, "total_num": 0}
    for i, result in enumerate(results):
        if "Response" in result and result["Response"] == "Failed!!!":
            continue
        
        counting["total"]["total_num"] += 1
        
        category = result["Category"]
        task = result["Task"]
        level = result["Level"]
        key = f"{category}-{task}-{level}"
        if category not in counting:
            counting[category] = {"correct_num": 0, "total_num": 1}
        else:
            counting[category]["total_num"] += 1
        
        if task not in counting:
            counting[task] = {"correct_num": 0, "total_num": 1}
        else:
            counting[task]["total_num"] += 1
        
        if key not in counting:
            counting[key] = {"correct_num": 0, "total_num": 1}
        else:
            counting[key]["total_num"] += 1
        
        question_type = forced_answer_type or ("number" if 'Answer_direct' in result else "option")
        ground_truth = str(result['Answer_direct']).strip() if question_type == "number" else result['Answer']
        
        predicted = None
        custom_patterns = extra_pattern_map.get(question_type)
        
        # Prefer FinalAnswer produced by run_model_for_item
        if "FinalAnswer" in result:
            pred_str = str(result["FinalAnswer"]).strip()
            predicted = extract_answer(
                pred_str, 
                question_type,
                extra_patterns=custom_patterns,
                allow_tail_fallback=tail_fallback
            )
        
        # Fall back to Response only if FinalAnswer is missing
        if predicted is None and "Response" in result:   
            response = result["Response"].split('<｜end of sentence｜>')[0]
            # Call the fixed extract_answer (now uses the updated _cleanup_answer logic)
            predicted = extract_answer(
                response, 
                question_type,
                extra_patterns=custom_patterns,
                allow_tail_fallback=tail_fallback
            )
        
        # if predicted_v2 != predicted:
        #     print(predicted, predicted_v2)
        #     print(str(result["FinalAnswer"]).strip())
        #     print({"response": response})
        #     print()

        if predicted is not None and ground_truth == str(predicted).strip():
            counting["total"]["correct_num"] += 1
            counting[category]["correct_num"] += 1
            counting[task]["correct_num"] += 1
            counting[key]["correct_num"] += 1
        
   
    for k in counting:
        _attach_accuracy_stats(counting[k])
    
    dump_json(f"{save_dir}/{file_path.split(os.sep)[-1][:-6]}_counting.json", counting)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_list", type=str, nargs='+', required=False,
                        default=["qwen2.5-vl-3b-instruct"],
                        help="List of model names or local paths to the models")
    parser.add_argument('--benchmark_test_path', type=str, required=True,
                        help="Root directory of the benchmark data.")
    parser.add_argument('--save_dir', type=str, required=True,
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
                        help="Maximum times to retry a failed API call per instance.")
    parser.add_argument('--request_interval', type=float, default=1.0,
                        help="Seconds to sleep between API attempts.")
    parser.add_argument('--text_only', action='store_true',
                        help="If set, skip attaching images and evaluate as text-only prompts.")
    parser.add_argument('--enable_sampling', action='store_true',
                        help="Enable random sampling per (Category, Task, Level) before evaluation.")
    parser.add_argument('--sample_per_level', type=int, default=10,
                        help="Number of samples per (Category, Task, Level) group when sampling is enabled.")
    parser.add_argument('--sample_seed', type=int, default=42,
                        help="Random seed used for sampling when --enable_sampling is set.")
    parser.add_argument('--logprobs', action='store_true',
                        help="Whether to request log probabilities from the API.")
    parser.add_argument('--top_logprobs', type=int, default=6,
                        help="Number of top log probabilities to request when logprobs is enabled.")
    parser.add_argument('--qwen_api_key', type=str, default=None,
                        help="API key for Qwen (DashScope) models.")
    parser.add_argument('--doubao_api_key', type=str, default=None,
                        help="API key for Doubao models.")
    parser.add_argument('--openai_api_key', type=str, default=None,
                        help="API key for OpenAI models.")
    parser.add_argument('--gemini_api_key', type=str, default=None,
                        help="API key for Gemini models.")
    parser.add_argument('--openrouter_api_key', type=str, default=None,
                        help="API key for OpenRouter models.")
     
    # (Keep tail_fallback disabled by default)
    parser.add_argument('--enable_tail_fallback', action='store_true',
                        help="[Not recommended] Enable low-accuracy fallback extraction when tags are missing.")
    args = parser.parse_args()
    print("Parsed args:", args)
    
    tail_fallback = args.enable_tail_fallback 
    use_images = not args.text_only

    data = load_json(f"{args.benchmark_test_path}/{args.data_file}")
    if args.enable_sampling:
        data = sample_data_per_level(
            data,
            args.sample_per_level,
            seed=args.sample_seed,
        )
    if args.run_mode == "modify":
        modify(
            args.model_list,
            data,
            args.benchmark_test_path,
            args.save_dir,
            use_images=use_images,
            logprobs=args.logprobs,
            top_logprobs=args.top_logprobs,
            use_direct_answer=args.use_direct_answer,
            choice_prompt=args.choice_prompt,
            direct_prompt=args.direct_prompt,
            max_retries=args.max_retries,
            request_interval=args.request_interval,
            tail_fallback=tail_fallback, 
            qwen_api_key=args.qwen_api_key,
            doubao_api_key=args.doubao_api_key,
            openai_api_key=args.openai_api_key,
            gemini_api_key=args.gemini_api_key,
            openrouter_api_key=args.openrouter_api_key,
        )
    else:
        evaluate(
            args.model_list,
            data,
            args.benchmark_test_path,
            args.save_dir,
            use_images=use_images,
            logprobs=args.logprobs,
            top_logprobs=args.top_logprobs,
            use_direct_answer=args.use_direct_answer,
            choice_prompt=args.choice_prompt,
            direct_prompt=args.direct_prompt,
            max_retries=args.max_retries,
            request_interval=args.request_interval,
            tail_fallback=tail_fallback, 
            qwen_api_key=args.qwen_api_key,
            doubao_api_key=args.doubao_api_key,
            openai_api_key=args.openai_api_key,
            gemini_api_key=args.gemini_api_key,
            openrouter_api_key=args.openrouter_api_key,
        )
    
    # ... (local get_answer invocation commented out)
