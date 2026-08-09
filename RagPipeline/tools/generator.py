from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from RagPipeline.tools.text_cleaner import clean_text
# Module-level cache — avoids reloading models on every call
_model_cache = {}


def _load_model(model_name: str, device: str):
    """Lazy-load tokenizer + model, cached per (model_name, device) key."""
    cache_key = (model_name, device)
    if cache_key not in _model_cache:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        ).to(device)
        model.eval()
        _model_cache[cache_key] = (tokenizer, model)
    return _model_cache[cache_key]


def _format_context(context) -> str:
    """
    Convert retrieved chunks (list of dicts) into a clean text block
    that the LLM can read and cite.
    """
    if isinstance(context, list):
        parts = []
        for i, chunk in enumerate(context, start=1):
            chunk_id = chunk.get("chunk_id", i)
            chunk_text = chunk.get("chunk_text", str(chunk))
            parts.append(f"[Chunk {chunk_id}]\n{chunk_text}")
        return "\n\n".join(parts)
    # Fallback: plain string or unexpected type
    return str(context)


def create_generator(
    query: str,
    context,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    max_token: int = 512,
    device: str = "cpu",
):
    """
    Generate an answer from retrieved context using a local LLM.

    Parameters
    ----------
    query : str
        The user question.
    context : list[dict] | str
        Retrieved chunks from retrieve_top_k (list of dicts) or raw text.
    model_name : str
        HuggingFace model ID.
    max_token : int
        Maximum new tokens to generate.
    device : str
        'cpu' or 'cuda'.

    Returns
    -------
    str
        The generated answer.
    """
    if query is None or len(query.strip()) == 0:
        raise ValueError("Question cannot be empty.")
    
    # ---- Build readable context string ----
    context_text = _format_context(context)
    query = clean_text(query)
    
    # ---- Build prompt ----
    prompt_text = f"""You are a helpful assistant that answers questions based ONLY on the provided context.
    
    Context:
    {context_text}

    Question: {query}

    Instructions:
    1. Answer using ONLY the information from the context above
    2. Cite the source chunks using [Chunk X] where X is the chunk number
    3. If the context doesn't contain the answer, say "I don't have enough information to answer this question"

    Answer: 
    """.strip()

    # ---- Load model (cached — only first call downloads) ----
    tokenizer, model = _load_model(model_name, device)

    # ---- Tokenize ----
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    # ---- Generate ----
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_token,
            temperature=0.7,
            top_p=0.9,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Strip the prompt portion from the decoded output
    response = response[len(prompt_text):].strip()
    return response