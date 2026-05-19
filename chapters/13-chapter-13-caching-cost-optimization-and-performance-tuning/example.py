import time
from functools import lru_cache

# --- Core Idea: Caching for Performance and Cost Optimization ---
# LLM calls can be expensive (time and money). Caching frequently
# requested or computationally intensive results avoids redundant calls.

# Simulate an expensive LLM call
def _expensive_llm_call(prompt: str) -> str:
    """
    Simulates a time-consuming and 'costly' LLM API call.
    In a real scenario, this would involve network I/O and API billing.
    """
    print(f"Simulating expensive LLM call for: '{prompt}'...")
    time.sleep(2)  # Simulate network latency and processing time
    return f"Response to: {prompt} from LLM."

# --- Strategy 1: Using functools.lru_cache for in-memory caching ---
# This is simple and effective for many cases.
@lru_cache(maxsize=128)  # Cache up to 128 recent unique results
def cached_llm_response(prompt: str) -> str:
    """
    A cached version of the LLM call using functools.lru_cache.
    The first call for a given prompt will be slow, subsequent calls fast.
    """
    return _expensive_llm_call(prompt)

# --- Demonstration ---

print("--- First set of calls (demonstrating cache misses and hits) ---")
start_time = time.time()
print(f"Call 1a: {cached_llm_response('What is Generative AI?')}")
print(f"Call 1b: {cached_llm_response('What is LangChain?')}")
end_time = time.time()
print(f"Time taken for first two unique calls: {end_time - start_time:.2f} seconds\n")

start_time = time.time()
print(f"Call 1c (cached hit): {cached_llm_response('What is Generative AI?')}")
print(f"Call 1d (cached hit): {cached_llm_response('What is LangChain?')}")
end_time = time.time()
print(f"Time taken for two cached calls: {end_time - start_time:.2f} seconds\n") # Significantly faster!

print("--- Second set of calls (demonstrating a new prompt and cache hit) ---")
start_time = time.time()
print(f"Call 2a: {cached_llm_response('Explain large language models.')}") # New prompt, cache miss
print(f"Call 2b (cached hit): {cached_llm_response('Explain large language models.')}")
end_time = time.time()
print(f"Time taken for new prompt and its cached hit: {end_time - start_time:.2f} seconds\n")

# --- Benefits Illustrated ---
# 1. Performance: Subsequent calls for the same prompt are almost instant.
# 2. Cost Optimization: Avoids re-calling the 'expensive' LLM API,
#    saving on API costs (though not directly demonstrated here).
# 3. Reduced API Rate Limits: Helps stay within LLM provider rate limits.

# Note: For production, consider persistent caching (e.g., Redis, database)
# and more sophisticated cache invalidation strategies.
