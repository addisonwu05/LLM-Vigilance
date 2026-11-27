"""Numbers as printed in the paper (LaTeX tables), used as regression targets."""

# Appendix table exp2_results_full: (model key, perspective, instructions) -> (Bayesian-LLM, Bayesian-Human, LLM-Human)
EXP2_FULL = {
    ("gpt-4o", "first", "cot"): (0.909, 0.925, 0.936), ("gpt-4o", "user", "cot"): (0.909, 0.925, 0.945),
    ("gpt-4o", "first", "direct"): (0.925, 0.925, 0.944), ("gpt-4o", "user", "direct"): (0.899, 0.940, 0.948),
    ("claude-3.5-sonnet", "first", "cot"): (0.843, 0.895, 0.931), ("claude-3.5-sonnet", "user", "cot"): (0.827, 0.901, 0.918),
    ("claude-3.5-sonnet", "first", "direct"): (0.866, 0.890, 0.966), ("claude-3.5-sonnet", "user", "direct"): (0.844, 0.871, 0.948),
    ("gemini-2.0-flash", "first", "cot"): (0.802, 0.906, 0.918), ("gemini-2.0-flash", "user", "cot"): (0.771, 0.871, 0.918),
    ("gemini-2.0-flash", "first", "direct"): (0.789, 0.920, 0.923), ("gemini-2.0-flash", "user", "direct"): (0.788, 0.905, 0.941),
    ("llama-3.3-70b", "first", "cot"): (0.908, 0.923, 0.926), ("llama-3.3-70b", "user", "cot"): (0.879, 0.907, 0.909),
    ("llama-3.3-70b", "first", "direct"): (0.851, 0.935, 0.937), ("llama-3.3-70b", "user", "direct"): (0.867, 0.928, 0.917),
    ("o1", "first", "reasoning"): (0.763, 0.906, 0.909), ("o1", "user", "reasoning"): (0.646, 0.882, 0.813),
    ("o3-mini", "first", "reasoning"): (0.764, 0.868, 0.772), ("o3-mini", "user", "reasoning"): (0.667, 0.870, 0.667),
    ("deepseek-r1", "first", "reasoning"): (0.793, 0.823, 0.841), ("deepseek-r1", "user", "reasoning"): (-0.141, 0.160, 0.445),
    ("llama-3.1-8b", "first", "cot"): (0.614, 0.784, 0.742), ("llama-3.1-8b", "user", "cot"): (0.613, 0.806, 0.694),
    ("llama-3.1-8b", "first", "direct"): (0.611, 0.819, 0.682), ("llama-3.1-8b", "user", "direct"): (0.593, 0.844, 0.686),
    ("llama-3.2-3b", "first", "cot"): (0.491, 0.604, 0.548), ("llama-3.2-3b", "user", "cot"): (0.238, 0.549, 0.563),
    ("llama-3.2-3b", "first", "direct"): (0.293, 0.570, 0.599), ("llama-3.2-3b", "user", "direct"): (0.372, 0.621, 0.490),
    ("gemma-3-4b", "first", "cot"): (0.292, 0.240, 0.513), ("gemma-3-4b", "user", "cot"): (0.403, 0.572, 0.279),
    ("gemma-3-4b", "first", "direct"): (-0.017, 0.104, 0.054), ("gemma-3-4b", "user", "direct"): (0.472, 0.442, 0.219),
}

# Table exp_2_corrs1 (LLM_bayesian_human.tex): model -> (Bayesian-LLM, Bayesian-Human, LLM-Human)
EXP2_MAIN = {
    "gpt-4o": (0.911, 0.929, 0.943), "claude-3.5-sonnet": (0.845, 0.889, 0.941),
    "gemini-2.0-flash": (0.788, 0.901, 0.925), "llama-3.3-70b": (0.876, 0.923, 0.922),
    "o1": (0.705, 0.894, 0.861), "o3-mini": (0.716, 0.869, 0.712), "deepseek-r1": (0.326, 0.492, 0.643),
    "llama-3.1-8b": (0.608, 0.813, 0.701), "llama-3.2-3b": (0.349, 0.586, 0.550), "gemma-3-4b": (0.288, 0.340, 0.266),
}

# Appendix table exp2_prompting_alignment (a), intra-character alignment (section_b.tex);
# panel (b) repeats the LLM-Human column of EXP2_FULL. "Llama" there is Llama 3.3-70B.
EXP2_INTRA = {
    ("llama-3.3-70b", "first", "cot"): 0.996, ("deepseek-r1", "first", "reasoning"): 0.995,
    ("gemini-2.0-flash", "user", "direct"): 0.994, ("claude-3.5-sonnet", "user", "cot"): 0.994,
    ("gemini-2.0-flash", "first", "direct"): 0.994, ("llama-3.3-70b", "user", "cot"): 0.994,
    ("claude-3.5-sonnet", "first", "cot"): 0.994, ("gpt-4o", "first", "direct"): 0.993,
    ("o1", "first", "reasoning"): 0.993, ("gpt-4o", "first", "cot"): 0.993,
    ("claude-3.5-sonnet", "user", "direct"): 0.992, ("gpt-4o", "user", "cot"): 0.991,
    ("claude-3.5-sonnet", "first", "direct"): 0.991, ("gemini-2.0-flash", "user", "cot"): 0.992,
    ("gemini-2.0-flash", "first", "cot"): 0.991, ("gpt-4o", "user", "direct"): 0.987,
    ("llama-3.3-70b", "first", "direct"): 0.980, ("llama-3.3-70b", "user", "direct"): 0.970,
    ("o1", "user", "reasoning"): 0.947, ("o3-mini", "first", "reasoning"): 0.932,
    ("o3-mini", "user", "reasoning"): 0.851, ("deepseek-r1", "user", "reasoning"): 0.472,
}

# Table youtube_steering: (model, instructions, perspective) -> (default, incentives steer)
EXP3_STEERING = {
    ("gpt-4o", "cot", "first"): (0.024, 0.137), ("gpt-4o", "cot", "user"): (0.008, 0.143),
    ("gpt-4o", "direct", "first"): (0.121, 0.234), ("gpt-4o", "direct", "user"): (-0.006, 0.312),
    ("claude-3.5-sonnet", "cot", "first"): (0.033, 0.215), ("claude-3.5-sonnet", "cot", "user"): (0.190, 0.214),
    ("claude-3.5-sonnet", "direct", "first"): (0.094, 0.200), ("claude-3.5-sonnet", "direct", "user"): (0.119, 0.283),
    ("llama-3.3-70b", "cot", "first"): (-0.011, 0.029), ("llama-3.3-70b", "cot", "user"): (0.032, 0.152),
    ("llama-3.3-70b", "direct", "first"): (0.062, 0.104), ("llama-3.3-70b", "direct", "user"): (0.098, 0.126),
}
# rows the paper marks with * (significant improvement)
EXP3_STARRED = {("gpt-4o", "cot", "first"), ("gpt-4o", "cot", "user"), ("gpt-4o", "direct", "first"), ("gpt-4o", "direct", "user"),
                ("claude-3.5-sonnet", "cot", "first"), ("claude-3.5-sonnet", "direct", "first"), ("claude-3.5-sonnet", "direct", "user"),
                ("llama-3.3-70b", "cot", "user")}

# Appendix table youtube_steering_additional (GPT-4o): (instructions, perspective) -> (none, incentives, gricean, bias)
EXP3_ADDITIONAL = {
    ("cot", "first"): (0.0240, 0.1367, 0.0849, 0.0759), ("cot", "user"): (0.0082, 0.1431, 0.1793, 0.1007),
    ("direct", "first"): (0.1211, 0.2338, -0.0184, 0.0901), ("direct", "user"): (-0.0056, 0.3121, -0.0097, 0.2321),
}

# Table model_correlation_quartiles (incentives steer): (model, perspective, instructions, quartile) -> r
EXP3_LENGTH = {
    ("llama-3.3-70b", "first", "cot", "Q1"): 0.142, ("llama-3.3-70b", "first", "cot", "Q4"): 0.029,
    ("gpt-4o", "first", "cot", "Q1"): 0.247, ("gpt-4o", "first", "cot", "Q4"): -0.098,
    ("claude-3.5-sonnet", "first", "cot", "Q1"): -1.67e-16, ("claude-3.5-sonnet", "first", "cot", "Q4"): -6.45e-16,
    ("llama-3.3-70b", "first", "direct", "Q1"): 0.178, ("llama-3.3-70b", "first", "direct", "Q4"): 0.080,
    ("gpt-4o", "first", "direct", "Q1"): 0.386, ("gpt-4o", "first", "direct", "Q4"): -0.001,
    ("claude-3.5-sonnet", "first", "direct", "Q1"): 0.304, ("claude-3.5-sonnet", "first", "direct", "Q4"): -5.58e-16,
    ("llama-3.3-70b", "user", "cot", "Q1"): 0.191, ("llama-3.3-70b", "user", "cot", "Q4"): 0.027,
    ("gpt-4o", "user", "cot", "Q1"): 4.72e-16, ("gpt-4o", "user", "cot", "Q4"): -6.90e-16,
    ("claude-3.5-sonnet", "user", "cot", "Q1"): 0.301, ("claude-3.5-sonnet", "user", "cot", "Q4"): 0.224,
    ("llama-3.3-70b", "user", "direct", "Q1"): 0.157, ("llama-3.3-70b", "user", "direct", "Q4"): 0.093,
    ("gpt-4o", "user", "direct", "Q1"): 0.303, ("gpt-4o", "user", "direct", "Q4"): 0.075,
    ("claude-3.5-sonnet", "user", "direct", "Q1"): 0.232, ("claude-3.5-sonnet", "user", "direct", "Q4"): 1.02e-15,
}
