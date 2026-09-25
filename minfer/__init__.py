def __getattr__(name):
    # lazy so that importing minfer.kernels doesn't pull in the whole engine
    if name == "LLMEngine":
        from minfer.engine import LLMEngine
        return LLMEngine
    if name == "SamplingParams":
        from minfer.sampling import SamplingParams
        return SamplingParams
    raise AttributeError(name)
