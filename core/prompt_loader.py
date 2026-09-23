from pathlib import Path

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

_cache: dict[str, str] = {}


def load_prompt(name: str, **kwargs) -> str:
    """Load a prompt template from prompts/ and format with kwargs."""
    if name not in _cache:
        filepath = PROMPTS_DIR / f"{name}.txt"
        if filepath.exists():
            _cache[name] = filepath.read_text(encoding="utf-8")
        else:
            _cache[name] = ""
    template = _cache[name]
    if kwargs:
        template = template.format(**kwargs)
    return template
