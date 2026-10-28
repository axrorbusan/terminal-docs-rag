"""Thin wrapper around the Anthropic Messages API.

These calls are short (query rewrite, cited answer, grading), so extended thinking is
turned off: with it on, a model can spend the whole max_tokens budget thinking and
return no text. Models differ in how thinking is switched off, so the setting that a
model accepts is remembered after the first call.
"""
import anthropic

# Tried in order. None = leave the parameter out (models that only allow adaptive
# thinking); max_tokens is then raised so thinking cannot use up the whole budget.
_THINKING_OFF = [{"type": "disabled"}, {"type": "between_tools"}, None]
_accepted = {}


def text_of(msg):
    """Join the text blocks of a response (skips thinking or other non-text blocks)."""
    return "".join(b.text for b in msg.content if getattr(b, "type", "text") == "text")


def create(client, model, **kw):
    options = [_accepted[model]] if model in _accepted else _THINKING_OFF
    last = None
    for opt in options:
        try:
            if opt is None:
                kw2 = dict(kw, max_tokens=max(kw.get("max_tokens", 0), 4000))
                msg = client.messages.create(model=model, **kw2)
            else:
                msg = client.messages.create(model=model, thinking=opt, **kw)
            _accepted[model] = opt
            return msg
        except anthropic.BadRequestError as e:
            if "thinking" not in str(e):
                raise
            last = e
    raise last
