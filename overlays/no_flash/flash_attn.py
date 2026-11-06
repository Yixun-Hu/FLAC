"""No-flash overlay (exp_35; Yixun 2026-11-05 decision 2). Shadows the installed flash_attn so that
src/models/transformer.py:12-18 falls back to torch SDPA, the attention backend exp_25 ran with.
Put on PYTHONPATH by depth_canon_eval_run.sh only; never installed, never imported elsewhere."""
raise ImportError("exp35 no-flash overlay: flash_attn disabled, attention backend = torch SDPA")
