# Anvil

**Distil LLM calls into small, fast task models, with measured parity.**

Give Anvil a prompt and a file of unlabeled inputs. It labels a batch with
Claude, fine-tunes a small model in PyTorch (ModernBERT for classification
and extraction, Qwen2.5-0.5B with LoRA for constrained generation), measures
agreement with the teacher on a held-out set as Cohen's kappa with a
confidence interval, picks the next batch to label by uncertainty and
diversity, and repeats until a parity target or a dollar budget is hit.
The result is a calibrated student exported to ONNX with a drop-in server
that escalates low-confidence inputs back to the teacher, plus the
cost-versus-quality frontier that says how many teacher labels it took.

Status: **in progress** (planning done; see [PLAN.md](PLAN.md) for the
milestones and [CONTRIBUTING.md](CONTRIBUTING.md) for how to work on it).
