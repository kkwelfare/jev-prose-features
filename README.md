# Jev prose features

An on-demand, advisory Hermes plugin for examining how Japanese prose reads to its intended reader in context. It preserves eight independent feature axes and returns source-bound labels, confidence, and small contextual review topics; it does not judge factual correctness or infer translation/authorship.

## Behavior and boundaries

- Registers `jev_classify_prose_features` only; no ordinary-chat hook, autosend path, automatic rewrite, deletion, worker block, or Kanban mutation.
- Classification is explicit and uses the configured TypeSafe or OpenRouter Jev decisions route. Provider/network/validation failures return `unknown`, not a false “no issues” result. Calls may disclose the supplied text to the explicitly configured external provider; do not send private/sensitive text without separate authorization.
- `review_candidates` is a contextual, human review aid at threshold 0.5, not a replacement for raw axis results or a rewrite request. `author_style_impression` alone never creates a review candidate.
- Taxonomy `jev-prose-features-1.5` has eight independent axes: `empty_preamble_conclusion`, `abstract_action_unclear`, `redundant_paraphrase`, `unnecessary_contrast`, `excessive_praise_empathy`, `japanese_naturalness`, `scope_expression`, and `author_style_impression`. Binary and categorical choices retain their documented semantics and raw confidence.
- `sentence_context.py` is offline only. It prepares stable sentence targets, whole-document previous/next context, global Python code-point spans, and up to 48 targets (12 sequential batches of at most four). It does not infer heading exclusions or call a provider; over-cap inputs raise rather than truncate.
- Explicit segment `previous_sentence` / `next_sentence` values are context only, never extra targets. Input limits and two-call cap are validated before sending. No automatic retry or fallback is performed.

## Lifecycle logging

The registered tool emits metadata-only `jev.lifecycle` INFO events for provider attempts, validation, and tool-result delivery. Request text and credentials are not logged; logging failures preserve the original advisory result.

## Use

Install this directory with the Hermes plugin manager and explicitly enable the plugin/tool in the intended profile. No credential or gateway setup is performed by this package. A docs proofreading flow can run its deterministic local filter first, then use `sentence_context.prepare_segments(...)` to prepare bounded advisory requests. The helper never sends text.

Offline CLI validation (no provider call):

```sh
printf '%s\n' '{"segments":[{"id":"synthetic-1","text":"確認します。","previous_sentence":"前の文。","next_sentence":"次の文。"}],"purpose":"説明","context":"文脈確認","reader":"初めて読む人"}' | python3 cli.py --validate-only
```

## Verify

```sh
python3 -m unittest -v test_core.py test_sentence_context.py
hermes plugins validate . --json
```

The tests use mocks and synthetic strings. They do not demonstrate real-provider classification accuracy or make a billable provider call. This public source package excludes task-specific article samples, private sentence maps, scratch drivers, and backups. The README reflects the optional bounded docs-proofreading workflow; callers must preserve the full document and report uncovered scope rather than silently truncating it.
