# Jev prose features

An on-demand, advisory Hermes plugin for examining how Japanese prose reads to its intended reader in context. Taxonomy `jev-prose-features-1.8` preserves the original nine independent feature axes and adds `action_effect_substitution` as a separate advisory axis. The existing `scope_boundary_blurring` remains a mismatch between actor, object, conditions, or focus expected by the stated purpose/context and the sentence's focus. The new axis checks only whether an action-guidance purpose has been replaced by a possibility/benefit/effect statement; capability/effect explanations remain absent when that is the purpose. Neither abstract wording nor 「できます／しやすくなります」 alone is a mismatch; both axes require evidence in supplied purpose/context. It returns source-bound labels, confidence, and small contextual review topics; it does not judge factual correctness or infer translation/authorship.

## Behavior and boundaries

- Registers `jev_classify_prose_features` only; no ordinary-chat hook, autosend path, automatic rewrite, deletion, worker block, or Kanban mutation.
- Classification is explicit and uses the configured TypeSafe or OpenRouter Jev decisions route. Provider/network/validation failures return `unknown`, not a false “no issues” result. Calls may disclose the supplied text to the explicitly configured external provider; do not send private/sensitive text without separate authorization.
- `review_candidates` is a contextual, human review aid at threshold 0.5, not a replacement for raw axis results or a rewrite request. `author_style_impression` alone never creates a review candidate.
- Taxonomy `jev-prose-features-1.8` has ten independent axes: `empty_preamble_conclusion`, `abstract_action_unclear`, `redundant_paraphrase`, `unnecessary_contrast`, `excessive_praise_empathy`, `japanese_naturalness`, `scope_expression`, `scope_boundary_blurring`, `action_effect_substitution`, and `author_style_impression`. The original nine axes and their criteria/meanings are retained. Binary and categorical choices retain their documented semantics and raw confidence. The boundary-shift axis remains advisory and independent: compare purpose/context-expected focus with the sentence's actual actor, object, conditions, and conversational focus; never infer an unstated expected focus. The new action/effect axis separately checks whether an action-guidance purpose is replaced only by possibility/benefit/effect language; when capability/effect explanation itself is the purpose, mark it absent. A final form such as 「できます／しやすくなります」 alone is insufficient for either axis. No automatic rewrite, deletion, or stop action follows. A review candidate is emitted only for a `present` result at confidence >= 0.5.
- `sentence_context.py` is offline only. It prepares stable sentence targets, whole-document previous/next context, global Python code-point spans, and up to 48 targets (12 sequential batches of at most four). It does not infer heading exclusions or call a provider; over-cap inputs raise rather than truncate.
- Explicit segment `previous_sentence` / `next_sentence` values are context only, never extra targets. Input limits and two-call cap are validated before sending. No automatic retry or fallback is performed.

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
