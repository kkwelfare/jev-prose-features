"""Offline regression tests for docs-specific sentence/context preparation."""
import unittest

from sentence_context import CoverageLimitError, MAX_BATCHES_PER_PASS, MAX_TARGETS_PER_BATCH, prepare_segments


class SentenceContextTests(unittest.TestCase):
    def test_neighbors_are_separate_and_offsets_bind_to_original_codepoints(self):
        body = "🙂冒頭です。『引用です。』と担当者は言いました。\n次の文です！最後です。"
        prepared = prepare_segments(body, purpose="説明", context="記事全体", reader="初めて読む人")
        self.assertEqual(prepared["target_count"], 4)
        rows = [row for batch in prepared["batches"] for row in batch["segments"]]
        self.assertEqual(rows[0]["text"], "🙂冒頭です。")
        self.assertEqual(rows[1]["text"], "『引用です。』と担当者は言いました。")
        self.assertEqual(rows[1]["previous_sentence"], rows[0]["text"])
        self.assertEqual(rows[1]["next_sentence"], "次の文です！")
        self.assertEqual(rows[0]["next_sentence"], rows[1]["text"])
        spans = [span for batch in prepared["batches"] for span in batch["source_spans"]]
        for row, span in zip(rows, spans):
            self.assertEqual(body[span["start"]:span["end"]], row["text"])
            self.assertEqual(row["id"], span["id"])
        self.assertEqual(rows[-1]["next_sentence"], "")
        self.assertEqual(rows[0]["previous_sentence"], "")

    def test_batch_edges_keep_global_neighbors_and_limits_are_explicit(self):
        body = "".join(f"文{i}です。" for i in range(1, 10))
        prepared = prepare_segments(body, purpose="確認", context="全体", reader="読者")
        self.assertEqual(prepared["batch_count"], 3)
        left = prepared["batches"][0]["segments"][-1]
        right = prepared["batches"][1]["segments"][0]
        self.assertEqual(left["next_sentence"], right["text"])
        self.assertEqual(right["previous_sentence"], left["text"])
        over_cap = "".join(f"文{i}。" for i in range(1, MAX_TARGETS_PER_BATCH * MAX_BATCHES_PER_PASS + 2))
        with self.assertRaises(CoverageLimitError) as raised:
            prepare_segments(over_cap, purpose="確認", context="全体")
        self.assertEqual(raised.exception.target_count, 49)

    def test_headings_are_never_heuristically_removed_and_explicit_exclusion_must_be_exact(self):
        body = "見出し。本文です。"
        prepared = prepare_segments(body, purpose="確認", context="全体")
        self.assertEqual(prepared["target_count"], 2)
        excluded = prepare_segments(body, purpose="確認", context="全体", exclude_spans=[{"start": 0, "end": 4}])
        self.assertEqual(excluded["target_count"], 1)
        self.assertEqual(excluded["batches"][0]["segments"][0]["text"], "本文です。")
        with self.assertRaises(ValueError):
            prepare_segments(body, purpose="確認", context="全体", exclude_spans=[{"start": 0, "end": 2}])


if __name__ == "__main__":
    unittest.main()
