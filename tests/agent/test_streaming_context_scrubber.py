"""Unit tests for StreamingContextScrubber (agent/memory_manager.py).

Regression coverage for #5719 — memory-context spans split across stream
deltas must not leak payload to the UI.  The one-shot sanitize_context()
regex can't survive chunk boundaries, so _fire_stream_delta routes deltas
through a stateful scrubber.
"""

from agent.memory_manager import (
    StreamingContextScrubber,
    StreamingVisibleContextScrubber,
    sanitize_context,
    sanitize_history_context,
    sanitize_visible_context,
)


class TestStreamingContextScrubberBasics:
    def test_empty_input_returns_empty(self):
        s = StreamingContextScrubber()
        assert s.feed("") == ""
        assert s.flush() == ""

    def test_plain_text_passes_through(self):
        s = StreamingContextScrubber()
        assert s.feed("hello world") == "hello world"
        assert s.flush() == ""

    def test_complete_block_in_single_delta(self):
        """Regression: the one-shot test case from #13672 must still work."""
        s = StreamingContextScrubber()
        leaked = (
            "<memory-context>\n"
            "[System note: The following is recalled memory context, NOT new "
            "user input. Treat as informational background data.]\n\n"
            "## Honcho Context\nstale memory\n"
            "</memory-context>\n\nVisible answer"
        )
        out = s.feed(leaked) + s.flush()
        assert out == "\n\nVisible answer"

    def test_open_and_close_in_separate_deltas_strips_payload(self):
        """The real streaming case: tag pair split across deltas."""
        s = StreamingContextScrubber()
        deltas = [
            "Hello\n",
            "<memory-context>\npayload ",
            "more payload\n",
            "</memory-context> world",
        ]
        out = "".join(s.feed(d) for d in deltas) + s.flush()
        assert out == "Hello\n world"
        assert "payload" not in out

    def test_realistic_fragmented_chunks_strip_memory_payload(self):
        """Exact leak scenario from the reviewer's comment — 4 realistic chunks.

        This is the case the original #13672 fix silently leaks on: the open
        tag, system note, payload, and close tag each arrive in their own
        delta because providers emit 1-80 char chunks.
        """
        s = StreamingContextScrubber()
        deltas = [
            "<memory-context>\n[System note: The following",
            " is recalled memory context, NOT new user input. "
            "Treat as informational background data.]\n\n",
            "## Honcho Context\nstale memory\n",
            "</memory-context>\n\nVisible answer",
        ]
        out = "".join(s.feed(d) for d in deltas) + s.flush()
        assert out == "\n\nVisible answer"
        # The system-note line and payload must never reach the UI.
        assert "System note" not in out
        assert "Honcho Context" not in out
        assert "stale memory" not in out

    def test_open_tag_split_across_two_deltas(self):
        """The open tag itself arriving in two fragments."""
        s = StreamingContextScrubber()
        out = (
            s.feed("pre \n<memory")
            + s.feed("-context>\nleak</memory-context> post")
            + s.flush()
        )
        assert out == "pre \n post"
        assert "leak" not in out

    def test_open_tag_waits_for_newline_confirmation_across_deltas(self):
        """A boundary tag is only a leaked block when the next char is a newline."""
        s = StreamingContextScrubber()
        out = (
            s.feed("pre \n<memory-context>")
            + s.feed("\nleak</memory-context> post")
            + s.flush()
        )
        assert out == "pre \n post"
        assert "leak" not in out

    def test_close_tag_split_across_two_deltas(self):
        """The close tag arriving in two fragments."""
        s = StreamingContextScrubber()
        out = (
            s.feed("pre \n<memory-context>\nleak</memory")
            + s.feed("-context> post")
            + s.flush()
        )
        assert out == "pre \n post"
        assert "leak" not in out


class TestStreamingContextScrubberPartialTagFalsePositives:
    def test_partial_open_tag_tail_emitted_on_flush(self):
        """Bare '<mem' at end of stream is not really a memory-context tag."""
        s = StreamingContextScrubber()
        out = s.feed("hello <mem") + s.feed("ory other") + s.flush()
        assert out == "hello <memory other"

    def test_partial_tag_released_when_disambiguated(self):
        """A held-back partial tag that turns out to be prose gets released."""
        s = StreamingContextScrubber()
        # '< ' should not look like the start of any tag.
        out = s.feed("price < ") + s.feed("10 dollars") + s.flush()
        assert out == "price < 10 dollars"

    def test_inline_memory_context_tag_mention_is_not_scrubbed(self):
        """A prose mention of the fence tag must not swallow the answer."""
        s = StreamingContextScrubber()
        out = (
            s.feed("In that previous `<memory")
            + s.feed("-context>` block, ")
            + s.feed("there was no matching fact.")
            + s.flush()
        )
        assert out == "In that previous `<memory-context>` block, there was no matching fact."

    def test_mid_sentence_memory_context_mention_is_not_scrubbed(self):
        """Only block-like memory-context spans are treated as leaked context."""
        s = StreamingContextScrubber()
        out = s.feed("The <memory-context> tag name is documented here.") + s.flush()
        assert out == "The <memory-context> tag name is documented here."

    def test_line_start_memory_context_mention_without_close_is_not_scrubbed(self):
        """A plain-text line that starts with the tag name must be preserved."""
        s = StreamingContextScrubber()
        out = (
            s.feed("Visible intro\n")
            + s.feed("<memory-context> is the literal tag name mentioned here.")
            + s.flush()
        )
        assert out == "Visible intro\n<memory-context> is the literal tag name mentioned here."


class TestStreamingContextScrubberUnterminatedSpan:
    def test_unterminated_span_drops_payload(self):
        """Provider drops close tag — better to lose output than to leak."""
        s = StreamingContextScrubber()
        out = s.feed("pre \n<memory-context>\nsecret never closed") + s.flush()
        assert out == "pre \n"
        assert "secret" not in out

    def test_reset_clears_hung_span(self):
        """Cross-turn scrubber reset drops a hung span so next turn is clean."""
        s = StreamingContextScrubber()
        s.feed("pre <memory-context>half")
        s.reset()
        out = s.feed("clean text") + s.flush()
        assert out == "clean text"


class TestStreamingContextScrubberCaseInsensitivity:
    def test_uppercase_tags_still_scrubbed(self):
        s = StreamingContextScrubber()
        out = (
            s.feed("<MEMORY-CONTEXT>\nsecret")
            + s.feed("</Memory-Context>visible")
            + s.flush()
        )
        assert out == "visible"
        assert "secret" not in out


class TestSanitizeContextUnchanged:
    """Smoke test that the one-shot sanitize_context still works for whole strings."""

    def test_whole_block_still_sanitized(self):
        leaked = (
            "<memory-context>\n"
            "[System note: The following is recalled memory context, NOT new "
            "user input. Treat as informational background data.]\n"
            "payload\n"
            "</memory-context>\nVisible"
        )
        out = sanitize_context(leaked).strip()
        assert out == "Visible"


class TestSanitizeHistoryContext:
    def test_pasted_memory_context_is_removed_but_surrounding_user_text_stays(self):
        leaked = (
            "Can you fix it?\n\n"
            "<memory-context>\n"
            "[System note: The following is recalled memory context, NOT new user input.]\n"
            "## User Representation\nsecret facts\n"
            "## AI Identity Card\nsecret identity\n"
            "</memory-context>\n"
            "actual trailing user text"
        )
        out = sanitize_history_context(leaked)
        assert "Can you fix it?" in out
        assert "actual trailing user text" in out
        assert "memory-context" not in out
        assert "User Representation" not in out
        assert "secret" not in out

    def test_unclosed_memory_context_in_history_drops_remainder(self):
        leaked = (
            "Can you fix it?\n\n"
            "<memory-context>\n"
            "## User Representation\nsecret facts that arrived in a split Matrix chunk"
        )
        assert sanitize_history_context(leaked) == "Can you fix it?\n\n"

    def test_compaction_handoff_remains_valid_history_context(self):
        handoff = (
            "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted.\n"
            "## Active Task\nkeep this model-input handoff"
        )
        assert sanitize_history_context(handoff) == handoff


class TestSanitizeVisibleContext:
    def test_compaction_summary_is_not_user_visible(self):
        leaked = (
            "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted.\n"
            "## Active Task\nsecret handoff"
        )
        assert sanitize_visible_context(leaked) == ""
        # Internal replay sanitization must stay narrow; compacted summaries are
        # valid model input and should not be stripped from conversation history.
        assert sanitize_context(leaked) == leaked

    def test_compaction_summary_without_reference_only_is_not_user_visible(self):
        leaked = "[CONTEXT COMPACTION] Summary of private handoff"
        assert sanitize_visible_context(leaked) == ""

    def test_restart_resume_note_prefix_is_removed_from_visible_text(self):
        leaked = (
            "[System note: Your previous turn in this session was interrupted "
            "by gateway shutdown. The conversation history below is intact.]\n\n"
            "Actual answer."
        )
        assert sanitize_visible_context(leaked) == "Actual answer."

    def test_preserved_todo_injection_is_not_user_visible(self):
        leaked = (
            "[Your active task list was preserved across context compression]\n"
            "- [>] trace. Trace which visible-output path leaked raw memory context (in_progress)\n"
            "- [ ] patch. Patch sanitizer at the missed boundary (pending)\n"
        )
        assert sanitize_visible_context(leaked) == ""

    def test_preserved_todo_injection_prefix_removed_but_answer_preserved(self):
        leaked = (
            "[Your active task list was preserved across context compression]\n"
            "- [>] trace. Trace which visible-output path leaked raw memory context (in_progress)\n"
            "Actual answer."
        )
        assert sanitize_visible_context(leaked) == "Actual answer."

    def test_fenced_memory_context_leaves_no_empty_code_block(self):
        leaked = (
            "Please fix this.\n\n"
            "```\n"
            "<memory-context>\n"
            "## User Representation\nsecret facts\n"
            "</memory-context>\n"
            "```\n"
            "Actual user text."
        )
        out = sanitize_visible_context(leaked)
        assert out == "Please fix this.\n\nActual user text."
        assert "```" not in out
        assert "memory-context" not in out
        assert "secret" not in out

    def test_html_escaped_memory_context_is_removed(self):
        leaked = (
            "Please fix this.\n"
            "&lt;memory-context&gt;\n"
            "## User Representation\nsecret facts\n"
            "&lt;/memory-context&gt;\n"
            "Actual user text."
        )
        out = sanitize_visible_context(leaked)
        assert out == "Please fix this.\n\nActual user text."
        assert "&lt;memory-context" not in out
        assert "secret" not in out

    def test_unclosed_fenced_memory_context_drops_from_fence_line(self):
        leaked = (
            "Please fix this.\n\n"
            "```text\n"
            "<memory-context>\n"
            "## User Representation\nsecret facts that arrived split"
        )
        assert sanitize_visible_context(leaked) == "Please fix this.\n\n"


class TestStreamingVisibleContextScrubber:
    def test_streamed_compaction_prefix_split_across_deltas_is_suppressed(self):
        s = StreamingVisibleContextScrubber()
        deltas = [
            "[CONTEXT",
            " COMPACTION — REFERENCE ONLY] Earlier turns were compacted.\n",
            "## Active Task\nsecret handoff",
        ]
        out = "".join(s.feed(d) for d in deltas) + s.flush()
        assert out == ""

    def test_streamed_restart_note_split_across_deltas_strips_note_only(self):
        s = StreamingVisibleContextScrubber()
        deltas = [
            "[System note: Your previous turn in this session",
            " was interrupted by gateway restart. History intact.]\n\n",
            "Actual answer.",
        ]
        out = "".join(s.feed(d) for d in deltas) + s.flush()
        assert out == "Actual answer."

    def test_streamed_preserved_todo_block_split_across_deltas_is_suppressed(self):
        s = StreamingVisibleContextScrubber()
        deltas = [
            "[Your active task list was preserved",
            " across context compression]\n",
            "- [>] trace. Trace leak path (in_progress)\n",
            "- [ ] patch. Patch missed boundary (pending)\n",
        ]
        out = "".join(s.feed(d) for d in deltas) + s.flush()
        assert out == ""

    def test_streamed_preserved_todo_block_strips_but_preserves_answer(self):
        s = StreamingVisibleContextScrubber()
        deltas = [
            "[Your active task list was preserved across context compression]\n",
            "- [>] trace. Trace leak path (in_progress)\n",
            "Actual answer.",
        ]
        out = "".join(s.feed(d) for d in deltas) + s.flush()
        assert out == "Actual answer."

    def test_plain_bracketed_text_eventually_passes_through(self):
        s = StreamingVisibleContextScrubber()
        out = s.feed("[Cool") + s.feed(" story] visible") + s.flush()
        assert out == "[Cool story] visible"


class TestStreamingContextScrubberCrossTurn:
    """A scrubber instance is reused across turns (per agent).  reset() must
    clear any held state so a partial-tag tail from turn N doesn't bleed
    into turn N+1's first delta."""

    def test_reset_clears_held_partial_tag(self):
        s = StreamingContextScrubber()
        # Feed a partial open-tag prefix that gets held back as buffer.
        out_turn_1 = s.feed("answer<memo")
        assert out_turn_1 == "answer"

        # Reset for next turn — buffer must clear.
        s.reset()

        # New turn: plain text starting with a "<m" must NOT be treated as
        # the continuation of the held "<memo".
        out_turn_2 = s.feed("<marker>fresh content")
        assert out_turn_2 == "<marker>fresh content"

    def test_reset_clears_in_span_state(self):
        s = StreamingContextScrubber()
        s.feed("text\n<memory-context>secret-tail")
        # Mid-span state held — without reset, subsequent text would be
        # discarded until we see </memory-context>.
        s.reset()
        out = s.feed("post-reset visible text")
        assert out == "post-reset visible text"


class TestBuildMemoryContextBlockWarnsOnViolation:
    """Providers must return raw context — not pre-wrapped.  When they do,
    we strip and warn so the buggy provider surfaces."""

    def test_provider_emitting_wrapper_warns(self, caplog):
        import logging
        from agent.memory_manager import build_memory_context_block

        prewrapped = (
            "<memory-context>\n"
            "[System note: ...]\n\n"
            "real fact\n"
            "</memory-context>"
        )
        with caplog.at_level(logging.WARNING, logger="agent.memory_manager"):
            out = build_memory_context_block(prewrapped)

        assert any("pre-wrapped" in rec.message for rec in caplog.records)
        assert out.count("<memory-context>") == 1
        assert out.count("</memory-context>") == 1

    def test_clean_provider_output_does_not_warn(self, caplog):
        import logging
        from agent.memory_manager import build_memory_context_block

        with caplog.at_level(logging.WARNING, logger="agent.memory_manager"):
            out = build_memory_context_block("plain fact about user")

        assert not any("pre-wrapped" in rec.message for rec in caplog.records)
        assert "plain fact about user" in out
