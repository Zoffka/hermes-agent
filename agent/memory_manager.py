"""MemoryManager — orchestrates memory providers for the agent.

Single integration point in run_agent.py. Replaces scattered per-backend
code with one manager that delegates to registered providers.

Only ONE external plugin provider is allowed at a time — attempting to
register a second external provider is rejected with a warning.  This
prevents tool schema bloat and conflicting memory backends.

Usage in run_agent.py:
    self._memory_manager = MemoryManager()
    # Only ONE of these:
    self._memory_manager.add_provider(plugin_provider)

    # System prompt
    prompt_parts.append(self._memory_manager.build_system_prompt())

    # Pre-turn
    context = self._memory_manager.prefetch_all(user_message)

    # Post-turn
    self._memory_manager.sync_all(user_msg, assistant_response)
    self._memory_manager.queue_prefetch_all(user_msg)
"""

from __future__ import annotations

import logging
import re
import inspect
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context fencing helpers
# ---------------------------------------------------------------------------

_FENCE_TAG_RE = re.compile(r'</?\s*memory-context\s*>', re.IGNORECASE)
_ESCAPED_FENCE_TAG_RE = re.compile(
    r'&lt;\s*(/?)\s*(memory-context|prior_memory_file)\s*&gt;',
    re.IGNORECASE,
)
_INLINE_CODE_FENCE_TAG_RE = re.compile(
    r'`(<\s*/?\s*(?:memory-context|prior_memory_file)\s*>)`',
    re.IGNORECASE,
)
_EMPTY_CODE_FENCE_RE = re.compile(
    r'(?m)^[ \t]*```[^\n]*\n[ \t\r\n]*?^[ \t]*```[ \t]*(?:\n|$)'
)
_INTERNAL_CONTEXT_RE = re.compile(
    r'<\s*memory-context\s*>[\s\S]*?</\s*memory-context\s*>',
    re.IGNORECASE,
)
_PRIOR_MEMORY_FILE_RE = re.compile(
    r'<\s*prior_memory_file\s*>[\s\S]*?</\s*prior_memory_file\s*>',
    re.IGNORECASE,
)
_INTERNAL_NOTE_RE = re.compile(
    r'\[System note:\s*The following is recalled memory context,\s*NOT new user input\.\s*Treat as (?:informational background data|authoritative reference data[^\]]*)\.\]\s*',
    re.IGNORECASE,
)
_VISIBLE_CONTEXT_COMPACTION_PREFIX_RE = re.compile(
    r'^\s*\[CONTEXT COMPACTION\b[^\]]*\]',
    re.IGNORECASE,
)
_VISIBLE_RESTART_NOTE_RE = re.compile(
    r'^\s*\[System note:\s*Your previous turn in this session was interrupted[^\]]*\]\s*',
    re.IGNORECASE | re.DOTALL,
)
_VISIBLE_TODO_PRESERVED_RE = re.compile(
    r'^\s*\[Your active task list was preserved across context compression\]\s*\n?'
    r'(?:\s*- \[[^\]]+\] [^\n]*(?:\n|$))*',
    re.IGNORECASE,
)


def _normalize_internal_context_markers(text: str) -> str:
    """Canonicalize internal-context tags before regex stripping.

    Matrix clients and markdown renderers can preserve a leaked block as HTML
    entities (``&lt;memory-context&gt;``) or wrap just the tag in inline code.
    Convert only these known marker forms so the existing scrubbers catch the
    block without globally unescaping user text.
    """
    text = _ESCAPED_FENCE_TAG_RE.sub(lambda m: f"<{'/' if m.group(1) else ''}{m.group(2).lower()}>", text)
    text = _INLINE_CODE_FENCE_TAG_RE.sub(r'\1', text)
    return text


def _cleanup_empty_internal_fences(text: str) -> str:
    """Remove empty markdown fences left after stripping a fenced leak."""
    return _EMPTY_CODE_FENCE_RE.sub('', text)


def sanitize_context(text: str) -> str:
    """Strip fence tags, injected context blocks, and system notes from provider output."""
    text = _normalize_internal_context_markers(text)
    text = _INTERNAL_CONTEXT_RE.sub('', text)
    text = _INTERNAL_NOTE_RE.sub('', text)
    text = _FENCE_TAG_RE.sub('', text)
    return _cleanup_empty_internal_fences(text)


def sanitize_history_context(text: str) -> str:
    """Strip leaked memory fences from persisted/user history before model replay.

    This is stricter than ``sanitize_context`` for memory/prior-memory fences:
    if a client split or stored an opening tag without the close tag, drop the
    rest of that message rather than replaying private memory payload as user
    text.  It is deliberately narrower than ``sanitize_visible_context`` so
    context-compaction handoff summaries remain valid model input when a session
    is resumed.
    """
    text = _normalize_internal_context_markers(text)
    text = _INTERNAL_CONTEXT_RE.sub('', text)
    text = _PRIOR_MEMORY_FILE_RE.sub('', text)
    text = re.sub(
        r'(?im)^[ \t]*(?:```[^\n]*\n)?[ \t]*<\s*(?:memory-context|prior_memory_file)\s*>[\s\S]*$',
        '',
        text,
    )
    text = _INTERNAL_NOTE_RE.sub('', text)
    text = _FENCE_TAG_RE.sub('', text)
    return _cleanup_empty_internal_fences(text)


def sanitize_visible_context(text: str) -> str:
    """Strip internal context that must never be sent to chat surfaces.

    ``sanitize_context`` is deliberately narrow because compacted summaries are
    valid *model input* when replaying a long session. Chat delivery has a
    stricter boundary: if a model parrots the context-compaction handoff or the
    restart-resume system note, drop it instead of showing sysadmin underwear to
    the user.

    Matrix and other clients can split very long pasted messages across events.
    One chunk may contain ``<memory-context>`` without the closing tag; at a
    visible/inbound boundary the safe behaviour is to drop everything from the
    opening tag onward rather than remove only the tag and leak the payload.
    """
    text = _normalize_internal_context_markers(text)
    # First remove complete fenced spans.
    text = _INTERNAL_CONTEXT_RE.sub('', text)
    text = _PRIOR_MEMORY_FILE_RE.sub('', text)
    # At visible boundaries, treat an unterminated opening fence as "drop the
    # remainder". This catches Matrix/client split chunks before batching sees
    # the matching close tag. Include an optional preceding markdown code fence
    # because old leaks were often displayed/quoted as fenced blocks.
    text = re.sub(
        r'(?im)^[ \t]*(?:```[^\n]*\n)?[ \t]*<\s*(?:memory-context|prior_memory_file)\s*>[\s\S]*$',
        '',
        text,
    )
    text = _INTERNAL_NOTE_RE.sub('', text)
    text = _FENCE_TAG_RE.sub('', text)
    text = _cleanup_empty_internal_fences(text)
    if _VISIBLE_CONTEXT_COMPACTION_PREFIX_RE.match(text):
        return ""
    text = _VISIBLE_RESTART_NOTE_RE.sub('', text)
    text = _VISIBLE_TODO_PRESERVED_RE.sub('', text)
    return _cleanup_empty_internal_fences(text)


class StreamingContextScrubber:
    """Stateful scrubber for streaming text that may contain split memory-context spans.

    The one-shot ``sanitize_context`` regex cannot survive chunk boundaries:
    a ``<memory-context>`` opened in one delta and closed in a later delta
    leaks its payload to the UI because the non-greedy block regex needs
    both tags in one string.  This scrubber runs a small state machine
    across deltas, holding back partial-tag tails and discarding
    everything inside a span (including the system-note line).

    Usage::

        scrubber = StreamingContextScrubber()
        for delta in stream:
            visible = scrubber.feed(delta)
            if visible:
                emit(visible)
        trailing = scrubber.flush()  # at end of stream
        if trailing:
            emit(trailing)

    The scrubber is re-entrant per agent instance.  Callers building new
    top-level responses (new turn) should create a fresh scrubber or call
    ``reset()``.
    """

    _OPEN_TAG = "<memory-context>"
    _CLOSE_TAG = "</memory-context>"

    def __init__(self) -> None:
        self._in_span: bool = False
        self._buf: str = ""
        self._at_block_boundary: bool = True

    def reset(self) -> None:
        self._in_span = False
        self._buf = ""
        self._at_block_boundary = True

    def feed(self, text: str) -> str:
        """Return the visible portion of ``text`` after scrubbing.

        Any trailing fragment that could be the start of an open/close tag
        is held back in the internal buffer and surfaced on the next
        ``feed()`` call or discarded/emitted by ``flush()``.
        """
        if not text:
            return ""
        buf = self._buf + text
        self._buf = ""
        out: list[str] = []

        while buf:
            if self._in_span:
                idx = buf.lower().find(self._CLOSE_TAG)
                if idx == -1:
                    # Hold back a potential partial close tag; drop the rest
                    held = self._max_partial_suffix(buf, self._CLOSE_TAG)
                    self._buf = buf[-held:] if held else ""
                    return "".join(out)
                # Found close — skip span content + tag, continue
                buf = buf[idx + len(self._CLOSE_TAG):]
                self._in_span = False
            else:
                idx = self._find_boundary_open_tag(buf)
                if idx == -1:
                    # No open tag — hold back a potential partial open tag
                    held = (
                        self._max_pending_open_suffix(buf)
                        or self._max_partial_suffix(buf, self._OPEN_TAG)
                    )
                    if held:
                        self._append_visible(out, buf[:-held])
                        self._buf = buf[-held:]
                    else:
                        self._append_visible(out, buf)
                    return "".join(out)
                # Emit text before the tag, enter span
                if idx > 0:
                    self._append_visible(out, buf[:idx])
                buf = buf[idx + len(self._OPEN_TAG):]
                self._in_span = True

        return "".join(out)

    def flush(self) -> str:
        """Emit any held-back buffer at end-of-stream.

        If we're still inside an unterminated span the remaining content is
        discarded (safer: leaking partial memory context is worse than a
        truncated answer).  Otherwise the held-back partial-tag tail is
        emitted verbatim (it turned out not to be a real tag).
        """
        if self._in_span:
            self._buf = ""
            self._in_span = False
            return ""
        tail = self._buf
        self._buf = ""
        return tail

    @staticmethod
    def _max_partial_suffix(buf: str, tag: str) -> int:
        """Return the length of the longest buf-suffix that is a tag-prefix.

        Case-insensitive.  Returns 0 if no suffix could start the tag.
        """
        tag_lower = tag.lower()
        buf_lower = buf.lower()
        max_check = min(len(buf_lower), len(tag_lower) - 1)
        for i in range(max_check, 0, -1):
            if tag_lower.startswith(buf_lower[-i:]):
                return i
        return 0

    def _find_boundary_open_tag(self, buf: str) -> int:
        """Find an opening fence only when it starts a block-like span."""
        buf_lower = buf.lower()
        search_start = 0
        while True:
            idx = buf_lower.find(self._OPEN_TAG, search_start)
            if idx == -1:
                return -1
            if self._is_block_boundary(buf, idx) and self._has_block_opener_suffix(buf, idx):
                return idx
            search_start = idx + 1

    def _max_pending_open_suffix(self, buf: str) -> int:
        """Hold a complete boundary tag until the following char confirms it."""
        if not buf.lower().endswith(self._OPEN_TAG):
            return 0
        idx = len(buf) - len(self._OPEN_TAG)
        if not self._is_block_boundary(buf, idx):
            return 0
        return len(self._OPEN_TAG)

    def _has_block_opener_suffix(self, buf: str, idx: int) -> bool:
        after_idx = idx + len(self._OPEN_TAG)
        if after_idx >= len(buf):
            return False
        return buf[after_idx] in "\r\n"

    def _is_block_boundary(self, buf: str, idx: int) -> bool:
        if idx == 0:
            return self._at_block_boundary
        preceding = buf[:idx]
        last_newline = preceding.rfind("\n")
        if last_newline == -1:
            return self._at_block_boundary and preceding.strip() == ""
        return preceding[last_newline + 1:].strip() == ""

    def _append_visible(self, out: list[str], text: str) -> None:
        if not text:
            return
        out.append(text)
        self._update_block_boundary(text)

    def _update_block_boundary(self, text: str) -> None:
        last_newline = text.rfind("\n")
        if last_newline != -1:
            self._at_block_boundary = text[last_newline + 1:].strip() == ""
        else:
            self._at_block_boundary = self._at_block_boundary and text.strip() == ""


class StreamingVisibleContextScrubber:
    """Streaming counterpart to :func:`sanitize_visible_context`.

    ``StreamingContextScrubber`` removes fenced ``<memory-context>`` spans
    wherever they appear. Visible-only scaffolding such as context-compaction
    handoffs and restart-resume notes is different: those markers are anchored
    at the start of a *response*. A per-delta regex would leak the first chunk
    (``"[CONTEXT"``) before the full prefix arrives, so this scrubber buffers
    only the response prefix until it can prove the text is safe or internal.
    """

    _COMPACTION_PREFIX = "[context compaction"
    _RESTART_PREFIX = "[system note: your previous turn in this session was interrupted"
    _TODO_PREFIX = "[your active task list was preserved across context compression]"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._buf = ""
        self._decided = False
        self._suppress_all = False
        self._context_scrubber = StreamingContextScrubber()

    def feed(self, text: str) -> str:
        if not text or self._suppress_all:
            return ""
        text = self._context_scrubber.feed(text)
        if not text:
            return ""
        if self._decided:
            return sanitize_visible_context(text)

        self._buf += text
        probe = self._buf.lstrip()
        probe_lower = probe.lower()

        if not probe:
            # Leading whitespace only — hold it until we know whether an
            # internal marker follows.
            return ""

        if self._COMPACTION_PREFIX.startswith(probe_lower):
            return ""
        if probe_lower.startswith(self._COMPACTION_PREFIX):
            # A compaction handoff is never user-visible; drop this and every
            # later delta for the response.
            self._buf = ""
            self._suppress_all = True
            return ""

        if self._RESTART_PREFIX.startswith(probe_lower):
            return ""
        if probe_lower.startswith(self._RESTART_PREFIX):
            close_idx = probe.find("]")
            if close_idx == -1:
                return ""
            # Preserve any text after the system note. Leading whitespace that
            # preceded the note is intentionally discarded with the note.
            rest = probe[close_idx + 1 :]
            self._buf = ""
            self._decided = True
            return sanitize_visible_context(rest).lstrip("\n")

        if self._TODO_PREFIX.startswith(probe_lower):
            return ""
        if probe_lower.startswith(self._TODO_PREFIX):
            stripped = sanitize_visible_context(probe)
            if not stripped:
                # Hold the whole preserved-task block until flush, or until a
                # non-task answer line arrives after it.
                return ""
            self._buf = ""
            self._decided = True
            return stripped.lstrip("\n")

        # The prefix is ordinary user-visible text.
        out = self._buf
        self._buf = ""
        self._decided = True
        return sanitize_visible_context(out)

    def flush(self) -> str:
        if self._suppress_all:
            self._buf = ""
            self._decided = True
            return ""
        tail_from_context = self._context_scrubber.flush()
        if tail_from_context:
            self._buf += tail_from_context
        tail = self._buf
        self._buf = ""
        self._decided = True
        return sanitize_visible_context(tail)


def build_memory_context_block(raw_context: str) -> str:
    """Wrap prefetched memory in a fenced block with system note."""
    if not raw_context or not raw_context.strip():
        return ""
    clean = sanitize_context(raw_context)
    if clean != raw_context:
        logger.warning("memory provider returned pre-wrapped context; stripped")
    return (
        "<memory-context>\n"
        "[System note: The following is recalled memory context, "
        "NOT new user input. Treat as authoritative reference data — "
        "this is the agent's persistent memory and should inform all responses.]\n\n"
        f"{clean}\n"
        "</memory-context>"
    )


class MemoryManager:
    """Orchestrates the built-in provider plus at most one external provider.

    The builtin provider is always first. Only one non-builtin (external)
    provider is allowed.  Failures in one provider never block the other.
    """

    def __init__(self) -> None:
        self._providers: List[MemoryProvider] = []
        self._tool_to_provider: Dict[str, MemoryProvider] = {}
        self._has_external: bool = False  # True once a non-builtin provider is added

    # -- Registration --------------------------------------------------------

    def add_provider(self, provider: MemoryProvider) -> None:
        """Register a memory provider.

        Built-in provider (name ``"builtin"``) is always accepted.
        Only **one** external (non-builtin) provider is allowed — a second
        attempt is rejected with a warning.
        """
        is_builtin = provider.name == "builtin"

        if not is_builtin:
            if self._has_external:
                existing = next(
                    (p.name for p in self._providers if p.name != "builtin"), "unknown"
                )
                logger.warning(
                    "Rejected memory provider '%s' — external provider '%s' is "
                    "already registered. Only one external memory provider is "
                    "allowed at a time. Configure which one via memory.provider "
                    "in config.yaml.",
                    provider.name, existing,
                )
                return
            self._has_external = True

        self._providers.append(provider)

        # Index tool names → provider for routing
        for schema in provider.get_tool_schemas():
            tool_name = schema.get("name", "")
            if tool_name and tool_name not in self._tool_to_provider:
                self._tool_to_provider[tool_name] = provider
            elif tool_name in self._tool_to_provider:
                logger.warning(
                    "Memory tool name conflict: '%s' already registered by %s, "
                    "ignoring from %s",
                    tool_name,
                    self._tool_to_provider[tool_name].name,
                    provider.name,
                )

        logger.info(
            "Memory provider '%s' registered (%d tools)",
            provider.name,
            len(provider.get_tool_schemas()),
        )

    @property
    def providers(self) -> List[MemoryProvider]:
        """All registered providers in order."""
        return list(self._providers)

    def get_provider(self, name: str) -> Optional[MemoryProvider]:
        """Get a provider by name, or None if not registered."""
        for p in self._providers:
            if p.name == name:
                return p
        return None

    # -- System prompt -------------------------------------------------------

    def build_system_prompt(self) -> str:
        """Collect system prompt blocks from all providers.

        Returns combined text, or empty string if no providers contribute.
        Each non-empty block is labeled with the provider name.
        """
        blocks = []
        for provider in self._providers:
            try:
                block = provider.system_prompt_block()
                if block and block.strip():
                    blocks.append(block)
            except Exception as e:
                logger.warning(
                    "Memory provider '%s' system_prompt_block() failed: %s",
                    provider.name, e,
                )
        return "\n\n".join(blocks)

    # -- Prefetch / recall ---------------------------------------------------

    def prefetch_all(self, query: str, *, session_id: str = "") -> str:
        """Collect prefetch context from all providers.

        Returns merged context text labeled by provider. Empty providers
        are skipped. Failures in one provider don't block others.
        """
        parts = []
        for provider in self._providers:
            try:
                result = provider.prefetch(query, session_id=session_id)
                if result and result.strip():
                    parts.append(result)
            except Exception as e:
                logger.debug(
                    "Memory provider '%s' prefetch failed (non-fatal): %s",
                    provider.name, e,
                )
        return "\n\n".join(parts)

    def queue_prefetch_all(self, query: str, *, session_id: str = "") -> None:
        """Queue background prefetch on all providers for the next turn."""
        for provider in self._providers:
            try:
                provider.queue_prefetch(query, session_id=session_id)
            except Exception as e:
                logger.debug(
                    "Memory provider '%s' queue_prefetch failed (non-fatal): %s",
                    provider.name, e,
                )

    # -- Sync ----------------------------------------------------------------

    def sync_all(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Sync a completed turn to all providers."""
        for provider in self._providers:
            try:
                provider.sync_turn(user_content, assistant_content, session_id=session_id)
            except Exception as e:
                logger.warning(
                    "Memory provider '%s' sync_turn failed: %s",
                    provider.name, e,
                )

    # -- Tools ---------------------------------------------------------------

    def get_all_tool_schemas(self) -> List[Dict[str, Any]]:
        """Collect tool schemas from all providers."""
        schemas = []
        seen = set()
        for provider in self._providers:
            try:
                for schema in provider.get_tool_schemas():
                    name = schema.get("name", "")
                    if name and name not in seen:
                        schemas.append(schema)
                        seen.add(name)
            except Exception as e:
                logger.warning(
                    "Memory provider '%s' get_tool_schemas() failed: %s",
                    provider.name, e,
                )
        return schemas

    def get_all_tool_names(self) -> set:
        """Return set of all tool names across all providers."""
        return set(self._tool_to_provider.keys())

    def has_tool(self, tool_name: str) -> bool:
        """Check if any provider handles this tool."""
        return tool_name in self._tool_to_provider

    def handle_tool_call(
        self, tool_name: str, args: Dict[str, Any], **kwargs
    ) -> str:
        """Route a tool call to the correct provider.

        Returns JSON string result. Raises ValueError if no provider
        handles the tool.
        """
        provider = self._tool_to_provider.get(tool_name)
        if provider is None:
            return tool_error(f"No memory provider handles tool '{tool_name}'")
        try:
            return provider.handle_tool_call(tool_name, args, **kwargs)
        except Exception as e:
            logger.error(
                "Memory provider '%s' handle_tool_call(%s) failed: %s",
                provider.name, tool_name, e,
            )
            return tool_error(f"Memory tool '{tool_name}' failed: {e}")

    # -- Lifecycle hooks -----------------------------------------------------

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """Notify all providers of a new turn.

        kwargs may include: remaining_tokens, model, platform, tool_count.
        """
        for provider in self._providers:
            try:
                provider.on_turn_start(turn_number, message, **kwargs)
            except Exception as e:
                logger.debug(
                    "Memory provider '%s' on_turn_start failed: %s",
                    provider.name, e,
                )

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Notify all providers of session end."""
        for provider in self._providers:
            try:
                provider.on_session_end(messages)
            except Exception as e:
                logger.debug(
                    "Memory provider '%s' on_session_end failed: %s",
                    provider.name, e,
                )

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        **kwargs,
    ) -> None:
        """Notify all providers that the agent's session_id has rotated.

        Fires on ``/resume``, ``/branch``, ``/reset``, ``/new``, and
        context compression — any path that reassigns
        ``AIAgent.session_id`` without tearing the provider down.

        Providers keep running; they only need to refresh cached
        per-session state so subsequent writes land in the correct
        session's record. See ``MemoryProvider.on_session_switch`` for
        the full contract.
        """
        if not new_session_id:
            return
        for provider in self._providers:
            try:
                provider.on_session_switch(
                    new_session_id,
                    parent_session_id=parent_session_id,
                    reset=reset,
                    **kwargs,
                )
            except Exception as e:
                logger.debug(
                    "Memory provider '%s' on_session_switch failed: %s",
                    provider.name, e,
                )

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Notify all providers before context compression.

        Returns combined text from providers to include in the compression
        summary prompt. Empty string if no provider contributes.
        """
        parts = []
        for provider in self._providers:
            try:
                result = provider.on_pre_compress(messages)
                if result and result.strip():
                    parts.append(result)
            except Exception as e:
                logger.debug(
                    "Memory provider '%s' on_pre_compress failed: %s",
                    provider.name, e,
                )
        return "\n\n".join(parts)

    @staticmethod
    def _provider_memory_write_metadata_mode(provider: MemoryProvider) -> str:
        """Return how to pass metadata to a provider's memory-write hook."""
        try:
            signature = inspect.signature(provider.on_memory_write)
        except (TypeError, ValueError):
            return "keyword"

        params = list(signature.parameters.values())
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
            return "keyword"
        if "metadata" in signature.parameters:
            return "keyword"

        accepted = [
            p for p in params
            if p.kind in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        ]
        if len(accepted) >= 4:
            return "positional"
        return "legacy"

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Notify external providers when the built-in memory tool writes.

        Skips the builtin provider itself (it's the source of the write).
        """
        for provider in self._providers:
            if provider.name == "builtin":
                continue
            try:
                metadata_mode = self._provider_memory_write_metadata_mode(provider)
                if metadata_mode == "keyword":
                    provider.on_memory_write(
                        action, target, content, metadata=dict(metadata or {})
                    )
                elif metadata_mode == "positional":
                    provider.on_memory_write(action, target, content, dict(metadata or {}))
                else:
                    provider.on_memory_write(action, target, content)
            except Exception as e:
                logger.debug(
                    "Memory provider '%s' on_memory_write failed: %s",
                    provider.name, e,
                )

    def on_delegation(self, task: str, result: str, *,
                      child_session_id: str = "", **kwargs) -> None:
        """Notify all providers that a subagent completed."""
        for provider in self._providers:
            try:
                provider.on_delegation(
                    task, result, child_session_id=child_session_id, **kwargs
                )
            except Exception as e:
                logger.debug(
                    "Memory provider '%s' on_delegation failed: %s",
                    provider.name, e,
                )

    def shutdown_all(self) -> None:
        """Shut down all providers (reverse order for clean teardown)."""
        for provider in reversed(self._providers):
            try:
                provider.shutdown()
            except Exception as e:
                logger.warning(
                    "Memory provider '%s' shutdown failed: %s",
                    provider.name, e,
                )

    def initialize_all(self, session_id: str, **kwargs) -> None:
        """Initialize all providers.

        Automatically injects ``hermes_home`` into *kwargs* so that every
        provider can resolve profile-scoped storage paths without importing
        ``get_hermes_home()`` themselves.
        """
        if "hermes_home" not in kwargs:
            from hermes_constants import get_hermes_home
            kwargs["hermes_home"] = str(get_hermes_home())
        for provider in self._providers:
            try:
                provider.initialize(session_id=session_id, **kwargs)
            except Exception as e:
                logger.warning(
                    "Memory provider '%s' initialize failed: %s",
                    provider.name, e,
                )
