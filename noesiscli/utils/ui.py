"""
UI & Terminal Utilities — Phase 7.3

Provides:

  stream_response(token_generator, title)
      Renders streamed LLM response tokens inside a Rich Live panel with
      Markdown formatting.  Tokens are accumulated in real-time and the
      panel is refreshed after every token so the user sees a live,
      progressively-rendered markdown document rather than raw text.

  make_progress(**columns)
      Factory that returns a pre-configured Rich Progress instance using
      the project's standard column layout (spinner, description, bar,
      M-of-N count, elapsed time).  Used for both parsing (Phase 5.1) and
      embedding (Phase 1.3) steps.

  embedding_progress(total_chunks)
      Context-manager wrapper around make_progress() tailored for the
      Voyage AI embedding step.  Yields a (progress, task_id) pair so the
      caller can update it as batches complete.

All functions degrade gracefully when `rich` is not installed — falling
back to plain print() output so the CLI remains functional in minimal
environments.
"""

from __future__ import annotations

from typing import Generator, Iterator, Tuple

# ---------------------------------------------------------------------------
# Rich imports with graceful fallback
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.progress import (
        Progress,
        SpinnerColumn,
        BarColumn,
        TextColumn,
        TimeElapsedColumn,
        MofNCompleteColumn,
        TaskID,
    )
    from rich.text import Text
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


# Shared console instance (stderr=False → stdout)
_console = Console() if _RICH_AVAILABLE else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def stream_response(
    token_generator: Iterator[str],
    title: str = "Response",
) -> str:
    """
    Stream LLM response tokens to the terminal with Rich Markdown rendering.

    Tokens are accumulated into a growing string and the terminal panel is
    refreshed after every token using ``rich.live.Live``.  The final
    accumulated response is returned as a plain string so callers can store
    it in the workflow state.

    When ``rich`` is not available the function falls back to printing each
    token directly to stdout (identical to the previous raw behaviour).

    Args:
        token_generator: Any iterator that yields ``str`` tokens.
                         Typically the return value of ``GeminiClient.stream()``.
        title:           Panel title shown above the markdown output.

    Returns:
        The full response string (all tokens concatenated).
    """
    accumulated = ""

    if not _RICH_AVAILABLE:
        # Plain fallback — identical to the old sys.stdout.write() loop
        import sys
        for token in token_generator:
            sys.stdout.write(token)
            sys.stdout.flush()
            accumulated += token
        print()
        return accumulated

    # Rich path — live-rendered Markdown panel
    with Live(
        _render_panel(accumulated, title),
        console=_console,
        refresh_per_second=15,
        vertical_overflow="visible",
    ) as live:
        for token in token_generator:
            accumulated += token
            live.update(_render_panel(accumulated, title))

    return accumulated


def make_progress(**extra_columns) -> "Progress":
    """
    Return a pre-configured Rich Progress instance with the project's
    standard column layout:

        [spinner]  description  [bar]  M/N  elapsed

    Args:
        **extra_columns: Ignored (accepted for forward-compatibility).

    Returns:
        A :class:`rich.progress.Progress` instance (not yet started).
        Use it as a context manager: ``with make_progress() as p: ...``

    Falls back to a no-op stub when ``rich`` is not installed.
    """
    if not _RICH_AVAILABLE:
        return _PlainProgress()

    return Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=_console,
    )


def embedding_progress(total_chunks: int):
    """
    Context manager for the Voyage AI embedding step (Phase 1.3).

    Yields a ``(progress, task_id)`` pair.  The caller increments
    ``task_id`` as each batch completes:

    .. code-block:: python

        with embedding_progress(len(chunks)) as (prog, task):
            for batch in batches:
                embeddings.extend(embed(batch))
                prog.advance(task, len(batch))

    Args:
        total_chunks: Total number of chunks to embed (used to set the
                      progress bar maximum).

    Yields:
        ``(Progress, TaskID)`` — the progress instance and the task handle.
    """
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        prog = make_progress()
        with prog:
            task = prog.add_task(
                f"Embedding {total_chunks} chunk(s) via Voyage AI",
                total=total_chunks,
            )
            yield prog, task

    return _ctx()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _render_panel(content: str, title: str) -> "Panel":
    """Render the accumulated response string as a Rich Markdown panel."""
    md = Markdown(content) if content.strip() else Text("")
    return Panel(md, title=f"[bold green]{title}[/bold green]", border_style="green")


# ---------------------------------------------------------------------------
# Fallback stubs (when rich is not installed)
# ---------------------------------------------------------------------------

class _PlainProgress:
    """
    Minimal no-op drop-in for ``rich.progress.Progress`` used when
    ``rich`` is not installed.  Supports the context-manager protocol
    and the ``add_task`` / ``advance`` / ``update`` methods so callers
    need no conditional logic.
    """

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def add_task(self, description: str, total: int = 100) -> int:
        print(f"  {description}…")
        self._total = total
        self._completed = 0
        return 0  # task_id stub

    def advance(self, task_id: int, amount: int = 1) -> None:
        self._completed += amount
        print(
            f"\r  {self._completed}/{self._total}",
            end="",
            flush=True,
        )

    def update(self, task_id: int, completed: int = None, **kwargs) -> None:
        if completed is not None:
            self._completed = completed
            print(
                f"\r  {self._completed}/{self._total}",
                end="",
                flush=True,
            )
