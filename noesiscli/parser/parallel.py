"""
Parallel Processing Parser Pipeline (Phase 5.1).

Distributes Tree-sitter AST parsing across all available CPU cores using
Python's ``concurrent.futures.ProcessPoolExecutor``, dramatically cutting
repository indexing time on large codebases.

Design notes
------------
* The worker function ``_parse_file_worker`` is a **module-level** function
  (not a bound method) so it is fully picklable by the ``multiprocessing``
  spawn/fork back-end used by ``ProcessPoolExecutor``.
* Each worker constructs its own ``TreeSitterParser`` instance — parser
  objects are not shared across process boundaries.
* Files are submitted in configurable batch chunks to amortise IPC overhead
  while keeping memory pressure bounded.
* The number of worker processes defaults to ``os.cpu_count()`` (or 1 if
  that returns *None*).

Data Flow
---------
  Input  : List[str]  — absolute paths to Python source files (from Phase 1.1)
  Output : List[dict] — aggregated structured Code Chunk dictionaries
                        (same schema as TreeSitterParser.parse_file output),
                        ready for Phase 4 (SymbolTable / DependencyGraph),
                        Phase 1.3 (Embedding), Phase 3.1 (BM25).
"""

from __future__ import annotations

import os
import math
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level worker — must live at module scope for pickling
# ---------------------------------------------------------------------------

def _parse_file_worker(file_path: str) -> list[dict]:
    """
    Worker function executed in a child process.

    Imports and instantiates ``TreeSitterParser`` locally so that the
    heavy tree-sitter Language object is not pickled across the process
    boundary.  Returns the list of structured Code Chunk dicts for the
    given *file_path*, or an empty list on any error.
    """
    try:
        # Local import ensures no shared state with the parent process
        from noesiscli.parser.tree_sitter_parser import TreeSitterParser
        parser = TreeSitterParser(language="python")
        return parser.parse_file(file_path)
    except Exception as exc:  # noqa: BLE001
        # Log but never propagate — a single bad file should not abort the run
        logger.warning("Failed to parse %s: %s", file_path, exc)
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ParallelParserPipeline:
    """
    Multiprocessing-based parser pipeline for Phase 5.1.

    Distributes ``TreeSitterParser.parse_file`` calls across multiple CPU
    cores using ``ProcessPoolExecutor``, collects results, and returns a
    flat aggregated list of Code Chunk dicts.

    Parameters
    ----------
    max_workers : int | None
        Number of worker processes.  Defaults to ``os.cpu_count()`` (or 1).
    chunk_batch_size : int
        Number of files dispatched to each worker at a time.  Larger batches
        reduce IPC round-trips but may increase peak memory.  Defaults to 10.
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        chunk_batch_size: int = 10,
    ) -> None:
        self.max_workers: int = max_workers or (os.cpu_count() or 1)
        self.chunk_batch_size: int = max(1, chunk_batch_size)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def parse_files(
        self,
        file_paths: list[str],
        *,
        progress_callback=None,
    ) -> list[dict]:
        """
        Parse *file_paths* in parallel and return the aggregated chunk list.

        Parameters
        ----------
        file_paths : list[str]
            Absolute paths to Python source files (output of Phase 1.1).
        progress_callback : callable | None
            Optional ``(completed: int, total: int) -> None`` callable invoked
            after each file completes.  Useful for driving progress bars in the
            CLI layer.

        Returns
        -------
        list[dict]
            Flat list of structured Code Chunk dictionaries, preserving
            relative file order within each file's chunks while respecting
            the non-deterministic completion order across workers.
        """
        if not file_paths:
            return []

        total = len(file_paths)
        # Use a single process for tiny repos to avoid fork/spawn overhead
        effective_workers = min(self.max_workers, total)

        logger.info(
            "ParallelParserPipeline: parsing %d file(s) with %d worker(s)",
            total,
            effective_workers,
        )

        # Map future → original file_path for ordered result collection
        future_to_path: dict = {}
        # Collect per-file results keyed by file_path to reassemble in order
        per_file_results: dict[str, list[dict]] = {}

        completed_count = 0

        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            # Submit all files up-front — the executor manages the queue
            for path in file_paths:
                future = executor.submit(_parse_file_worker, path)
                future_to_path[future] = path

            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    chunks = future.result()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Worker raised exception for %s: %s", path, exc
                    )
                    chunks = []

                per_file_results[path] = chunks
                completed_count += 1

                if progress_callback is not None:
                    try:
                        progress_callback(completed_count, total)
                    except Exception:  # noqa: BLE001
                        pass  # Never let a progress callback abort the pipeline

        # Reassemble in the original file order so output is deterministic
        all_chunks: list[dict] = []
        for path in file_paths:
            all_chunks.extend(per_file_results.get(path, []))

        logger.info(
            "ParallelParserPipeline: produced %d chunk(s) from %d file(s)",
            len(all_chunks),
            total,
        )
        return all_chunks

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @staticmethod
    def worker_count() -> int:
        """Return the number of logical CPU cores available (minimum 1)."""
        return os.cpu_count() or 1

    def __repr__(self) -> str:
        return (
            f"ParallelParserPipeline("
            f"max_workers={self.max_workers}, "
            f"chunk_batch_size={self.chunk_batch_size})"
        )
