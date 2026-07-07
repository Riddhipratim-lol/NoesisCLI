# NoesisCLI — Local Codebase Architect

## Overview

NoesisCLI is an AI-powered Local Codebase Architect designed to help developers understand large software repositories without manually reading thousands of lines of source code.

Traditional Retrieval-Augmented Generation (RAG) systems treat source code as plain text and split it into arbitrary character chunks. This often breaks functions, loses semantic boundaries, and retrieves incomplete logic, resulting in inaccurate responses.

NoesisCLI solves this problem by performing syntax-aware indexing using Abstract Syntax Tree (AST) parsing with Tree-sitter. Instead of chunking code by character count, it understands programming language structure (supporting Python, JavaScript, TypeScript, Go, Java, and C++) and indexes complete functions, classes, methods, interfaces, and modules. To optimize ingestion performance, parsing is distributed in parallel across multiple CPU cores.

During indexing, the system constructs a global Symbol Table and a directed Dependency Graph to track imports, inheritance, and function call chains. It enriches chunks lacking documentation with AI-generated summaries and structured metadata, generating embeddings in batches using the Voyage AI API with the `voyage-code-3` model.

Before executing any RAG pipeline components, NoesisCLI performs query validation to ensure the prompt is programming-related. Once validated, the query router determines whether the query requires repository analysis. General programming questions are answered directly, while repository-specific queries are routed through the RAG pipeline, avoiding unnecessary retrieval, reducing latency, and saving computational resources.

For repository-aware queries, NoesisCLI executes parallel semantic search (using ChromaDB dense vectors) and lexical search (using a BM25 index), merging the results using Reciprocal Rank Fusion (RRF). Rather than sending raw files, the system uses the Symbol Table and Dependency Graph to surgically prune non-essential code bodies into signatures and placeholders, presenting a context-minimized skeletal file structure to the model.

To ensure robustness and low latency, the system utilizes a multi-model architecture. It leverages Gemini 3.5 Flash for complex reasoning and summarization tasks, and Gemini 3.1 Flash-Lite for rapid validation, routing, and fallback support (automatically falling back if rate limits or API failures occur on the primary model). Finally, explanations are streamed directly to the terminal in real-time.

The system uses local database storage and hybrid retrieval to maintain fast performance, combined with API-based embedding and LLM models.

---

# Problem Statement

Modern software repositories often contain:

- Thousands of source files
- Hundreds of classes
- Complex dependency chains
- Multiple contributors
- Poor or outdated documentation

Developers joining a new team frequently spend days understanding project architecture before becoming productive.

General-purpose LLMs cannot efficiently analyze large repositories because:

- Entire repositories exceed context windows.
- Character-based chunking destroys code structure.
- Large context increases latency and token cost.
- Keyword search misses semantically related code.
- Vector search alone sometimes ignores exact symbol matches.

Additionally, not every developer query actually requires repository retrieval. Sending every question through a RAG pipeline wastes computational resources and increases response time for simple programming questions.

NoesisCLI addresses these limitations through intelligent query routing, syntax-aware indexing, hybrid retrieval, and context-aware prompt construction.

---

# Objectives

The project aims to:

- Determine whether a query is programming-related before processing it.
- Distinguish between general coding questions and repository-specific questions.
- Route repository questions through the RAG pipeline only when necessary.
- Understand repository structure using AST parsing.
- Preserve programming language semantics during indexing.
- Build a searchable knowledge base of the codebase.
- Retrieve relevant code using both semantic and lexical search.
- Minimize LLM context while preserving architectural understanding.
- Stream responses in real time.
- Operate entirely on local repositories.

---

# High-Level Architecture

```
                    User Query
                         │
                         ▼
               Query Validation Layer
                         │
             ┌───────────┴───────────┐
             │                       │
       Invalid Coding Query     Valid Coding Query
             │                       │
             ▼                       ▼
     Ask User to Rephrase      Query Router
                                       │
                         ┌─────────────┴─────────────┐
                         │                           │
                  General Coding Query       Repository Query
                         │                           │
                         ▼                           ▼
                  Direct LLM Answer         Repository Scanner
                                                     │
                                                     ▼
                                            Parallel AST Parsing
                                                     │
                                                     ▼
                                 Symbol Table & Dependency Graph Construction
                                                     │
                                                     ▼
                                            Semantic Code Chunking
                                                     │
                                                     ▼
                                         Metadata & Docstring Generation
                                                     │
                                                     ▼
                                           Embedding Generation
                                                     │
                                      ┌──────────────┴──────────────┐
                                      ▼                             ▼
                               Chroma Vector DB                BM25 Index
                                      │                             │
                                      └──────────────┬──────────────┘
                                                     ▼
                                              Hybrid Retriever
                                                     │
                                                     ▼
                                             Context Pruning
                                                     │
                                                     ▼
                                             Prompt Construction
                                                     │
                                                     ▼
                                              Large Language Model
                                                     │
                                                     ▼
                                             Streaming Response
```

---

# Complete Workflow

## Phase 1 — User Query Validation
Every interaction begins with a lightweight validation and routing stage consolidated into a single call powered by the cost-effective and low-latency **Gemini 3.1 Flash-Lite** model. This prevents sequential LLM execution latency and minimizes cold start times. To achieve peak efficiency, this node utilizes native Pydantic structured outputs (binding to a `QueryClassification` model) and restricts generation limits to `max_output_tokens=50` to minimize execution latency.

The system evaluates the query in one step to determine both its validity (whether the prompt is programming/software-related) and its target routing target.

Examples of valid queries:
- Explain recursion.
- How does dependency injection work?
- Where is authentication implemented in this repository?
- Explain the login flow.
- What does this function do?

Examples of invalid queries:
- What's the weather today?
- Recommend a movie.
- Tell me a joke.

If the query is not related to programming, the system does not continue further. Instead, it politely asks the user to provide a programming or repository-related question.

### Input
Raw user prompt.

### Output
- Valid coding query (with pre-computed route)
- Invalid query (request user to rephrase)

---

## Phase 2 — Intelligent Query Routing
Once the query is validated and routed in the initial single-call optimization stage, the Router Node reads the cached routing decision from the graph's workflow state. If the cached decision is absent, it falls back to routing classification using **Gemini 3.1 Flash-Lite** with structured output support and a strict limit of `max_output_tokens=50` to maintain rapid routing speeds.

Two categories of programming questions are supported:

### General Programming Questions
These are conceptual questions that do not depend on the contents of the repository.

Examples:
- What is polymorphism?
- Explain Python decorators.
- What is dependency injection?
- Explain REST APIs.

These questions are answered directly by the LLM (using Gemini 3.1 Flash-Lite) without invoking the RAG pipeline.

Advantages:
- Faster response
- Lower latency
- No retrieval overhead
- Reduced computational cost

---

### Repository-Specific Questions
These questions require understanding the uploaded repository.

Examples:
- Explain the authentication flow.
- Where is this API called?
- How does the payment module work?
- Which function creates database connections?
- Explain this class.

These queries are forwarded to the repository analysis pipeline.

### Output
- Direct LLM route
- Repository RAG route

---

## Phase 3 — Repository Ingestion

The user provides the path to a local Git repository.

Example:

```bash
noesiscli analyze ~/Projects/MyApplication
```

The repository scanner recursively traverses the directory structure and identifies supported programming language files.

Examples include:

- Python
- Java
- JavaScript
- TypeScript
- C++
- Go

At this stage no code understanding occurs.

### Input

Local repository path.

### Output

List of source files.

---

## Phase 4 — Parallel AST Parsing

Instead of processing files one by one, NoesisCLI distributes parsing tasks across multiple CPU cores using Python's multiprocessing module.

Each worker independently parses a subset of files using Tree-sitter.

Since parsing is CPU-intensive, parallel execution drastically reduces repository indexing time.

Each parser extracts:

- Classes and Class Headers (signatures with method summaries)
- Functions (including async functions)
- Methods (retained in scope with parent class context)
- Imports (aggregated into a first-class chunk)
- Constants (ALL_CAPS module-level variables)
- Type Aliases (module-level typing constructs)
- Docstrings (module-level, class-level, and function-level)
- Decorators (retained and associated with classes/methods/functions)

To ensure reliable, syntax-aware chunking without structural fragmentation, the Tree-sitter parser implements the following specialized strategies and bug fixes:

### Applied Strategies
- **[S1] Module-Level Chunking:** Emits a `module` chunk containing the file-level docstring, aggregated import list, and file-level metadata.
- **[S2] Class-Header Chunking:** Emits a class signature header containing only the class definition, docstring, and method signatures (with bodies replaced by `...` placeholders) alongside the full `class` chunk. This acts as a skeletal context for pruning.
- **[S3] Global Node Classification:** Classifies global nodes as `constant`, `type_alias`, or `global` based on AST shape rather than always lumping everything under `global`.
- **[S4] Rich Per-Chunk Metadata:** Attaches decorator lists, `is_async` flags, `parent_class`, `is_dunder` flags, special type tags (e.g. `@property`, `@staticmethod`, `@classmethod`), docstrings, and file-level `imports_in_file` to every chunk.

### Applied Bug Fixes
- **[Fix 1] Decorator Line Inclusion:** Resolves start lines of decorated functions/classes using the `decorated_definition` wrapper to ensure decorator lines are included in the extracted `code_content`.
- **[Fix 2] Import Statement Isolation:** Prevents import statements from being treated as flush triggers for global code blocks, preventing accidental fragmentation of global constructs.
- **[Fix 3] Per-File Import Collection:** Collects import statements per-file so every chunk carries the file's import list to downstream dependency graph and retrieval phases.
- **[Fix 4] Nested Functions Prevention:** Does not recurse into nested functions from within a parent `function_definition` to avoid duplicate chunks; they remain in the parent's `code_content` and are separately extracted only from class body/module traversal.
- **[Fix 5] Async Function Detection:** Handles `async_function_definition` identically to `function_definition` by inspecting child tokens for the `async` keyword so async structures are never missed.
- **[Fix 6] Class Body Traversal:** Walks the explicit `block` child of a class definition rather than iterating all children, avoiding double-traversal of name, colon, or base-class nodes.

Rather than producing plain text, Tree-sitter generates an Abstract Syntax Tree representing the grammatical structure of each file.

Example:

Instead of:

```
1000-line Python file
```

The parser produces:

```
Module: UserService Module
Imports: import sys, from typing import Optional
Constant: MAX_RETRIES = 5
Class: UserService
Class Header: UserService (with docstring & signatures)
Method: authenticate()
Method: logout()
Function: verify_token()
```

### Input

Source files.

### Output

Structured Code Chunks.

---

## Phase 5 — Symbol Table & Dependency Graph Construction

To enable precise cross-referencing and contextual understanding, NoesisCLI processes the structured Code Chunks to construct a global Symbol Table and a Dependency Graph.

### Global Symbol Table
The symbol table maps all code symbols (such as classes, methods, functions, and interfaces) to their definitions, file locations, signatures, and scopes. This allows the system to resolve exact symbol names during retrieval and locate where they are declared.

### Dependency Graph
The dependency graph maps imports, inheritance, function call chains, and module relationships across the entire codebase. This captures how components interact and depend on each other.

By building these structures, NoesisCLI can resolve references and fetch related dependency context (e.g., parent classes, helper functions, or called methods) rather than just isolated code snippets.

### Input

Structured Code Chunks.

### Output

- Global Symbol Table
- Codebase Dependency Graph

---

## Phase 6 — Semantic Code Chunking

Traditional RAG systems split source code every 500 or 1000 characters.

This often divides functions into multiple chunks.

NoesisCLI instead creates chunks directly from AST nodes.

Each chunk represents exactly one logical programming construct.

Examples:

- Module overview (`module`)
- Dedicated import aggregation (`imports`)
- Full class body (`class`)
- Class header summary (`class_header`)
- Individual method (`method`)
- Standalone function (`function`)
- Constant assignment (`constant`)
- Type alias declaration (`type_alias`)
- Other top-level block (`global`)

Each chunk contains:

- Source code (`code_content`)
- File path (`file_path`)
- Node type (`node_type`)
- Start line & End line (`start_line`, `end_line`)
- Rich structural metadata (`decorators`, `is_async`, `parent_class`, `is_dunder`, `special_type`, `docstring`, `imports_in_file`)

This preserves complete semantic meaning.

### Output

Structured code chunks.

---

## Phase 7 — Metadata Generation

Every code chunk is enriched with searchable metadata, including relationships extracted from the Dependency Graph.

Example metadata:

```
Function Name

authenticate()

Language

Python

File

src/auth/service.py

Parent Class

UserService

Arguments

username
password

Returns

JWT Token

Visibility

Public

Dependencies
- Database (class)
- validate_token (function)
```

If documentation is missing, the high-stakes **Gemini 3.5 Flash** model is invoked to generate concise summaries describing each function, ensuring high-quality descriptions. If Gemini 3.5 Flash is rate-limited or fails, the system automatically falls back to **Gemini 3.1 Flash-Lite**.

These summaries improve retrieval quality without modifying the original source code.

### Output

Rich metadata attached to every chunk.

---

## Phase 8 — Embedding Generation

Each code chunk and generated summary is converted into dense vector embeddings.

To maximize throughput, embeddings are generated in batches rather than one chunk at a time.

The system uses the API-based embedding model:

```
voyage-code-3
```

provided by Voyage AI.

Advantages:

- State-of-the-art code retrieval performance
- Large context support for long code snippets
- No local GPU/CPU inference overhead
- Specialized for code and repository structure

Each embedding represents the semantic meaning of the corresponding code chunk.

### Output

Embedding vectors.

---

## Phase 9 — Index Construction

NoesisCLI builds two complementary search indexes, while keeping the Symbol Table and Dependency Graph in memory for real-time relational queries.

### Dense Vector Index

Stored in ChromaDB.

Supports semantic similarity search.

Example query:

```
How is authentication handled?
```

Even if no function contains the word "authentication", vector search may retrieve:

```
verify_token()

login_user()

generate_jwt()
```

---

### Lexical BM25 Index

Keyword-based retrieval.

Useful for exact matches.

Example:

```
UserService
```

or

```
authenticate()
```

BM25 immediately finds exact occurrences.

---

## Phase 10 — Hybrid Retrieval

When the user asks a repository-specific question, both retrieval systems execute simultaneously.

Example:

```
How are users authenticated?
```

Vector Search returns:

- validate_token()
- login()

BM25 returns:

- authenticate()

The hybrid retriever merges, ranks, and removes duplicates.

This combines semantic understanding with exact keyword matching, improving retrieval accuracy.

### Output

Ranked list of relevant code chunks.

---

## Phase 11 — Context Pruning

Large Language Models perform poorly when unnecessary code is included.

Rather than sending entire files, NoesisCLI constructs a minimal context. It uses the Symbol Table and Dependency Graph to identify direct dependencies (e.g., interfaces implemented, parent classes, or helper functions called) that are crucial for understanding the retrieved code.

Suppose only one function is relevant:

```python
def authenticate():
```

Instead of sending the entire 1200-line file, NoesisCLI uses the Dependency Graph to determine if `authenticate()` calls other local functions (e.g., `validate_token()`). It then constructs a pruned file structure:

```python
class UserService:

    def authenticate():
        # Implementation here
        validate_token(token)

    # Other non-essential functions are pruned to signatures/placeholders
    def hidden_helper():
        ...
```

If `validate_token()` is defined elsewhere, the Symbol Table is queried to resolve its location, and the Dependency Graph helps decide if its definition or signature should also be included in the context.

Benefits:

- Lower token usage
- Faster inference
- Accurate cross-file symbol resolution
- Better architectural understanding
- Reduced latency

---

## Phase 12 — Prompt Construction

The prompt contains:

- User question
- Relevant code chunks
- Dependency relationships and Symbol definitions
- Metadata
- File locations
- Generated summaries

Because only the most relevant functions and their direct dependency relationships are included, the prompt remains concise while retaining important context.

---

## Phase 13 — LLM Reasoning

The selected Large Language Model analyzes the prepared context and generates the final response. 

To optimize performance and accuracy, NoesisCLI employs a multi-model fallback strategy:
- **Path A (Direct LLM Response):** General programming queries are handled directly by the fast, low-cost **Gemini 3.1 Flash-Lite** model.
- **Path B (Repository-Aware Response):** Complex, repository-specific queries (a high-stakes task requiring deep code comprehension) are routed to **Gemini 3.5 Flash**. If Gemini 3.5 Flash encounters a rate limit or API failure, the system falls back to **Gemini 3.1 Flash-Lite** to generate the response and ensure continuous availability.

There are two execution paths.

### Path A — Direct LLM Response

For general programming questions, the LLM answers immediately without retrieval.

Example:

- Explain encapsulation.
- What is Big-O notation?

---

### Path B — Repository-Aware Response

For repository-specific questions, the LLM reasons over the retrieved code chunks.

Examples:

- Explain this module.
- How does authentication work?
- Trace the login flow.
- Where is this API called?
- Explain this class.
- Find potential bugs.
- Which function creates database connections?

The model reasons only over retrieved code rather than the entire repository.

---

## Phase 14 — Streaming Response

Instead of waiting for the entire answer to finish generating, NoesisCLI streams tokens to the terminal in real time.

Benefits:

- Near-zero perceived latency
- Faster user feedback
- Improved interactive experience

The total generation time remains similar, but the user receives information immediately.

---

# Data Flow

```
                     User Query
                          │
                          ▼
                Query Validation Layer
                          │
                Is it a coding question?
                 ┌────────┴────────┐
                 │                 │
                No                Yes
                 │                 │
                 ▼                 ▼
      Ask user to rephrase   Intelligent Query Router
                                   │
                      ┌────────────┴────────────┐
                      │                         │
               General Coding           Repository Query
                      │                         │
                      ▼                         ▼
               Direct LLM Answer      Local Repository
                                              │
                                              ▼
                                    Repository Scanner
                                              │
                                              ▼
                                   Parallel Tree-sitter Parsing
                                              │
                                              ▼
                                          AST Nodes
                                              │
                                              ▼
                           Symbol Table & Dependency Graph Creation
                                              │
                                              ▼
                                    Semantic Code Chunks
                                              │
                                              ▼
                                    Metadata Generation
                                              │
                                              ▼
                                 Batch Embedding Creation
                                              │
                                              ▼
                                 Dense Vector Index (ChromaDB)
                                              │
                                   ┌──────────┴──────────┐
                                   ▼                     ▼
                             Vector Search          BM25 Search
                                   └──────────┬──────────┘
                                              ▼
                                      Hybrid Retrieval
                                              ▼
                                      Context Pruning
                                              ▼
                                     Prompt Construction
                                              ▼
                                     Large Language Model
                                              ▼
                                     Streaming Explanation
```

---

# Technology Stack

| Component | Technology |
|-----------|------------|
| Programming Language | Python |
| Workflow Orchestration | LangGraph |
| LLM Framework | LangChain |
| Code Parsing | Tree-sitter |
| Parallel Processing | multiprocessing |
| Symbol Table | Custom Registry |
| Dependency Graph | NetworkX |
| Embedding Model | Voyage AI voyage-code-3 |
| Inference Engine | Voyage AI API (HTTP Client) |
| Vector Database | ChromaDB |
| Lexical Search | BM25 |
| Hybrid Retrieval | Custom Rank Fusion |
| CLI Framework | argparse (Standard Library) |
| LLM | Gemini 3.5 Flash (Primary) & Gemini 3.1 Flash-Lite (Fallback/Routing) |
| Streaming | LangChain Streaming Callbacks |

---

# Key Features

- Consolidated query validation and routing using Gemini 3.1 Flash-Lite to eliminate sequential LLM latency
- Multi-model architecture: Gemini 3.5 Flash for high-stakes reasoning/summarization, and Gemini 3.1 Flash-Lite for routing, validation, and fallback support
- Automatic fallback: fallback to Gemini 3.1 Flash-Lite in case of Gemini 3.5 Flash API rate limits or failures
- Automatic routing between direct LLM and RAG pipeline
- Syntax-aware AST code chunking
- Parallel repository parsing using multiprocessing
- Global Symbol Table and Dependency Graph for relationship tracing
- Automatic metadata and summary generation
- Batch embedding generation using Voyage AI API (voyage-code-3)
- Hybrid retrieval using dense vectors and BM25
- Context-aware prompt pruning
- Streaming LLM responses
- Local-first architecture with local database storage and hybrid retrieval
- Efficient token usage and reduced inference latency

---

# Expected Outcome

NoesisCLI enables developers to interact with unfamiliar codebases conversationally while preserving programming language semantics throughout the retrieval pipeline. Before any repository analysis begins, the system intelligently validates and classifies user queries, ensuring that only repository-dependent questions invoke the RAG pipeline while general programming questions are answered directly by the LLM. By combining intelligent query routing, AST-based parsing, a global symbol table and dependency graph, hybrid search, context pruning, and streaming inference, NoesisCLI delivers fast, accurate, and context-efficient explanations of complex software repositories without requiring manual code exploration.