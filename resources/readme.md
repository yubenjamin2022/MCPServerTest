# Demo MCP Server

This is a minimal example of a **Model Context Protocol (MCP)** server.

It demonstrates:
- How to register a Markdown resource (`resources://readme`)
- How to define and serve a simple tool (`tools://summarize_text`)

## Available Tool
### `summarize_text`
Summarizes long text inputs into shorter previews (a toy example).

## Future Extensions
- Integrate a real LLM-based summarizer
- Add domain-specific scientific tools (e.g., molecule parsers, data visualizers)
