# EDGAR Data Chatbot

Natural-language interface to your EDGAR extraction database. Ask questions in plain English; the chatbot translates them into SQL, runs the query, and explains the results.

**Port:** 5100 (so it doesn't conflict with EDGAR Loader on 5070, DQX on 5050, SEDAR+ on 5080, deXtra on 5090)

## What It Can Answer

**Filing History**
- "What's the most recent 10-K we've extracted for CIK 1869453?"
- "Show me all filings from PIMCO in 2025"
- "How many 10-K filings have we processed?"

**NAV & Performance**
- "What's the latest NAV for CIK 1869453?"
- "Show me NAV history for all share classes of CIK 1869453"
- "What's the 1-year return for this fund's share classes?"

**Distributions**
- "Show me all distributions paid by CIK 1869453 in 2025"
- "What's the total distributions YTD for this portfolio?"

**Extraction Analytics**
- "How many extractions have we run this month?"
- "What's the total cost of all extractions?"
- "Show me extractions with errors"
- "How many corrections have reviewers logged?"

**Composition**
- "What's the asset allocation for CIK 1869453?"
- "Which funds have more than 20% in real estate?"

## Architecture

1. **User types a question** in natural English
2. **Claude (Sonnet 4.6)** translates it into SQL — it knows the full schema (34 tables, 508 fields)
3. **Safety layer** validates the SQL is read-only (only SELECT allowed, forbidden keywords blocked)
4. **PostgreSQL** executes the query against the extraction database
5. **Claude** summarizes the results in natural language (tables, highlights, context)

The schema context is baked into the system prompt — Claude knows about every PE table, primary keys, relationships, date formats, and quoting rules.

## Setup

```bash
pip install -r requirements.txt

# Edit chatbot_config.json:
#   - database.password
#   - anthropic.api_key

python chatbot.py
# Opens on http://localhost:5100
```

## Features

- Conversational follow-ups (remembers the last 10 turns)
- SQL visibility (click "📝 SQL" to see the generated query)
- Raw data inspection (click "📊 Data" to see the query results as a table)
- Conversation history (sidebar)
- Markdown rendering (tables, lists, code)
- Dark/light theme
- Health indicator (DB connection + API key status)

## Safety

- Only SELECT queries allowed
- Forbidden keywords blocked: INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, GRANT, REVOKE, COPY
- Result limit: 500 rows per query
- All queries logged with input/output token usage

## Configuration

`chatbot_config.json` — database connection, API key, port, max rows
`conversations/*.json` — one file per conversation (auto-saved)

**(C) 2026 EXF Financial Data Solutions**
