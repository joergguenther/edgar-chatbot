#!/usr/bin/env python3
"""
EDGAR Data Chatbot
Natural language interface to extracted EDGAR data.
Uses Claude to translate questions into SQL against the PE schema.
"""
import json
import os
import re
import getpass
import traceback
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, render_template, Response

import psycopg2
import psycopg2.extras
import requests

BASE_DIR = Path(__file__).parent
CONFIG_FILE = BASE_DIR / "chatbot_config.json"

# ──────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────

def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {
        "database": {
            "host": "a402389-akamai-prod-6092099-default.g2a.akamaidb.net",
            "port": 16149,
            "dbname": "defaultdb",
            "user": "akmadmin",
            "password": "",
            "schema": "newdev_private_equity",
            "reference_schema": "newdev_public_equity",
            "sslmode": "require",
        },
        "anthropic": {
            "api_key": "",
            "model": "claude-sonnet-4-6",
        },
        "models": [
            {"id": "claude-opus-4-6", "name": "Claude Opus 4.6", "description": "Most capable, slower, higher cost",
             "input_price": 15.0, "output_price": 75.0},
            {"id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6", "description": "Balanced — recommended default",
             "input_price": 3.0, "output_price": 15.0},
            {"id": "claude-haiku-4-5-20251001", "name": "Claude Haiku 4.5", "description": "Fastest and cheapest",
             "input_price": 1.0, "output_price": 5.0},
        ],
        "app": {
            "port": 5100,
            "max_rows": 500,
            "conversation_limit": 20,  # how many past turns to keep
            "loader_url": "http://localhost:5070",  # EDGAR Loader API for triggering extractions
        },
    }

def save_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

CONFIG = load_config()

# Don't prompt for password if running in production — use Settings UI instead
if not CONFIG.get("database", {}).get("password"):
    env_pw = os.environ.get("EDGAR_DB_PASSWORD", "")
    if env_pw:
        CONFIG["database"]["password"] = env_pw
        save_config(CONFIG)


def get_conn():
    db = CONFIG["database"]
    if not db.get("password"):
        print(f"  [db] ERROR: No password in CONFIG. Keys present: {list(db.keys())}")
        raise RuntimeError("Database password not configured. Open Settings to configure.")
    print(f"[db] Connecting to {db['host']}:{db.get('port', 16149)}/{db['dbname']} as {db['user']} (pw: {len(db['password'])} chars)")
    return psycopg2.connect(
        host=db["host"], port=db["port"],
        dbname=db["dbname"], user=db["user"],
        password=db["password"], sslmode=db.get("sslmode", "require"),
    )


def get_model_pricing(model_id):
    """Return (input_price_per_mtok, output_price_per_mtok) for a model."""
    for m in CONFIG.get("models", []):
        if m.get("id") == model_id:
            return float(m.get("input_price", 0)), float(m.get("output_price", 0))
    # Fallback defaults for unknown models
    return 3.0, 15.0


def calc_cost(input_tokens, output_tokens, model_id):
    """Calculate USD cost from token counts. Prices are per million tokens."""
    in_price, out_price = get_model_pricing(model_id)
    cost = (input_tokens * in_price + output_tokens * out_price) / 1_000_000
    return round(cost, 6)


# ──────────────────────────────────────────────────────────────────
# SCHEMA CONTEXT — fed to Claude so it knows what to query
# ──────────────────────────────────────────────────────────────────

_SCHEMA_CACHE = {"context": None, "updated": None}


def introspect_schema():
    """Query the database for actual tables and columns in the configured schemas.
    Returns a formatted string that goes into Claude's system prompt."""
    pe_schema = CONFIG["database"].get("schema", "newdev_private_equity")
    ref_schema = CONFIG["database"].get("reference_schema", "newdev_public_equity")

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # Get all tables + columns for both schemas
            cur.execute("""
                SELECT table_schema, table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema IN (%s, %s)
                ORDER BY table_schema, table_name, ordinal_position
            """, (pe_schema, ref_schema))
            rows = cur.fetchall()

            # Get primary keys
            cur.execute("""
                SELECT tc.table_schema, tc.table_name, kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                 AND tc.table_schema = kcu.table_schema
                WHERE tc.constraint_type = 'PRIMARY KEY'
                  AND tc.table_schema IN (%s, %s)
                ORDER BY tc.table_name, kcu.ordinal_position
            """, (pe_schema, ref_schema))
            pk_rows = cur.fetchall()
    finally:
        conn.close()

    # Organize columns by table
    tables = {}
    for schema, table, col, dtype in rows:
        key = (schema, table)
        tables.setdefault(key, []).append((col, dtype))

    # Organize PKs
    pks = {}
    for schema, table, col in pk_rows:
        pks.setdefault((schema, table), []).append(col)

    if not tables:
        return (
            f"WARNING: No tables found in schemas '{pe_schema}' or '{ref_schema}'.\n"
            "Check that the schema name in Settings is correct and the DDL has been run."
        )

    lines = []
    lines.append("You are a SQL assistant for a PostgreSQL database containing extracted EDGAR fund data.\n")
    lines.append("# SCHEMAS")
    lines.append(f"- `{pe_schema}` — extracted fund data")
    lines.append(f"- `{ref_schema}` — reference tables (portfolios, share classes)")
    lines.append("")
    lines.append("CRITICAL: PostgreSQL requires mixed-case schema/table/column names to be DOUBLE-QUOTED separately.")
    lines.append(f'Correct:   SELECT * FROM "{pe_schema}"."T_PE_FUND_SHARE_CLASS_NAV_PRICING"')
    lines.append(f'Wrong:     SELECT * FROM {pe_schema}.T_PE_FUND_SHARE_CLASS_NAV_PRICING  (lowercases identifiers)')
    lines.append("")
    lines.append("# TABLES (exact names and columns from the live database)")
    lines.append("")

    # Sort: reference schema first, then PE tables
    sorted_tables = sorted(tables.keys(), key=lambda k: (k[0] != ref_schema, k[0], k[1]))
    for schema, table in sorted_tables:
        cols = tables[(schema, table)]
        pk_cols = pks.get((schema, table), [])
        pk_str = f" — PK: {', '.join(pk_cols)}" if pk_cols else ""
        lines.append(f'## "{schema}"."{table}"{pk_str}')
        col_strs = [f'{c} ({t})' for c, t in cols]
        lines.append(f"  {', '.join(col_strs)}")
        lines.append("")

    lines.append("# QUERY GUIDELINES")
    lines.append("1. ALWAYS use the EXACT schema-qualified table names shown above, with double quotes.")
    lines.append('   CORRECT:   FROM "{}"."TableName"'.format(pe_schema))
    lines.append('   WRONG:     FROM "TableName"       (no schema)')
    lines.append('   WRONG:     FROM TableName          (no quotes)')
    lines.append('   WRONG:     FROM tablename          (lowercase — not a real table)')
    lines.append("2. Every table referenced MUST appear in the list above. Do NOT invent table names.")
    lines.append("3. If the user asks about a concept and no matching table exists, respond that the data isn't available")
    lines.append("   rather than guessing at a table name.")
    lines.append("4. Filter by CIK when searching for a specific company — CIKs are usually stored without leading zeros.")
    lines.append("5. LIMIT all queries to 500 rows maximum.")
    lines.append("6. Join via PortfolioID or ShareClassID, not on CIK alone.")
    lines.append("7. Dates are DATE type. Use 'YYYY-MM-DD' literals.")
    lines.append('8. For "most recent", use ORDER BY <date_col> DESC LIMIT 1.')
    lines.append("9. Return ONLY the SQL query — no explanation, no markdown, no backticks.")
    lines.append("10. If a table name or column name contains mixed case, you MUST double-quote it.")

    return "\n".join(lines)


def get_schema_context(force_refresh=False):
    """Return cached schema context, introspecting if needed."""
    if _SCHEMA_CACHE["context"] and not force_refresh:
        return _SCHEMA_CACHE["context"]
    try:
        ctx = introspect_schema()
        _SCHEMA_CACHE["context"] = ctx
        _SCHEMA_CACHE["updated"] = datetime.utcnow().isoformat()
        # Count tables found
        table_count = ctx.count("## \"")
        print(f"[schema] Refreshed: {len(ctx):,} chars, ~{table_count} tables")
        return ctx
    except Exception as e:
        print(f"[schema] Introspection FAILED: {e}")
        traceback.print_exc()
        # Mark cache as failed with clear marker
        ctx = (
            f"SCHEMA_INTROSPECTION_FAILED\n"
            f"Error: {e}\n"
            f"Host: {CONFIG['database'].get('host')}\n"
            f"DB: {CONFIG['database'].get('dbname')}\n"
            f"Schemas: {CONFIG['database'].get('schema')}, {CONFIG['database'].get('reference_schema')}\n"
            f"Fix this by opening Settings → Diagnose."
        )
        _SCHEMA_CACHE["context"] = ctx
        _SCHEMA_CACHE["updated"] = datetime.utcnow().isoformat()
        return ctx




# ──────────────────────────────────────────────────────────────────
# CLAUDE NL→SQL
# ──────────────────────────────────────────────────────────────────

def call_claude(messages, system=None, max_tokens=4096):
    """Call Claude API with the given messages."""
    api_key = CONFIG["anthropic"]["api_key"]
    if not api_key:
        raise RuntimeError("Anthropic API key not configured")

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": CONFIG["anthropic"]["model"],
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        payload["system"] = system

    resp = requests.post("https://api.anthropic.com/v1/messages",
                         json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    return text.strip(), data.get("usage", {})


_KNOWN_TABLES_CACHE = {"tables": None, "updated": None}


def get_known_tables():
    """Return set of (schema, table) tuples that actually exist in the configured schemas."""
    if _KNOWN_TABLES_CACHE["tables"] is not None:
        return _KNOWN_TABLES_CACHE["tables"]
    pe_schema = CONFIG["database"].get("schema", "")
    ref_schema = CONFIG["database"].get("reference_schema", "")
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_schema, table_name FROM information_schema.tables
                WHERE table_schema IN (%s, %s)
            """, (pe_schema, ref_schema))
            result = set()
            for s, t in cur.fetchall():
                result.add((s.lower(), t.lower()))
            _KNOWN_TABLES_CACHE["tables"] = result
            return result
    finally:
        conn.close()


def validate_sql_tables(sql):
    """Check that every FROM/JOIN references a known table. Returns error message or None."""
    try:
        known = get_known_tables()
    except Exception:
        return None  # can't validate, let the query run and fail naturally

    if not known:
        return "No tables found in configured schemas — open Settings → Diagnose"

    # Extract all FROM/JOIN table references
    # Matches: FROM/JOIN "schema"."table" or FROM/JOIN schema.table or FROM/JOIN table
    pattern = re.compile(
        r'\b(?:FROM|JOIN)\s+'
        r'(?:"([^"]+)"\.)?'       # optional schema (quoted)
        r'(?:([a-zA-Z_][a-zA-Z0-9_]*)\.)?'  # or unquoted schema
        r'(?:"([^"]+)"|([a-zA-Z_][a-zA-Z0-9_]*))',  # table name (quoted or bare)
        re.IGNORECASE
    )
    refs = []
    for m in pattern.finditer(sql):
        quoted_schema, bare_schema, quoted_table, bare_table = m.groups()
        schema = (quoted_schema or bare_schema or "").lower()
        table = (quoted_table or bare_table or "").lower()
        if not table:
            continue
        # Skip obvious subquery aliases and information_schema
        if table in ('information_schema', 'pg_catalog'):
            continue
        refs.append((schema, table))

    # Check each reference
    pe_schema = CONFIG["database"].get("schema", "").lower()
    ref_schema = CONFIG["database"].get("reference_schema", "").lower()

    for schema, table in refs:
        if not schema:
            # Unqualified table — check if it exists in either configured schema
            found = any(s in (pe_schema, ref_schema) and t == table for s, t in known)
            if not found:
                return (
                    f"Query references unqualified table '{table}' which doesn't exist "
                    f"in schema '{pe_schema}' or '{ref_schema}'."
                )
        else:
            if (schema, table) not in known:
                return (
                    f"Query references table \"{schema}\".\"{table}\" which doesn't exist."
                )
    return None


def nl_to_sql(question, conversation_history=None):
    """Translate a natural language question into SQL — or detect action intent.
    Returns (response_type, payload, usage) where:
    - response_type='sql': payload is the SQL query string
    - response_type='extract': payload is a dict with extraction parameters
    - response_type='edgar_search': payload is a dict with search parameters
    - response_type='edgar_filing': payload is a dict with filing parameters
    - response_type='web_search': payload is a dict with search query
    """
    history = conversation_history or []

    # Get schema context FIRST. Fail loudly if empty or failed.
    schema_ctx = get_schema_context()
    if not schema_ctx or "WARNING: No tables found" in schema_ctx or "SCHEMA_INTROSPECTION_FAILED" in schema_ctx:
        raise RuntimeError(
            f"Schema not loaded. {schema_ctx[:500] if schema_ctx else 'Empty context.'} "
            f"Open Settings → 🔍 Diagnose to see what's wrong."
        )
    if schema_ctx.count("## \"") == 0:
        raise RuntimeError(
            "Schema context has no tables. Open Settings → 🔍 Diagnose to see what's in the database."
        )
    print(f"[nl_to_sql] schema context: {len(schema_ctx):,} chars, {schema_ctx.count('## \"')} tables")

    # Add extraction capability context
    extraction_ctx = """

# CAPABILITIES
You have three modes of response depending on what the user needs:

## MODE 1: SQL QUERY (default)
For questions about data ALREADY in our database. Return ONLY a SELECT query.
Examples: "Show NAV for CIK 1234", "Compare leverage across BDCs", "Latest distributions"

## MODE 2: EXTRACTION (new data from EDGAR filings into our DB)
When user wants to "extract", "fetch", "pull", "load" NEW structured data from SEC filings.
Return: ACTION:EXTRACT:{"cik": "0001920145", "domains": ["returns"], "period": "most_recent"}
Or: ACTION:EXTRACT:{"company_name": "Blue Owl", "domains": ["nav_pricing", "returns"], "period": "annual", "year": 2024}

Available domains: filing_master, nav_pricing, shares_outstanding, volatility, distributions,
dist_ytd, dist_drip, dist_metrics, dist_tax, composition, comp_objectives, leverage,
leverage_summary, leverage_covenant, returns, fees, repurchase_fees, investor_eligibility,
offering_price, redemptions, tender_program, account_metrics, operational_details

## MODE 3: EDGAR SEARCH (browse filings on SEC EDGAR)
When user asks "what filings does X have", "search EDGAR for", "look up CIK", "find filings".
Return: ACTION:EDGAR_SEARCH:{"cik": "0001920145"}
Or: ACTION:EDGAR_SEARCH:{"company_name": "Goldman Sachs Private Credit"}
Or for full-text search: ACTION:EDGAR_SEARCH:{"query": "tender offer Blue Owl", "forms": "SC TO-I"}

## MODE 4: EDGAR FILING (read a specific filing)
When user asks to "read", "show me the filing", "what does the 10-K say about".
Return: ACTION:EDGAR_FILING:{"cik": "0001920145", "filing_type": "10-K", "query": "total return"}
Or: ACTION:EDGAR_FILING:{"cik": "0001920145", "accession": "0001193125-25-277550", "query": "leverage"}

## MODE 5: WEB SEARCH (general internet search)
When user asks about general market info, news, or anything not in our DB or EDGAR.
Return: ACTION:WEB_SEARCH:{"query": "BDC industry trends 2025"}

DECISION RULES:
- "Show NAV for CIK 1234" → SQL (existing data)
- "Extract NAV for CIK 1234" → EXTRACT (new data from filings)
- "What filings does Blue Owl have?" → EDGAR_SEARCH
- "What does the 10-K say about leverage?" → EDGAR_FILING
- "What are current BDC market trends?" → WEB_SEARCH
- "Look up CIK for Goldman Sachs" → EDGAR_SEARCH
- "Read the latest 8-K for CIK 1920145" → EDGAR_FILING

COMMON SQL PATTERNS:
- Company lookup: JOIN reference portfolio table on "PortfolioID" (has "PortfolioLongName", "CIK")
- NAV: T_PE_FUND_SHARE_CLASS_NAV_PRICING."NAVPS", "ReportDate"
- Returns: T_PE_FUND_SHARE_CLASS_RETURNS."Fund1Year", "FundYTD"
- Distributions: T_PE_FUND_SHARE_CLASS_DISTRIBUTION_HISTORY."DistributionPerShare"
- Extraction log: T_PE_FUND_EXTRACTION_LOG for timestamps and costs

Return ONLY one of: SELECT query | ACTION:EXTRACT:{json} | ACTION:EDGAR_SEARCH:{json} | ACTION:EDGAR_FILING:{json} | ACTION:WEB_SEARCH:{json}"""

    # Build messages
    messages = []
    for turn in history[-10:]:
        if turn.get("error"):
            continue
        if turn.get("user"):
            messages.append({"role": "user", "content": turn["user"]})
        if turn.get("assistant_sql"):
            messages.append({"role": "assistant", "content": turn["assistant_sql"]})

    messages.append({
        "role": "user",
        "content": (
            f"Answer this request: {question}\n\n"
            f"Return ONLY one of:\n"
            f"- A SELECT SQL query (for existing data)\n"
            f"- ACTION:EXTRACT:{{json}} (for new extractions)\n"
            f"- ACTION:EDGAR_SEARCH:{{json}} (to browse EDGAR filings)\n"
            f"- ACTION:EDGAR_FILING:{{json}} (to read a specific filing)\n"
            f"- ACTION:WEB_SEARCH:{{json}} (for internet search)\n"
            f"Remember: SQL tables MUST use exact schema names, fully double-quoted."
        )
    })

    total_usage = {"input_tokens": 0, "output_tokens": 0}
    response = ""

    for attempt in range(2):
        response, usage = call_claude(messages, system=schema_ctx + extraction_ctx)
        total_usage["input_tokens"] += usage.get("input_tokens", 0)
        total_usage["output_tokens"] += usage.get("output_tokens", 0)

        response = response.strip()

        # Check for any ACTION: prefix
        action_match = re.search(r'ACTION:(EXTRACT|EDGAR_SEARCH|EDGAR_FILING|WEB_SEARCH):\s*(\{.*\})', response, re.DOTALL)
        if action_match:
            action_type = action_match.group(1).lower()
            json_str = action_match.group(2)
            # Try to parse, with brace-depth repair
            params = _parse_action_json(json_str)
            if params is not None:
                return action_type, params, total_usage

        # Otherwise treat as SQL
        sql = re.sub(r"^```(?:sql)?\s*", "", response)
        sql = re.sub(r"\s*```$", "", sql)
        sql = sql.strip()

        # Validate table references
        error = validate_sql_tables(sql)
        if not error:
            return "sql", sql, total_usage

        if attempt == 0:
            print(f"[nl_to_sql] Validation failed (attempt 1): {error}")
            messages.append({"role": "assistant", "content": sql})
            messages.append({
                "role": "user",
                "content": (
                    f"That query references tables that don't exist. Error: {error}\n\n"
                    f"Re-read the table list and generate a new query using ONLY tables that appear there."
                )
            })
        else:
            print(f"[nl_to_sql] Validation failed (attempt 2): {error} — returning anyway")

    return "sql", sql, total_usage


def summarize_results(question, sql, rows, columns):
    """Ask Claude to summarize the query results in natural language."""
    # Truncate for prompt if too many rows
    preview_rows = rows[:50]
    rows_json = json.dumps(preview_rows, default=str, indent=2)

    system = (
        "You are a data analyst. Explain query results clearly and concisely. "
        "Focus on the key insight. If the result is a single number or date, "
        "state it directly. If it's a list, highlight the most important entries. "
        "Use markdown tables for tabular comparisons. Keep responses under 200 words."
    )

    prompt = f"""Question: {question}

SQL executed:
{sql}

Results ({len(rows)} row(s), showing first {len(preview_rows)}):
{rows_json}

Columns: {columns}

Answer the user's question based on these results. Be specific and concise."""

    text, usage = call_claude([{"role": "user", "content": prompt}],
                              system=system, max_tokens=1500)
    return text, usage


def execute_sql(sql, max_rows=500):
    """Execute SQL safely (read-only) and return rows + columns."""
    # Safety: only SELECT queries allowed
    sql_stripped = sql.strip().upper()
    if not sql_stripped.startswith("SELECT") and not sql_stripped.startswith("WITH"):
        raise ValueError("Only SELECT queries are allowed")

    # Block SQL comments (can be used to hide malicious code after a LIMIT injection)
    if "--" in sql or "/*" in sql or "*/" in sql:
        raise ValueError("SQL comments are not allowed")

    # Block dangerous keywords — use word-boundary regex on actual keywords only
    forbidden_kw = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
                    "CREATE", "GRANT", "REVOKE", "COPY", "VACUUM", "ANALYZE",
                    "REINDEX", "CLUSTER", "LOCK", "EXECUTE", "CALL"]
    for kw in forbidden_kw:
        if re.search(rf"\b{kw}\b", sql, re.IGNORECASE):
            raise ValueError(f"Forbidden keyword in SQL: {kw}")

    # Block multiple statements (semicolon not at the very end)
    sql_no_trailing = sql.rstrip().rstrip(";").rstrip()
    if ";" in sql_no_trailing:
        raise ValueError("Multiple SQL statements are not allowed")

    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Wrap in a LIMIT if there isn't one
            if "LIMIT" not in sql.upper():
                sql_with_limit = sql.rstrip().rstrip(";") + f" LIMIT {max_rows}"
            else:
                sql_with_limit = sql

            cur.execute(sql_with_limit)
            if cur.description:
                columns = [d.name for d in cur.description]
                rows = [dict(r) for r in cur.fetchall()]
            else:
                columns = []
                rows = []
        return rows, columns
    finally:
        conn.close()


# ──────────────────────────────────────────────────────────────────
# CONVERSATION STORAGE
# ──────────────────────────────────────────────────────────────────

CONVERSATIONS_DIR = BASE_DIR / "conversations"
CONVERSATIONS_DIR.mkdir(exist_ok=True)

def load_conversation(conv_id):
    path = CONVERSATIONS_DIR / f"{conv_id}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"id": conv_id, "turns": [], "created_at": datetime.utcnow().isoformat()}

def save_conversation(conv):
    path = CONVERSATIONS_DIR / f"{conv['id']}.json"
    with open(path, "w") as f:
        json.dump(conv, f, indent=2, default=str)


# ──────────────────────────────────────────────────────────────────
# FLASK APP
# ──────────────────────────────────────────────────────────────────

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))


@app.route("/")
def index():
    return render_template("chatbot.html")


def _parse_action_json(json_str):
    """Parse JSON from an ACTION response, with brace-depth repair for trailing text."""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Try brace-depth repair
        depth = 0
        for i, ch in enumerate(json_str):
            if ch == '{': depth += 1
            elif ch == '}': depth -= 1
            if depth == 0 and i > 0:
                try:
                    return json.loads(json_str[:i+1])
                except json.JSONDecodeError:
                    break
    return None


def _trigger_extraction(params):
    """Call the EDGAR Loader API to trigger a new extraction.
    params: {"cik": "...", "domains": [...], "period": "most_recent", "year": 2024, "quarter": 1}
    Also supports {"company_name": "Blue Owl", ...} — will look up CIK first.
    """
    loader_url = CONFIG.get("app", {}).get("loader_url", "http://localhost:5070")
    cik = params.get("cik", "").strip()
    company_name = params.get("company_name", "").strip()
    domains = params.get("domains", [])
    period = params.get("period", "most_recent")

    # If company_name provided but no CIK, look it up from the DB first
    if not cik and company_name:
        try:
            conn = get_conn()
            ref_schema = CONFIG["database"].get("reference_schema", "newdev_public_equity")
            with conn.cursor() as cur:
                cur.execute(f'''
                    SELECT "CIK", "PortfolioLongName" FROM "{ref_schema}"."T_PORT_PORTFOLIO"
                    WHERE "PortfolioLongName" ILIKE %s
                    ORDER BY "PortfolioID" LIMIT 5
                ''', (f'%{company_name}%',))
                matches = cur.fetchall()
            conn.close()

            if matches:
                if len(matches) == 1:
                    cik = str(matches[0][0])
                    company_name = matches[0][1]
                    print(f"  [extract] Resolved company '{company_name}' → CIK {cik}")
                else:
                    match_list = ", ".join(f"{m[1]} (CIK {m[0]})" for m in matches[:5])
                    return {"answer": f"Multiple matches found for '{company_name}': {match_list}. Please specify the CIK number."}
            else:
                return {"answer": f"No portfolio found matching '{company_name}' in the database. Please provide the CIK number directly."}
        except Exception as e:
            return {"answer": f"Could not look up company name: {e}. Please provide the CIK number directly."}

    if not cik:
        return {"error": "No CIK provided for extraction", "answer": "I need a CIK number or company name to trigger an extraction."}

    if not domains:
        return {"error": "No domains specified", "answer": "Please specify which domains to extract (e.g., returns, nav_pricing, distributions)."}

    try:
        # Step 1: Search EDGAR for the company
        search_resp = requests.get(f"{loader_url}/api/edgar/search",
                                   params={"cik": cik}, timeout=30)
        if search_resp.status_code != 200:
            return {"error": f"EDGAR search failed: HTTP {search_resp.status_code}",
                    "answer": f"Could not search EDGAR for CIK {cik}. Is the EDGAR Loader running at {loader_url}?"}

        search_data = search_resp.json()
        if search_data.get("error"):
            return {"error": search_data["error"], "answer": f"EDGAR search error: {search_data['error']}"}

        company = search_data.get("company", {})
        company_name = company.get("name", f"CIK {cik}")
        filings = search_data.get("filings", [])

        # Step 1b: Detect fund type for better plan building
        fund_type = params.get("fund_type", "Unknown")
        if fund_type == "Unknown":
            try:
                dt_resp = requests.post(f"{loader_url}/api/detect-fund-type", json={
                    "content": "", "company_name": company_name,
                    "entity_type": company.get("entityType", ""),
                    "sic": company.get("sic", ""),
                    "filing_type": "", "filing_forms": [f.get("form", "") for f in filings[:40]],
                }, timeout=15)
                if dt_resp.status_code == 200:
                    fund_type = dt_resp.json().get("fund_type", "Unknown")
                    print(f"  [extract] Detected fund type: {fund_type}")
            except Exception:
                pass

        # Step 2: Build a plan using Smart Fetch
        plan_resp = requests.post(f"{loader_url}/api/smart-fetch/plan", json={
            "cik": cik,
            "fund_type": fund_type,
            "domains": domains,
            "period_type": period,
            "year": params.get("year"),
            "quarter": params.get("quarter"),
        }, timeout=30)

        if plan_resp.status_code != 200:
            return {"error": f"Plan failed: HTTP {plan_resp.status_code}",
                    "answer": f"Could not build extraction plan for {company_name}."}

        plan_data = plan_resp.json()
        plan = plan_data.get("plan", [])

        if not plan:
            return {"answer": f"No filings found for {company_name} matching the requested domains ({', '.join(domains)}) and period ({period}). The company has {len(filings)} total filings on EDGAR."}

        filing_summary = ", ".join(f"{p['filing_type']} ({len(p.get('domains', []))} domains)" for p in plan)

        # Step 3: Fetch filing content from EDGAR
        print(f"  [extract] Step 3: Fetching {len(plan)} filing(s) from EDGAR...")
        exec_resp = requests.post(f"{loader_url}/api/smart-fetch/execute", json={
            "cik": cik,
            "company_name": company_name,
            "plan": plan,
            "entity_type": company.get("entityType", ""),
            "sic": company.get("sic", ""),
            "sic_description": company.get("sicDescription", ""),
            "fund_type": fund_type,
            "deep_scan": True,
        }, timeout=120)

        if exec_resp.status_code != 200:
            return {"error": f"Filing fetch failed: HTTP {exec_resp.status_code}",
                    "answer": f"Could not fetch filings for {company_name}."}

        exec_data = exec_resp.json()
        ready_results = [r for r in exec_data.get("results", []) if r.get("status") == "ready"]

        if not ready_results:
            return {"answer": f"No filings could be fetched for {company_name}. Plan had {len(plan)} filing(s) but none were retrievable."}

        # Step 4: Extract each domain from each filing via /api/extract
        print(f"  [extract] Step 4: Extracting {len(domains)} domain(s) from {len(ready_results)} filing(s)...")
        extraction_results = []
        total_rows = 0

        for filing_result in ready_results:
            filing = filing_result.get("filing", {})
            content = filing_result.get("content", "")
            raw_content = filing_result.get("raw_content")
            filing_url = filing_result.get("filing_url", "")
            f_type = filing.get("form", filing_result.get("plan_item", {}).get("filing_type", ""))
            accession = filing.get("accessionNumber", "")
            domain_ids = filing_result.get("domains", [])

            is_nport = f_type.upper().startswith("N-PORT")

            for domain_id in domain_ids:
                if domain_id not in domains:
                    continue

                print(f"  [extract]   → {domain_id} from {f_type} ({accession[:20]}...)...")

                try:
                    ext_resp = requests.post(f"{loader_url}/api/extract", json={
                        "domain": domain_id,
                        "content": raw_content if raw_content and is_nport else content,
                        "cik": cik,
                        "company_name": company_name,
                        "accession": accession,
                        "filing_type": f_type,
                        "filing_url": filing_url,
                        "entity_type": company.get("entityType", ""),
                        "sic": company.get("sic", ""),
                        "sic_description": company.get("sicDescription", ""),
                        "fund_type": fund_type,
                        "deep_scan": True,
                        "check_filing_type": True,
                        "check_relevance": True,
                        "retry_on_empty": True,
                        "check_thousands": True,
                    }, timeout=300)

                    if ext_resp.status_code == 200:
                        ext_data = ext_resp.json()
                        row_count = ext_data.get("row_count", 0)
                        domain_name = ext_data.get("domain", domain_id)
                        skipped = ext_data.get("skipped", False)
                        skip_reason = ext_data.get("skip_reason", "")

                        extraction_results.append({
                            "domain_id": domain_id,
                            "domain_name": domain_name,
                            "row_count": row_count,
                            "filing_type": f_type,
                            "skipped": skipped,
                            "skip_reason": skip_reason,
                        })
                        total_rows += row_count
                        status = f"SKIPPED: {skip_reason}" if skipped else f"{row_count} rows"
                        print(f"  [extract]     ✓ {domain_name}: {status}")
                    else:
                        extraction_results.append({
                            "domain_id": domain_id,
                            "domain_name": domain_id,
                            "row_count": 0,
                            "error": f"HTTP {ext_resp.status_code}",
                        })
                        print(f"  [extract]     ✗ {domain_id}: HTTP {ext_resp.status_code}")
                except Exception as e:
                    extraction_results.append({
                        "domain_id": domain_id,
                        "domain_name": domain_id,
                        "row_count": 0,
                        "error": str(e),
                    })
                    print(f"  [extract]     ✗ {domain_id}: {e}")

        # Build summary
        filing_types = sorted(set(r.get("filing_type", "") for r in extraction_results if r.get("filing_type")))
        domain_lines = []
        for r in extraction_results:
            if r.get("skipped"):
                domain_lines.append(f"- **{r['domain_name']}**: skipped ({r.get('skip_reason', '')})")
            elif r.get("error"):
                domain_lines.append(f"- **{r['domain_name']}**: ✗ error ({r['error']})")
            else:
                domain_lines.append(f"- **{r['domain_name']}**: {r['row_count']} rows")

        extracted_count = sum(1 for r in extraction_results if r.get("row_count", 0) > 0)
        skipped_count = sum(1 for r in extraction_results if r.get("skipped"))

        answer = f"✅ Extraction complete for **{company_name}** (CIK {cik}, {fund_type}).\n\n"
        answer += f"Fetched {len(ready_results)} filing(s) ({', '.join(filing_types)}), "
        answer += f"extracted **{total_rows} rows** across {extracted_count} domain(s)"
        if skipped_count:
            answer += f", {skipped_count} skipped by quality checks"
        answer += f".\n\n"
        answer += "\n".join(domain_lines)
        answer += f"\n\nData is now on the **Review** tab in the EDGAR Loader for approval."

        return {"answer": answer, "results": extraction_results}

    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to EDGAR Loader",
                "answer": f"Cannot connect to the EDGAR Loader at {loader_url}. Make sure it's running on port {loader_url.split(':')[-1]}."}
    except Exception as e:
        return {"error": str(e), "answer": f"Extraction failed: {e}"}


EDGAR_HEADERS = {"User-Agent": "EXF Financial Data Solutions dev@exf-financial.com"}


def _trigger_extraction_streaming(params):
    """Generator version of _trigger_extraction — yields progress events."""
    loader_url = CONFIG.get("app", {}).get("loader_url", "http://localhost:5070")
    cik = params.get("cik", "").strip()
    company_name = params.get("company_name", "").strip()
    domains = params.get("domains", [])
    period = params.get("period", "most_recent")

    # Company name resolution
    if not cik and company_name:
        yield {"type": "progress", "message": f"🔍 Looking up '{company_name}' in database..."}
        try:
            conn = get_conn()
            ref_schema = CONFIG["database"].get("reference_schema", "newdev_public_equity")
            with conn.cursor() as cur:
                cur.execute(f'''
                    SELECT "CIK", "PortfolioLongName" FROM "{ref_schema}"."T_PORT_PORTFOLIO"
                    WHERE "PortfolioLongName" ILIKE %s ORDER BY "PortfolioID" LIMIT 5
                ''', (f'%{company_name}%',))
                matches = cur.fetchall()
            conn.close()
            if matches and len(matches) == 1:
                cik = str(matches[0][0])
                company_name = matches[0][1]
                yield {"type": "progress", "message": f"✓ Resolved: {company_name} → CIK {cik}"}
            elif matches:
                match_list = ", ".join(f"{m[1]} (CIK {m[0]})" for m in matches[:5])
                yield {"type": "result", "answer": f"Multiple matches: {match_list}. Please specify the CIK."}
                return
            else:
                yield {"type": "result", "answer": f"No portfolio found matching '{company_name}'."}
                return
        except Exception as e:
            yield {"type": "result", "answer": f"Lookup failed: {e}"}
            return

    if not cik:
        yield {"type": "result", "answer": "I need a CIK number or company name."}
        return
    if not domains:
        yield {"type": "result", "answer": "Please specify which domains to extract."}
        return

    try:
        # Step 1: Search EDGAR
        yield {"type": "progress", "message": f"🔍 Searching EDGAR for CIK {cik}..."}
        sr = requests.get(f"{loader_url}/api/edgar/search", params={"cik": cik}, timeout=30)
        if sr.status_code != 200:
            yield {"type": "result", "answer": f"EDGAR search failed. Is the Loader running at {loader_url}?"}
            return
        sd = sr.json()
        if sd.get("error"):
            yield {"type": "result", "answer": f"EDGAR error: {sd['error']}"}
            return
        company = sd.get("company", {})
        name = company.get("name", f"CIK {cik}")
        filings = sd.get("filings", [])
        yield {"type": "progress", "message": f"✓ Found: {name} ({len(filings)} filings on EDGAR)"}

        # Step 2: Detect fund type
        fund_type = "Unknown"
        try:
            yield {"type": "progress", "message": "📋 Detecting fund type..."}
            dt = requests.post(f"{loader_url}/api/detect-fund-type", json={
                "content": "", "company_name": name,
                "entity_type": company.get("entityType", ""),
                "sic": company.get("sic", ""),
                "filing_type": "", "filing_forms": [f.get("form", "") for f in filings[:40]],
            }, timeout=15).json()
            fund_type = dt.get("fund_type", "Unknown")
            yield {"type": "progress", "message": f"✓ Fund type: {fund_type}"}
        except Exception:
            yield {"type": "progress", "message": "⚠ Fund type detection failed, using Unknown"}

        # Step 3: Build plan
        yield {"type": "progress", "message": f"📝 Building extraction plan for {len(domains)} domain(s)..."}
        pr = requests.post(f"{loader_url}/api/smart-fetch/plan", json={
            "cik": cik, "fund_type": fund_type, "domains": domains,
            "period_type": period, "year": params.get("year"), "quarter": params.get("quarter"),
        }, timeout=30)
        plan = pr.json().get("plan", []) if pr.status_code == 200 else []
        if not plan:
            yield {"type": "result", "answer": f"No filings found for {name} matching domains ({', '.join(domains)})."}
            return
        filing_types = [p['filing_type'] for p in plan]
        total_domains = sum(len(p.get('domains', [])) for p in plan)
        yield {"type": "progress", "message": f"✓ Plan: {len(plan)} filing(s) ({', '.join(filing_types)}), {total_domains} domain extraction(s)"}

        # Step 4: Fetch filings from EDGAR
        yield {"type": "progress", "message": f"📥 Fetching filing content from EDGAR..."}
        er = requests.post(f"{loader_url}/api/smart-fetch/execute", json={
            "cik": cik, "company_name": name, "plan": plan,
            "entity_type": company.get("entityType", ""),
            "sic": company.get("sic", ""),
            "sic_description": company.get("sicDescription", ""),
            "fund_type": fund_type, "deep_scan": True,
        }, timeout=120)
        if er.status_code != 200:
            yield {"type": "result", "answer": f"Filing fetch failed (HTTP {er.status_code})."}
            return
        ready_results = [r for r in er.json().get("results", []) if r.get("status") == "ready"]
        if not ready_results:
            yield {"type": "result", "answer": f"No filings could be fetched for {name}."}
            return
        total_content = sum(r.get("content_length", 0) for r in ready_results)
        yield {"type": "progress", "message": f"✓ Fetched {len(ready_results)} filing(s) ({total_content:,} chars total)"}

        # Step 5: Extract each domain
        extraction_results = []
        total_rows = 0
        for fi, filing_result in enumerate(ready_results):
            filing = filing_result.get("filing", {})
            content = filing_result.get("content", "")
            raw_content = filing_result.get("raw_content")
            filing_url = filing_result.get("filing_url", "")
            f_type = filing.get("form", filing_result.get("plan_item", {}).get("filing_type", ""))
            accession = filing.get("accessionNumber", "")
            domain_ids = filing_result.get("domains", [])
            is_nport = f_type.upper().startswith("N-PORT")

            for domain_id in domain_ids:
                if domain_id not in domains:
                    continue
                yield {"type": "progress", "message": f"📊 Extracting {domain_id} from {f_type}..."}
                try:
                    ext_resp = requests.post(f"{loader_url}/api/extract", json={
                        "domain": domain_id,
                        "content": raw_content if raw_content and is_nport else content,
                        "cik": cik, "company_name": name, "accession": accession,
                        "filing_type": f_type, "filing_url": filing_url,
                        "entity_type": company.get("entityType", ""),
                        "sic": company.get("sic", ""),
                        "sic_description": company.get("sicDescription", ""),
                        "fund_type": fund_type, "deep_scan": True,
                        "check_filing_type": True, "check_relevance": True,
                        "retry_on_empty": True, "check_thousands": True,
                    }, timeout=300)

                    if ext_resp.status_code == 200:
                        ext_data = ext_resp.json()
                        row_count = ext_data.get("row_count", 0)
                        domain_name = ext_data.get("domain", domain_id)
                        skipped = ext_data.get("skipped", False)
                        skip_reason = ext_data.get("skip_reason", "")
                        ds_info = ext_data.get("deep_scan_info", {})

                        # Report diagnostics
                        diag_parts = []
                        if ds_info.get("deep_scan"):
                            diag_parts.append(f"deep scan {ds_info.get('original_size',0):,}→{ds_info.get('full_size',0):,} chars")
                        if ds_info.get("smart_chunk"):
                            diag_parts.append(f"smart chunk {ds_info.get('chunk_from',0):,}→{ds_info.get('chunk_to',0):,} chars")
                        if ext_data.get("scale_multiplier", 1) != 1:
                            diag_parts.append(f"×{ext_data['scale_multiplier']:,} multiplier")
                        if ext_data.get("ml_corrections_applied", 0) > 0:
                            diag_parts.append(f"{ext_data['ml_corrections_applied']} ML corrections")

                        extraction_results.append({
                            "domain_id": domain_id, "domain_name": domain_name,
                            "row_count": row_count, "filing_type": f_type,
                            "skipped": skipped, "skip_reason": skip_reason,
                        })
                        total_rows += row_count

                        if skipped:
                            yield {"type": "progress", "message": f"  ⏭ {domain_name}: skipped — {skip_reason}"}
                        else:
                            diag_str = f" ({', '.join(diag_parts)})" if diag_parts else ""
                            yield {"type": "progress", "message": f"  ✓ {domain_name}: {row_count} rows{diag_str}"}
                    else:
                        extraction_results.append({"domain_id": domain_id, "domain_name": domain_id, "row_count": 0, "error": f"HTTP {ext_resp.status_code}"})
                        yield {"type": "progress", "message": f"  ✗ {domain_id}: HTTP {ext_resp.status_code}"}
                except Exception as e:
                    extraction_results.append({"domain_id": domain_id, "domain_name": domain_id, "row_count": 0, "error": str(e)})
                    yield {"type": "progress", "message": f"  ✗ {domain_id}: {str(e)[:100]}"}

        # Build summary
        f_types = sorted(set(r.get("filing_type", "") for r in extraction_results if r.get("filing_type")))
        domain_lines = []
        for r in extraction_results:
            if r.get("skipped"):
                domain_lines.append(f"- **{r['domain_name']}**: skipped ({r.get('skip_reason', '')})")
            elif r.get("error"):
                domain_lines.append(f"- **{r['domain_name']}**: ✗ error ({r['error']})")
            else:
                domain_lines.append(f"- **{r['domain_name']}**: {r['row_count']} rows")

        extracted_count = sum(1 for r in extraction_results if r.get("row_count", 0) > 0)
        skipped_count = sum(1 for r in extraction_results if r.get("skipped"))

        answer = f"✅ Extraction complete for **{name}** (CIK {cik}, {fund_type}).\n\n"
        answer += f"Fetched {len(ready_results)} filing(s) ({', '.join(f_types)}), "
        answer += f"extracted **{total_rows} rows** across {extracted_count} domain(s)"
        if skipped_count:
            answer += f", {skipped_count} skipped"
        answer += f".\n\n" + "\n".join(domain_lines)
        answer += f"\n\nData is now on the **Review** tab in the EDGAR Loader for approval."

        yield {"type": "result", "answer": answer, "results": extraction_results}

    except requests.exceptions.ConnectionError:
        yield {"type": "result", "answer": f"Cannot connect to the EDGAR Loader at {loader_url}. Make sure it's running."}
    except Exception as e:
        yield {"type": "result", "answer": f"Extraction failed: {e}"}


def _handle_edgar_search(params):
    """Search EDGAR for company filings."""
    cik = params.get("cik", "").strip()
    company_name = params.get("company_name", "").strip()
    query = params.get("query", "").strip()
    forms = params.get("forms", "")

    try:
        if query:
            # Full-text search via EDGAR EFTS
            efts_url = "https://efts.sec.gov/LATEST/search-index"
            efts_params = {"q": query, "dateRange": "custom", "startdt": "2023-01-01"}
            if forms:
                efts_params["forms"] = forms
            resp = requests.get(efts_url, params=efts_params, headers=EDGAR_HEADERS, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                hits = data.get("hits", {}).get("hits", [])
                total = data.get("hits", {}).get("total", {}).get("value", 0)
                results = []
                for h in hits[:15]:
                    src = h.get("_source", {})
                    results.append({
                        "form": src.get("root_forms", [""])[0],
                        "company": src.get("display_names", [""])[0],
                        "date": src.get("file_date", ""),
                        "accession": h.get("_id", "").split(":")[0],
                    })
                lines = "\n".join(f"- **{r['form']}** — {r['company']} — {r['date']} (Accession: `{r['accession']}`)" for r in results)
                return {"answer": f"Found **{total}** result(s) on EDGAR for \"{query}\"{' (form: ' + forms + ')' if forms else ''}:\n\n{lines}\n\n{'Showing first 15.' if total > 15 else ''}"}
            else:
                return {"answer": f"EDGAR full-text search failed (HTTP {resp.status_code})."}

        elif cik or company_name:
            # Company search via submissions API
            if not cik and company_name:
                # Try to find CIK from company name via EDGAR company search
                search_url = f"https://efts.sec.gov/LATEST/search-index?q=%22{requests.utils.quote(company_name)}%22&dateRange=custom&startdt=2020-01-01"
                resp = requests.get(search_url, headers=EDGAR_HEADERS, timeout=15)
                if resp.status_code == 200:
                    hits = resp.json().get("hits", {}).get("hits", [])
                    if hits:
                        # Extract CIK from first result
                        ciks = hits[0].get("_source", {}).get("ciks", [])
                        if ciks:
                            cik = ciks[0].lstrip("0")

            if not cik:
                return {"answer": f"Could not find CIK for '{company_name}'. Try providing the CIK directly."}

            sub_url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
            resp = requests.get(sub_url, headers=EDGAR_HEADERS, timeout=30)
            if resp.status_code != 200:
                return {"answer": f"EDGAR submissions lookup failed for CIK {cik} (HTTP {resp.status_code})."}

            data = resp.json()
            name = data.get("name", f"CIK {cik}")
            entity_type = data.get("entityType", "")
            sic = data.get("sic", "")
            sic_desc = data.get("sicDescription", "")

            recent = data.get("filings", {}).get("recent", {})
            filing_count = len(recent.get("accessionNumber", []))

            # Show recent filings (last 15)
            filings = []
            for i in range(min(15, filing_count)):
                filings.append({
                    "form": recent.get("form", [])[i] if i < len(recent.get("form", [])) else "",
                    "date": recent.get("filingDate", [])[i] if i < len(recent.get("filingDate", [])) else "",
                    "accession": recent.get("accessionNumber", [])[i] if i < len(recent.get("accessionNumber", [])) else "",
                    "doc": recent.get("primaryDocument", [])[i] if i < len(recent.get("primaryDocument", [])) else "",
                })

            lines = "\n".join(f"- **{f['form']}** — {f['date']} — `{f['accession']}`" for f in filings)
            # Count by form type
            form_counts = {}
            for i in range(filing_count):
                form = recent.get("form", [])[i] if i < len(recent.get("form", [])) else ""
                form_counts[form] = form_counts.get(form, 0) + 1
            top_forms = ", ".join(f"{k}: {v}" for k, v in sorted(form_counts.items(), key=lambda x: -x[1])[:8])

            return {
                "answer": (
                    f"**{name}** (CIK {cik})\n"
                    f"Entity: {entity_type} · SIC: {sic} ({sic_desc})\n"
                    f"Total filings: **{filing_count}** ({top_forms})\n\n"
                    f"**Recent filings:**\n{lines}"
                ),
                "rows": filings,
                "columns": ["form", "date", "accession"],
            }
        else:
            return {"answer": "Please provide a CIK, company name, or search query."}

    except Exception as e:
        return {"answer": f"EDGAR search failed: {e}"}


def _handle_edgar_filing(params):
    """Fetch and summarize a specific EDGAR filing."""
    cik = params.get("cik", "").strip()
    accession = params.get("accession", "").strip()
    filing_type = params.get("filing_type", "").strip()
    query = params.get("query", "").strip()

    if not cik:
        return {"answer": "I need a CIK number to look up filings."}

    try:
        # Find the filing
        sub_url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
        resp = requests.get(sub_url, headers=EDGAR_HEADERS, timeout=30)
        if resp.status_code != 200:
            return {"answer": f"EDGAR lookup failed for CIK {cik}."}

        data = resp.json()
        name = data.get("name", f"CIK {cik}")
        recent = data.get("filings", {}).get("recent", {})

        # Find the target filing
        target_idx = None
        for i in range(len(recent.get("accessionNumber", []))):
            if accession and recent["accessionNumber"][i] == accession:
                target_idx = i
                break
            if filing_type and recent.get("form", [])[i].upper() == filing_type.upper():
                target_idx = i
                break  # first match = most recent

        if target_idx is None:
            return {"answer": f"No {'filing ' + accession if accession else filing_type + ' filing'} found for {name}."}

        acc = recent["accessionNumber"][target_idx]
        form = recent.get("form", [])[target_idx]
        date = recent.get("filingDate", [])[target_idx]
        doc = recent.get("primaryDocument", [])[target_idx]
        acc_nodashes = acc.replace("-", "")

        # Fetch the filing content
        if doc:
            doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik.zfill(10)}/{acc_nodashes}/{doc}"
        else:
            doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik.zfill(10)}/{acc_nodashes}/{acc}-index.htm"

        print(f"  [edgar_filing] Fetching {doc_url}")
        doc_resp = requests.get(doc_url, headers=EDGAR_HEADERS, timeout=60)
        if doc_resp.status_code != 200:
            return {"answer": f"Could not fetch filing content (HTTP {doc_resp.status_code})."}

        # Strip HTML to text
        text = re.sub(r'<style[^>]*>.*?</style>', '', doc_resp.text, flags=re.DOTALL)
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'&nbsp;', ' ', text)
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&#\d+;', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        # If there's a specific query, search for relevant sections
        if query:
            # Find paragraphs containing the query terms
            terms = query.lower().split()
            sentences = text.split('. ')
            relevant = [s for s in sentences if any(t in s.lower() for t in terms)]
            if relevant:
                excerpt = '. '.join(relevant[:10])[:3000]
            else:
                excerpt = text[:3000]
        else:
            excerpt = text[:3000]

        # Use Claude to summarize
        summary_prompt = f"""Filing: {form} filed {date} by {name} (CIK {cik}, Accession {acc})
{f'User is asking about: {query}' if query else 'Provide a general summary.'}

Filing excerpt ({len(text):,} total chars):
{excerpt}

Summarize the key information from this filing{f' related to "{query}"' if query else ''}. Be specific with numbers and dates."""

        summary, s_usage = call_claude(
            [{"role": "user", "content": summary_prompt}],
            system="You are an SEC filing analyst. Summarize filings concisely, focusing on key data points.",
            max_tokens=1500,
        )

        filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik.zfill(10)}/{acc_nodashes}/{doc or acc + '-index.htm'}"

        return {
            "answer": (
                f"**{form}** — {name} — Filed {date}\n"
                f"Accession: `{acc}` · [View on EDGAR]({filing_url})\n\n"
                f"{summary}"
            ),
            "usage": s_usage,
        }

    except Exception as e:
        return {"answer": f"Filing retrieval failed: {e}"}


def _handle_web_search(params):
    """Search the web for general information."""
    query = params.get("query", "").strip()
    if not query:
        return {"answer": "Please provide a search query."}

    try:
        # Use EDGAR EFTS as primary search (SEC-focused)
        efts_url = "https://efts.sec.gov/LATEST/search-index"
        resp = requests.get(efts_url, params={"q": query, "dateRange": "custom", "startdt": "2024-01-01"},
                           headers=EDGAR_HEADERS, timeout=15)

        results_text = ""
        if resp.status_code == 200:
            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])
            total = data.get("hits", {}).get("total", {}).get("value", 0)
            if hits:
                lines = []
                for h in hits[:8]:
                    src = h.get("_source", {})
                    lines.append(f"- {src.get('display_names', [''])[0]} — {src.get('root_forms', [''])[0]} — {src.get('file_date', '')}")
                results_text = f"\n\n**Related SEC filings ({total} found):**\n" + "\n".join(lines)

        # Use Claude to provide a knowledgeable answer
        answer, usage = call_claude(
            [{"role": "user", "content": f"Answer this question about financial markets, SEC regulations, or fund industry: {query}\n\nBe specific and factual. If you're not sure, say so."}],
            system="You are a financial data expert specializing in SEC filings, BDCs, REITs, interval funds, and alternative investments. Provide factual, concise answers.",
            max_tokens=1500,
        )

        return {"answer": answer + results_text, "usage": usage}

    except Exception as e:
        return {"answer": f"Search failed: {e}"}


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """Main chatbot endpoint: NL question → SQL → results → NL answer.
    If Accept header includes text/event-stream, returns SSE with progress."""
    data = request.get_json()
    question = (data.get("question") or "").strip()
    conv_id = data.get("conversation_id", "")
    stream = "text/event-stream" in request.headers.get("Accept", "")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    if stream:
        return Response(_stream_ask(question, conv_id), content_type="text/event-stream")
    else:
        return _ask_sync(question, conv_id)


def _sse_event(data_dict):
    """Format a dict as an SSE event."""
    return f"data: {json.dumps(data_dict, default=str)}\n\n"


def _stream_ask(question, conv_id):
    """Generator for SSE streaming responses."""
    try:
        conv = load_conversation(conv_id) if conv_id else {"id": datetime.utcnow().strftime("%Y%m%d_%H%M%S"), "turns": []}

        yield _sse_event({"type": "progress", "message": "🧠 Analyzing your question..."})

        response_type, payload, sql_usage = nl_to_sql(question, conv.get("turns", []))

        if response_type == "extract":
            yield _sse_event({"type": "progress", "message": "📋 Extraction request detected — starting pipeline..."})
            # Stream extraction with progress
            result = {}
            for event in _trigger_extraction_streaming(payload):
                if event.get("type") == "progress":
                    yield _sse_event(event)
                elif event.get("type") == "result":
                    result = event
            sql = f"ACTION:EXTRACT:{json.dumps(payload, default=str)}"
            answer = result.get("answer", "Extraction triggered.")
            rows, columns, error = [], [], result.get("error")
            summary_usage = {}

        elif response_type == "edgar_search":
            yield _sse_event({"type": "progress", "message": "🔍 Searching SEC EDGAR..."})
            result = _handle_edgar_search(payload)
            sql = f"ACTION:EDGAR_SEARCH:{json.dumps(payload, default=str)}"
            answer = result.get("answer", "Search complete.")
            rows, columns = result.get("rows", []), result.get("columns", [])
            error = None
            summary_usage = result.get("usage", {})

        elif response_type == "edgar_filing":
            yield _sse_event({"type": "progress", "message": "📄 Fetching filing from EDGAR..."})
            result = _handle_edgar_filing(payload)
            sql = f"ACTION:EDGAR_FILING:{json.dumps(payload, default=str)}"
            answer = result.get("answer", "Filing retrieved.")
            rows, columns, error = [], [], None
            summary_usage = result.get("usage", {})

        elif response_type == "web_search":
            yield _sse_event({"type": "progress", "message": "🌐 Searching the web..."})
            result = _handle_web_search(payload)
            sql = f"ACTION:WEB_SEARCH:{json.dumps(payload, default=str)}"
            answer = result.get("answer", "Search complete.")
            rows, columns, error = [], [], None
            summary_usage = result.get("usage", {})

        else:
            # SQL query
            sql = payload
            yield _sse_event({"type": "progress", "message": f"🔎 Running SQL query..."})

            try:
                rows, columns = execute_sql(sql, max_rows=CONFIG["app"]["max_rows"])
                error = None
            except Exception as e:
                rows, columns, error = [], [], str(e)

            if error:
                answer = f"⚠️ Query failed: {error}\n\nSQL I tried:\n```sql\n{sql}\n```"
                summary_usage = {}
            elif not rows:
                answer = f"No results found for your question.\n\n```sql\n{sql}\n```"
                summary_usage = {}
            else:
                yield _sse_event({"type": "progress", "message": f"📊 Got {len(rows)} row(s) — summarizing..."})
                answer, summary_usage = summarize_results(question, sql, rows, columns)

        # Build final response (same format as sync)
        model_id = CONFIG["anthropic"].get("model", "claude-sonnet-4-6")
        total_input = sql_usage.get("input_tokens", 0) + summary_usage.get("input_tokens", 0)
        total_output = sql_usage.get("output_tokens", 0) + summary_usage.get("output_tokens", 0)
        sql_cost = calc_cost(sql_usage.get("input_tokens", 0), sql_usage.get("output_tokens", 0), model_id)
        summary_cost = calc_cost(summary_usage.get("input_tokens", 0), summary_usage.get("output_tokens", 0), model_id)
        total_cost = round(sql_cost + summary_cost, 6)

        turn = {
            "user": question, "assistant_sql": sql, "assistant_answer": answer,
            "row_count": len(rows), "columns": columns, "rows_preview": rows[:20],
            "timestamp": datetime.utcnow().isoformat(), "error": error, "model": model_id,
            "usage": {
                "sql_input_tokens": sql_usage.get("input_tokens", 0),
                "sql_output_tokens": sql_usage.get("output_tokens", 0),
                "summary_input_tokens": summary_usage.get("input_tokens", 0),
                "summary_output_tokens": summary_usage.get("output_tokens", 0),
                "total_input_tokens": total_input, "total_output_tokens": total_output,
                "sql_cost_usd": sql_cost, "summary_cost_usd": summary_cost, "total_cost_usd": total_cost,
            }
        }
        conv["turns"].append(turn)
        save_conversation(conv)

        conv_total_cost = sum(t.get("usage", {}).get("total_cost_usd", 0) for t in conv["turns"])

        yield _sse_event({
            "type": "done",
            "conversation_id": conv["id"],
            "question": question, "sql": sql, "answer": answer,
            "row_count": len(rows), "columns": columns, "rows": rows[:100],
            "total_rows": len(rows), "error": error, "model": model_id,
            "usage": turn["usage"],
            "conversation_totals": {
                "total_cost_usd": round(conv_total_cost, 6),
                "turn_count": len(conv["turns"]),
            }
        })

    except Exception as e:
        traceback.print_exc()
        yield _sse_event({"type": "error", "error": f"Request failed: {e}"})


def _ask_sync(question, conv_id):
    """Synchronous (non-streaming) ask handler."""
    try:
        conv = load_conversation(conv_id) if conv_id else {"id": datetime.utcnow().strftime("%Y%m%d_%H%M%S"), "turns": []}

        response_type, payload, sql_usage = nl_to_sql(question, conv.get("turns", []))

        rows = []
        columns = []
        error = None
        summary_usage = {}

        if response_type == "extract":
            result = _trigger_extraction(payload)
            sql = f"ACTION:EXTRACT:{json.dumps(payload, default=str)}"
            answer = result.get("answer", "Extraction triggered.")
            error = result.get("error")
            summary_usage = result.get("usage", {})

        elif response_type == "edgar_search":
            result = _handle_edgar_search(payload)
            sql = f"ACTION:EDGAR_SEARCH:{json.dumps(payload, default=str)}"
            answer = result.get("answer", "Search complete.")
            rows = result.get("rows", [])
            columns = result.get("columns", [])
            summary_usage = result.get("usage", {})

        elif response_type == "edgar_filing":
            result = _handle_edgar_filing(payload)
            sql = f"ACTION:EDGAR_FILING:{json.dumps(payload, default=str)}"
            answer = result.get("answer", "Filing retrieved.")
            summary_usage = result.get("usage", {})

        elif response_type == "web_search":
            result = _handle_web_search(payload)
            sql = f"ACTION:WEB_SEARCH:{json.dumps(payload, default=str)}"
            answer = result.get("answer", "Search complete.")
            summary_usage = result.get("usage", {})

        else:
            # SQL query
            sql = payload

            # Step 2: Execute SQL
            try:
                rows, columns = execute_sql(sql, max_rows=CONFIG["app"]["max_rows"])
                error = None
            except Exception as e:
                rows, columns, error = [], [], str(e)

            # Step 3: Summarize results in NL
            if error:
                answer = f"⚠️ Query failed: {error}\n\nSQL I tried:\n```sql\n{sql}\n```"
                summary_usage = {}
            elif not rows:
                answer = f"No results found for your question.\n\n```sql\n{sql}\n```"
                summary_usage = {}
            else:
                answer, summary_usage = summarize_results(question, sql, rows, columns)

        # Calculate cost
        model_id = CONFIG["anthropic"].get("model", "claude-sonnet-4-6")
        total_input = sql_usage.get("input_tokens", 0) + summary_usage.get("input_tokens", 0)
        total_output = sql_usage.get("output_tokens", 0) + summary_usage.get("output_tokens", 0)
        sql_cost = calc_cost(sql_usage.get("input_tokens", 0), sql_usage.get("output_tokens", 0), model_id)
        summary_cost = calc_cost(summary_usage.get("input_tokens", 0), summary_usage.get("output_tokens", 0), model_id)
        total_cost = round(sql_cost + summary_cost, 6)

        # Save turn to conversation
        turn = {
            "user": question,
            "assistant_sql": sql,
            "assistant_answer": answer,
            "row_count": len(rows),
            "columns": columns,
            "rows_preview": rows[:20],
            "timestamp": datetime.utcnow().isoformat(),
            "error": error,
            "model": model_id,
            "usage": {
                "sql_input_tokens": sql_usage.get("input_tokens", 0),
                "sql_output_tokens": sql_usage.get("output_tokens", 0),
                "summary_input_tokens": summary_usage.get("input_tokens", 0),
                "summary_output_tokens": summary_usage.get("output_tokens", 0),
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "sql_cost_usd": sql_cost,
                "summary_cost_usd": summary_cost,
                "total_cost_usd": total_cost,
            }
        }
        conv["turns"].append(turn)
        save_conversation(conv)

        # Compute conversation-level totals
        conv_total_cost = sum(t.get("usage", {}).get("total_cost_usd", 0) for t in conv["turns"])
        conv_total_input = sum(t.get("usage", {}).get("total_input_tokens", 0) for t in conv["turns"])
        conv_total_output = sum(t.get("usage", {}).get("total_output_tokens", 0) for t in conv["turns"])

        return jsonify({
            "conversation_id": conv["id"],
            "question": question,
            "sql": sql,
            "answer": answer,
            "row_count": len(rows),
            "columns": columns,
            "rows": rows[:100],  # limit payload
            "total_rows": len(rows),
            "error": error,
            "model": model_id,
            "usage": turn["usage"],
            "conversation_totals": {
                "total_cost_usd": round(conv_total_cost, 6),
                "total_input_tokens": conv_total_input,
                "total_output_tokens": conv_total_output,
                "turn_count": len(conv["turns"]),
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Request failed: {e}"}), 500


@app.route("/api/conversations")
def api_conversations():
    """List past conversations."""
    convs = []
    for path in sorted(CONVERSATIONS_DIR.glob("*.json"), reverse=True)[:50]:
        try:
            with open(path) as f:
                c = json.load(f)
            convs.append({
                "id": c["id"],
                "created_at": c.get("created_at", ""),
                "turn_count": len(c.get("turns", [])),
                "first_question": c["turns"][0]["user"] if c.get("turns") else "",
            })
        except Exception:
            continue
    return jsonify(convs)


@app.route("/api/conversations/<conv_id>")
def api_conversation_get(conv_id):
    return jsonify(load_conversation(conv_id))


@app.route("/api/conversations/<conv_id>", methods=["DELETE"])
def api_conversation_delete(conv_id):
    path = CONVERSATIONS_DIR / f"{conv_id}.json"
    if path.exists():
        path.unlink()
    return jsonify({"deleted": True})


@app.route("/api/suggestions")
def api_suggestions():
    """Return example questions users can ask."""
    return jsonify({
        "suggestions": [
            {"category": "Filing History", "questions": [
                "What's the most recent 10-K we've extracted for CIK 1869453?",
                "Show me all filings from CIK 1869453 in 2025",
                "How many 10-K filings have we extracted in total?",
                "What filing types have we processed for PIMCO?",
            ]},
            {"category": "NAV & Performance", "questions": [
                "What's the latest NAV for CIK 1869453?",
                "Show me NAV history for all share classes of CIK 1869453",
                "Which funds have NAV above $10?",
                "What's the 1-year return for CIK 1869453's share classes?",
            ]},
            {"category": "Distributions", "questions": [
                "Show me all distributions paid by CIK 1869453 in 2025",
                "What's the total distributions YTD for this portfolio?",
                "Which funds paid a special distribution last quarter?",
            ]},
            {"category": "Extraction Analytics", "questions": [
                "How many extractions have we run this month?",
                "What's the total cost of all extractions?",
                "Show me extractions with errors",
                "Which model do we use most often?",
                "How many corrections have reviewers logged?",
            ]},
            {"category": "Composition", "questions": [
                "What's the asset allocation for CIK 1869453?",
                "Which funds have more than 20% in real estate?",
                "Show geographic composition of all BDCs",
            ]},
            {"category": "Extract New Data", "questions": [
                "Extract returns for CIK 1920145 from the most recent filings",
                "Pull NAV and distribution data for CIK 1869453",
                "Fetch composition data for CIK 1803498 from 2024",
                "Extract all domains for CIK 1920145",
            ]},
        ]
    })


@app.route("/api/settings", methods=["GET"])
def api_settings_get():
    """Return current settings (password masked)."""
    cfg = load_config()
    db = dict(cfg.get("database", {}))
    # Mask password — show whether it's set, not its value
    db["password"] = "••••••••" if db.get("password") else ""
    anthropic = dict(cfg.get("anthropic", {}))
    anthropic["api_key"] = "••••••••" if anthropic.get("api_key") else ""
    return jsonify({
        "database": db,
        "anthropic": anthropic,
        "app": cfg.get("app", {}),
        "models": cfg.get("models", [
            {"id": "claude-opus-4-6", "name": "Claude Opus 4.6"},
            {"id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6"},
            {"id": "claude-haiku-4-5-20251001", "name": "Claude Haiku 4.5"},
        ]),
    })


@app.route("/api/settings", methods=["POST"])
def api_settings_post():
    """Update settings. Empty/masked values are preserved from existing config."""
    global CONFIG
    data = request.get_json() or {}
    cfg = load_config()

    # Database
    if "database" in data:
        db_in = data["database"]
        db_cur = cfg.setdefault("database", {})
        for key in ("host", "port", "dbname", "user", "schema", "reference_schema", "sslmode"):
            if key in db_in and db_in[key] != "":
                db_cur[key] = db_in[key] if key != "port" else int(db_in[key] or 16149)
        # Password: only update if non-masked value provided
        if "password" in db_in:
            pw = db_in["password"]
            print(f"  [settings] Password field received: {'[empty]' if not pw else '••••' if pw == '••••••••' else f'[{len(pw)} chars]'}")
            if pw and pw != "••••••••":
                db_cur["password"] = pw
                print(f"  [settings] Password SAVED ({len(pw)} chars)")
            else:
                print(f"  [settings] Password NOT updated (masked or empty)")
        else:
            print(f"  [settings] No password field in request")

    # Anthropic
    if "anthropic" in data:
        a_in = data["anthropic"]
        a_cur = cfg.setdefault("anthropic", {})
        if "model" in a_in and a_in["model"]:
            a_cur["model"] = a_in["model"]
        if "api_key" in a_in:
            key = a_in["api_key"]
            if key and key != "••••••••":
                a_cur["api_key"] = key

    # App settings
    if "app" in data:
        app_in = data["app"]
        app_cur = cfg.setdefault("app", {})
        if "max_rows" in app_in:
            app_cur["max_rows"] = int(app_in["max_rows"] or 500)
        if "conversation_limit" in app_in:
            app_cur["conversation_limit"] = int(app_in["conversation_limit"] or 20)
        if "loader_url" in app_in and app_in["loader_url"]:
            app_cur["loader_url"] = app_in["loader_url"]

    # Models list — allow adding/removing/editing available models
    if "models" in data and isinstance(data["models"], list):
        cfg["models"] = data["models"]

    save_config(cfg)
    CONFIG = cfg  # reload in-memory copy
    print(f"  [settings] Config saved. DB password in CONFIG: {'YES (' + str(len(CONFIG['database'].get('password', ''))) + ' chars)' if CONFIG['database'].get('password') else 'NO/EMPTY'}")
    # Invalidate schema cache so next query re-introspects with new DB settings
    _SCHEMA_CACHE["context"] = None
    _SCHEMA_CACHE["updated"] = None
    _KNOWN_TABLES_CACHE["tables"] = None
    return jsonify({"success": True})


@app.route("/api/schema/refresh", methods=["POST"])
def api_schema_refresh():
    """Force a refresh of the schema context from the database."""
    try:
        _KNOWN_TABLES_CACHE["tables"] = None
        _SCHEMA_CACHE["context"] = None
        ctx = get_schema_context(force_refresh=True)

        # Verify it worked
        if "SCHEMA_INTROSPECTION_FAILED" in ctx or "WARNING: No tables found" in ctx:
            return jsonify({
                "success": False,
                "error": ctx,
                "current_db": CONFIG["database"].get("dbname"),
                "current_schema": CONFIG["database"].get("schema"),
                "current_ref_schema": CONFIG["database"].get("reference_schema"),
            }), 400

        # Extract table names for display
        table_names = re.findall(r'## "([^"]+)"\."([^"]+)"', ctx)

        return jsonify({
            "success": True,
            "chars": len(ctx),
            "updated": _SCHEMA_CACHE["updated"],
            "table_count": len(table_names),
            "tables": [f"{s}.{t}" for s, t in table_names],
            "current_db": CONFIG["database"].get("dbname"),
            "current_schema": CONFIG["database"].get("schema"),
            "current_ref_schema": CONFIG["database"].get("reference_schema"),
            "preview": ctx[:800] + ("..." if len(ctx) > 800 else ""),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/schema")
def api_schema_get():
    """Return the current schema context (for inspection/debugging)."""
    return jsonify({
        "context": get_schema_context(),
        "updated": _SCHEMA_CACHE["updated"],
        "chars": len(_SCHEMA_CACHE["context"] or ""),
    })


@app.route("/api/schema/diagnose")
def api_schema_diagnose():
    """List ALL schemas and tables so the user can see what's actually in the DB."""
    try:
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                # Confirm WHICH database we're actually connected to
                cur.execute("SELECT current_database(), current_user, version()")
                actual_db, actual_user, version = cur.fetchone()

                # All schemas
                cur.execute("""
                    SELECT schema_name FROM information_schema.schemata
                    WHERE schema_name NOT IN ('pg_catalog','information_schema','pg_toast','pg_temp_1','pg_toast_temp_1')
                    ORDER BY schema_name
                """)
                schemas = [r[0] for r in cur.fetchall()]

                # Tables per schema
                cur.execute("""
                    SELECT table_schema, table_name
                    FROM information_schema.tables
                    WHERE table_schema NOT IN ('pg_catalog','information_schema')
                    ORDER BY table_schema, table_name
                """)
                tables_by_schema = {}
                for schema, table in cur.fetchall():
                    tables_by_schema.setdefault(schema, []).append(table)

                pe_schema = CONFIG["database"].get("schema", "")
                ref_schema = CONFIG["database"].get("reference_schema", "")
                configured_db = CONFIG["database"].get("dbname", "")

            return jsonify({
                "configured_dbname": configured_db,
                "actual_dbname": actual_db,
                "actual_user": actual_user,
                "db_matches_config": actual_db == configured_db,
                "all_schemas": schemas,
                "configured_pe_schema": pe_schema,
                "configured_ref_schema": ref_schema,
                "pe_schema_exists": pe_schema in schemas,
                "ref_schema_exists": ref_schema in schemas,
                "tables_by_schema": tables_by_schema,
                "table_counts": {s: len(t) for s, t in tables_by_schema.items()},
            })
        finally:
            conn.close()
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/settings/test-db", methods=["POST"])
def api_settings_test_db():
    """Test database connection with optionally-overridden credentials."""
    data = request.get_json() or {}
    try:
        db = data.get("database") or CONFIG["database"]
        # If password is masked, use the stored one
        pw = db.get("password", "")
        if pw == "••••••••" or not pw:
            pw = CONFIG["database"].get("password", "")
        conn = psycopg2.connect(
            host=db["host"], port=int(db.get("port", 16149)),
            dbname=db["dbname"], user=db["user"],
            password=pw, sslmode=db.get("sslmode", "require"),
            connect_timeout=10,
        )
        with conn.cursor() as cur:
            cur.execute("SELECT current_database(), current_user, version()")
            db_name, db_user, version = cur.fetchone()
        conn.close()
        return jsonify({
            "success": True,
            "database": db_name,
            "user": db_user,
            "version": version.split(",")[0],
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/settings/test-api", methods=["POST"])
def api_settings_test_api():
    """Test the Anthropic API key."""
    data = request.get_json() or {}
    try:
        anthropic = data.get("anthropic") or {}
        api_key = anthropic.get("api_key", "")
        if api_key == "••••••••" or not api_key:
            api_key = CONFIG["anthropic"].get("api_key", "")
        if not api_key:
            return jsonify({"success": False, "error": "No API key configured"}), 400

        model = anthropic.get("model") or CONFIG["anthropic"].get("model", "claude-sonnet-4-6")
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": 20,
            "messages": [{"role": "user", "content": "Reply with just 'ok'"}],
        }
        resp = requests.post("https://api.anthropic.com/v1/messages",
                             json=payload, headers=headers, timeout=20)
        if resp.status_code == 200:
            return jsonify({"success": True, "model": model})
        else:
            err_data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            msg = err_data.get("error", {}).get("message", resp.text[:200])
            return jsonify({"success": False, "error": f"HTTP {resp.status_code}: {msg}"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/stats")
def api_stats():
    """Return aggregate usage statistics across all conversations."""
    total_cost = 0.0
    total_input = 0
    total_output = 0
    total_turns = 0
    total_convs = 0
    cost_by_model = {}

    for path in CONVERSATIONS_DIR.glob("*.json"):
        try:
            with open(path) as f:
                conv = json.load(f)
            total_convs += 1
            for turn in conv.get("turns", []):
                total_turns += 1
                usage = turn.get("usage", {})
                cost = usage.get("total_cost_usd", 0)
                total_cost += cost
                total_input += usage.get("total_input_tokens", 0)
                total_output += usage.get("total_output_tokens", 0)
                model = turn.get("model", "unknown")
                cost_by_model[model] = cost_by_model.get(model, 0) + cost
        except Exception:
            continue

    return jsonify({
        "total_cost_usd": round(total_cost, 6),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_turns": total_turns,
        "total_conversations": total_convs,
        "cost_by_model": {m: round(c, 6) for m, c in cost_by_model.items()},
    })


@app.route("/api/health")
def api_health():
    try:
        conn = get_conn()
        conn.close()
        db_ok = True
    except Exception as e:
        db_ok = False

    return jsonify({
        "status": "ok",
        "db": "connected" if db_ok else "disconnected",
        "has_api_key": bool(CONFIG["anthropic"].get("api_key")),
        "current_model": CONFIG["anthropic"].get("model", ""),
    })


if __name__ == "__main__":
    port = CONFIG["app"].get("port", 5100)
    print(f"\n  EDGAR DATA CHATBOT")
    print(f"  ─────────────────")
    print(f"  Running on http://localhost:{port}")
    print(f"  Schema: {CONFIG['database']['schema']}")
    print(f"  Model:  {CONFIG['anthropic']['model']}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
