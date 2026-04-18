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
        raise RuntimeError("Database password not configured. Open Settings to configure.")
    print(f"[db] Connecting to {db['host']}:{db.get('port', 16149)}/{db['dbname']} as {db['user']}")
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
    """Translate a natural language question into SQL — or detect extraction intent.
    Returns (response_type, payload, usage) where:
    - response_type='sql': payload is the SQL query string
    - response_type='extract': payload is a dict with extraction parameters
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

# EXTRACTION CAPABILITY
You can trigger new data extractions from SEC EDGAR filings. If the user asks to
"extract", "fetch", "pull", "get new data", "load data from EDGAR", or similar action
verbs that imply getting NEW data from filings (not querying existing data), respond with
a JSON action instead of SQL.

Format for extraction requests:
ACTION:EXTRACT:{"cik": "0001920145", "domains": ["returns", "nav_pricing"], "period": "most_recent"}

You can also specify company_name instead of CIK — the system will look it up:
ACTION:EXTRACT:{"company_name": "Blue Owl", "domains": ["returns"], "period": "most_recent"}

For time-specific extractions:
ACTION:EXTRACT:{"cik": "0001920145", "domains": ["distributions"], "period": "quarter", "year": 2024, "quarter": 3}
ACTION:EXTRACT:{"cik": "0001920145", "domains": ["returns", "fees"], "period": "annual", "year": 2024}

Available domains and what they extract:
- filing_master → Fund regulatory filing metadata
- nav_pricing → NAV per share, offering/repurchase prices
- shares_outstanding → Shares outstanding per class
- volatility → Volatility & risk metrics
- distributions → Distribution history per share class
- dist_ytd/dist_drip/dist_metrics/dist_tax → YTD totals, DRIP, metrics, tax character
- composition/comp_objectives → Portfolio composition by asset type, geography, industry
- leverage/leverage_summary/leverage_covenant → Debt facilities, ratios, covenant compliance
- returns → Total return, period performance, benchmark comparison
- fees → Management fees, expense ratios
- repurchase_fees/investor_eligibility → Share class eligibility, repurchase fees
- offering_price → Offering/repurchase prices per class
- redemptions/liquidation_program/share_conversion → Liquidity programs
- death_disability/sc_liquidity → Death/disability provisions, share class liquidity
- interval_detail/interval_next_dates/interval_gates/interval_suspension/interval_early_withdrawal → Interval fund specifics
- tender_program/tender_suspension/tender_next_dates → Tender offer specifics
- account_metrics → Account-level metrics
- operational_details → Service providers, board details

Period options: "most_recent" (default), "quarter" (needs year+quarter), "annual" (needs year)

DECISION RULES:
- "Show me NAV for CIK 1234" → SQL query (querying EXISTING data)
- "Extract NAV for CIK 1234" → Extraction action (fetching NEW data from EDGAR)
- "What's the latest distribution?" → SQL query
- "Pull fresh distribution data from EDGAR for Blue Owl" → Extraction action
- "Compare returns across all BDCs" → SQL query (analyzing existing data)
- "Get returns data for CIK 1234 from the 2024 10-K" → Extraction action

COMMON QUERY PATTERNS:
- To find a company: JOIN with the reference portfolio table on "PortfolioID" which has "CompanyName" and "CIK"
- NAV data: T_PE_FUND_SHARE_CLASS_NAV_PRICING has "NAVPS", "ReportDate"
- Returns: T_PE_FUND_SHARE_CLASS_RETURNS has "Fund1Year", "FundYTD", "FundSinceInception"
- Distributions: T_PE_FUND_SHARE_CLASS_DISTRIBUTION_HISTORY has "DistributionPerShare", "RecordDate"
- Leverage: T_PE_FUND_LEVERAGE_DETAIL has facility-level debt; T_INST_PE_FUND_LEVERAGE_SUMMARY has totals
- Extraction log: T_PE_FUND_EXTRACTION_LOG tracks all extractions with timestamps and costs

For SQL queries: respond with ONLY the SQL query, starting with SELECT.
For extraction actions: respond with ONLY the ACTION:EXTRACT:{json} line."""

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
            f"If this is a DATA QUERY (show, list, compare existing data): return ONLY a SELECT SQL query.\n"
            f"If this is an EXTRACTION REQUEST (extract, fetch, pull new data from EDGAR): return ACTION:EXTRACT:{{json}}.\n"
            f"Remember: every table MUST be referenced with its exact schema name, fully double-quoted."
        )
    })

    total_usage = {"input_tokens": 0, "output_tokens": 0}
    response = ""

    for attempt in range(2):
        response, usage = call_claude(messages, system=schema_ctx + extraction_ctx)
        total_usage["input_tokens"] += usage.get("input_tokens", 0)
        total_usage["output_tokens"] += usage.get("output_tokens", 0)

        # Check if this is an extraction action
        if response.strip().startswith("ACTION:EXTRACT:"):
            json_str = response.strip()[len("ACTION:EXTRACT:"):]
            try:
                extract_params = json.loads(json_str)
                return "extract", extract_params, total_usage
            except json.JSONDecodeError:
                pass

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
                    SELECT "CIK", "CompanyName" FROM "{ref_schema}"."T_PORT_PORTFOLIO"
                    WHERE "CompanyName" ILIKE %s
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

        # Step 3: Execute the plan
        exec_resp = requests.post(f"{loader_url}/api/smart-fetch/execute", json={
            "cik": cik,
            "company_name": company_name,
            "plan": plan,
            "entity_type": company.get("entityType", ""),
            "sic": company.get("sic", ""),
            "sic_description": company.get("sicDescription", ""),
        }, timeout=300)

        if exec_resp.status_code != 200:
            return {"error": f"Execution failed: HTTP {exec_resp.status_code}",
                    "answer": f"Extraction plan was built ({filing_summary}) but execution failed."}

        exec_data = exec_resp.json()
        results = exec_data.get("results", [])

        # Build summary
        total_rows = sum(r.get("row_count", 0) for r in results)
        domain_summary = "\n".join(f"- **{r.get('domain_name', r.get('domain_id', '?'))}**: {r.get('row_count', 0)} rows" for r in results)

        answer = (
            f"✅ Extraction complete for **{company_name}** (CIK {cik}, {fund_type}).\n\n"
            f"Fetched {len(plan)} filing(s) ({filing_summary}), extracted {len(results)} domain(s) with **{total_rows} total rows**.\n\n"
            f"{domain_summary}\n\n"
            f"The data is now on the **Review** tab in the EDGAR Loader for approval."
        )

        return {"answer": answer, "results": results}

    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to EDGAR Loader",
                "answer": f"Cannot connect to the EDGAR Loader at {loader_url}. Make sure it's running on port {loader_url.split(':')[-1]}."}
    except Exception as e:
        return {"error": str(e), "answer": f"Extraction failed: {e}"}


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """Main chatbot endpoint: NL question → SQL → results → NL answer."""
    data = request.get_json()
    question = (data.get("question") or "").strip()
    conv_id = data.get("conversation_id", "")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        # Load conversation history
        conv = load_conversation(conv_id) if conv_id else {"id": datetime.utcnow().strftime("%Y%m%d_%H%M%S"), "turns": []}

        # Step 1: Translate NL → SQL or detect extraction intent
        response_type, payload, sql_usage = nl_to_sql(question, conv.get("turns", []))

        if response_type == "extract":
            # Handle extraction request — call the EDGAR Loader API
            extract_result = _trigger_extraction(payload)
            sql = f"ACTION:EXTRACT:{json.dumps(payload, default=str)}"
            answer = extract_result.get("answer", "Extraction triggered.")
            rows = []
            columns = []
            error = extract_result.get("error")
            summary_usage = {}
        else:
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
            if pw and pw != "••••••••":
                db_cur["password"] = pw

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
