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
        "app": {
            "port": 5100,
            "max_rows": 500,
            "conversation_limit": 20,  # how many past turns to keep
        },
    }

def save_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

CONFIG = load_config()
if not CONFIG.get("database", {}).get("password"):
    CONFIG["database"]["password"] = os.environ.get("EDGAR_DB_PASSWORD", "") or getpass.getpass("DB password: ")
    save_config(CONFIG)


def get_conn():
    db = CONFIG["database"]
    return psycopg2.connect(
        host=db["host"], port=db["port"],
        dbname=db["dbname"], user=db["user"],
        password=db["password"], sslmode=db.get("sslmode", "require"),
    )


# ──────────────────────────────────────────────────────────────────
# SCHEMA CONTEXT — fed to Claude so it knows what to query
# ──────────────────────────────────────────────────────────────────

SCHEMA_CONTEXT = """
You are a SQL assistant for a PostgreSQL database containing data extracted from SEC EDGAR filings.

# SCHEMAS
- `newdev_private_equity` — extracted fund data (34 tables)
- `newdev_public_equity` — reference tables

IMPORTANT: PostgreSQL requires mixed-case schema/table/column names to be DOUBLE-QUOTED separately.
Correct:   SELECT * FROM "newdev_private_equity"."T_PE_FUND_SHARE_CLASS_NAV_PRICING"
Wrong:     SELECT * FROM newdev_private_equity.T_PE_FUND_SHARE_CLASS_NAV_PRICING

# REFERENCE TABLES
`newdev_public_equity"."T_PORT_PORTFOLIO` — fund master
  Columns: PortfolioID (BIGINT PK), PortfolioName, CIK, FundTypeCode, FundSubTypeCode, CreatedAt

`newdev_public_equity"."T_PORT_SHARE_CLASS` — share class master
  Columns: ShareClassID (BIGINT PK), PortfolioID (FK), ShareClassName, ShareClassCode

# FILING MASTER
`newdev_private_equity"."T_PE_FUND_REGULATORY_FILING_MASTER` — all processed filings
  Columns: RegulatoryFilingID (BIGINT PK), PortfolioID, FilingType, FilingDate, ReportPeriodEndDate,
           FilingURL, CIK, AccessionID, RegistrantName, EntityType, SICCode, SYSTEM_INSERTED

# EXTRACTION LOG
`newdev_private_equity"."T_PE_FUND_EXTRACTION_LOG` — every extraction attempt
  Columns: ExtractionID, CIK, CompanyName, AccessionID, DomainID, DomainName, TargetTable,
           FilingType, Model, ExtractionMethod, Status (APPROVED/REJECTED/PENDING_REVIEW),
           RowCount, InputTokens, OutputTokens, CostUSD, FilingURL, FilingDate, ExtractedAt

# ML CORRECTIONS
`newdev_private_equity"."T_PE_EXTRACTION_CORRECTIONS` — reviewer corrections
  Columns: CorrectionID, ExtractionID, DomainID, FieldName, OriginalValue, CorrectedValue,
           CIK, CompanyName, AccessionID, FilingType, CorrectedAt

# DOMAIN DATA TABLES (34)
All domain tables have CIK, AccessionID, Source, SYSTEM_INSERTED/UPDATED/CHANGEBY columns.

## NAV & Pricing
`T_PE_FUND_SHARE_CLASS_NAV_PRICING` (PK: ShareClassID, NAVDate, PortfolioID)
  NAVDate, NAVPerShare, TotalNetAssets, TotalGrossAssets, SharesOutstanding,
  OfferingPrice, RepurchasePrice, PremiumDiscount, NAVChangePercent

`T_PE_FUND_SHARE_CLASS_OFFERING_PRICE` (PK: ShareClassID, PricingDate, PortfolioID)
  PricingDate, OfferingPrice, PricingBasis (Fixed/NAV/NAV+Load), IncludesLoad, LoadPct

## Distributions
`T_PE_FUND_SHARE_CLASS_DISTRIBUTION` (PK: ShareClassID, PortfolioID, PaymentDate, Type)
  PaymentDate, RecordDate, DeclarationDate, Type (Regular/Special/Return of Capital),
  DistributionPerShare, DistributionAmount, Frequency, SpecialIndicator

`T_PE_FUND_SHARE_CLASS_DISTRIBUTIONS_YTD` (PK: ShareClassID, PortfolioID, PeriodEndDate)
  PeriodEndDate, TotalDistributionsYTD, DistributionsFromIncome, DistributionsFromCapitalGains,
  DistributionsFromReturnOfCapital

`T_PE_FUND_SHARE_CLASS_DISTRIBUTIONS_DRIP_SCHEDULE` — reinvestment plan details
`T_PE_FUND_SHARE_CLASS_DISTRIBUTION_METRICS` — annualized rate, yield, coverage
`T_PE_FUND_DISTRIBUTIONS_TAX` — tax character (ordinary/capital gains/ROC)

## Composition
`T_PE_FUND_COMPOSITION_ACTUALS` (PK: PortfolioID, ReportDate, NodeType, NodeName)
  ReportDate, NodeType (Asset Class/Industry/Geography/Security Type),
  NodeLevel, SubNodeLevel, NodeName, AllocationPercent, FairValue, Cost, NumberOfPositions

`T_PE_FUND_COMPOSITION_OBJECTIVES` — target/policy allocations

## Leverage
`T_PE_FUND_LEVERAGE_DETAIL` (PK: PortfolioID, FacilityName, ReportDate)
  FacilityName, FacilityType, ReportDate, CommittedAmount, DrawnAmount, AvailableAmount,
  InterestRate, MaturityDate, Lender

`T_INST_PE_FUND_LEVERAGE_SUMMARY` — aggregate borrowings, asset coverage
`T_PE_FUND_LEVERAGE_COVENANT_COMPLIANCE` — covenant tests and ratios

## Performance
`T_PE_FUND_SHARE_CLASS_RETURNS` (PK: ShareClassID, PortfolioID, ReportDate)
  ReportDate, Return1Month, Return3Month, ReturnYTD, Return1Year, Return3Year,
  Return5Year, Return10Year, ReturnSinceInception, InceptionDate, BenchmarkName, BenchmarkReturn1Year

`T_PE_FUND_SHARE_CLASS_VOLATILITY` — standard deviation, Sharpe, beta, alpha

## Fees
`T_PE_FUND_FEES` (PK: ShareClassID, PortfolioID, EffectiveDate)
  EffectiveDate, ManagementFeePercent, IncentiveFeePercent, AdminFeePercent,
  TotalExpenseRatio, ExpenseCap, WaiverEndDate, PerformanceFee

## Shares Outstanding
`T_PE_FUND_SHARE_CLASS_SHARES_OUTSTANDING` (PK: ShareClassID, PortfolioID, ReportDate)
  ReportDate, SharesOutstanding, SharesAuthorized, SharesIssued

## Operational
`T_PE_FUND_MASTER_OPERATIONAL_DETAILS` (PK: PortfolioID)
  InceptionDate, FiscalYearEnd, FundManager, SubAdviser, Custodian, Auditor,
  TransferAgent, LegalCounsel, Administrator, Distributor, DomicileCountry

## Liquidity
`T_PE_FUND_REDEMPTIONS` — repurchase/redemption programs
`T_PE_FUND_LIQUIDATION_PROGRAM` — wind-down plans
`T_PE_FUND_DEATH_DISABILITY` — death/disability benefit provisions
`T_PE_FUND_SHARE_CONVERSION_PROGRAM` — class conversion features
`T_PE_FUND_SHARE_CLASS_LIQUIDITY` — share class liquidity events

## Interval Fund (for interval funds)
`T_PE_FUND_INTERVAL_FUND_DETAIL`, `T_PE_FUND_INTERVAL_FUND_NEXT_DATES`,
`T_PE_FUND_INTERVAL_FUND_GATE_PROVISIONS`, `T_PE_FUND_INTERVAL_FUND_SUSPENSION_FRAMEWORK`,
`T_PE_FUND_SHARE_CLASS_INTERVAL_FUND_EARLY_WITHDRAWAL`

## Tender Offer (for tender offer funds)
`T_PE_FUND_TENDER_OFFER_PROGRAM`, `T_PE_FUND_TENDER_OFFER_FUND_NEXT_DATES`,
`T_PE_FUND_TENDER_OFFER_SUSPENSION_FRAMEWORK`

## Other
`T_PE_FUND_REPURCHASE_FEES` — early redemption penalties
`T_PE_FUND_SHARE_CLASS_INVESTOR_ELIGIBILITY` — accredited investor requirements
`T_PE_FUND_SHARE_CLASS_ACCOUNT_METRICS` — number of accounts/shareholders

# QUERY PATTERNS

## "Most recent 10-K for CIK X"
SELECT * FROM "newdev_private_equity"."T_PE_FUND_REGULATORY_FILING_MASTER"
WHERE "CIK" = 'X' AND "FilingType" = '10-K'
ORDER BY "FilingDate" DESC LIMIT 1

## "Latest NAV for fund X"
SELECT sc."ShareClassName", nav."NAVDate", nav."NAVPerShare"
FROM "newdev_private_equity"."T_PE_FUND_SHARE_CLASS_NAV_PRICING" nav
JOIN "newdev_public_equity"."T_PORT_SHARE_CLASS" sc ON sc."ShareClassID" = nav."ShareClassID"
JOIN "newdev_public_equity"."T_PORT_PORTFOLIO" p ON p."PortfolioID" = nav."PortfolioID"
WHERE p."CIK" = 'X'
ORDER BY nav."NAVDate" DESC

## "How many extractions have we run?"
SELECT COUNT(*), "Status" FROM "newdev_private_equity"."T_PE_FUND_EXTRACTION_LOG"
GROUP BY "Status"

# SQL GUIDELINES
1. ALWAYS use double-quoted identifiers for schema, table, and column names.
2. Filter by CIK when searching for a specific company — CIKs are stored with leading zeros stripped.
3. LIMIT all queries to 500 rows maximum.
4. Join via PortfolioID or ShareClassID — don't join on CIK alone (duplicates possible).
5. Dates are DATE type, not TIMESTAMP. Use 'YYYY-MM-DD' literals.
6. For "most recent", use ORDER BY date DESC LIMIT 1.
7. Return ONLY the SQL query — no explanation, no markdown, no backticks.
"""


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


def nl_to_sql(question, conversation_history=None):
    """Translate a natural language question into SQL using Claude."""
    history = conversation_history or []

    # Build messages including prior turns
    messages = []
    for turn in history[-10:]:  # last 10 turns for context
        if turn.get("user"):
            messages.append({"role": "user", "content": turn["user"]})
        if turn.get("assistant_sql"):
            messages.append({"role": "assistant", "content": turn["assistant_sql"]})

    messages.append({
        "role": "user",
        "content": f"Generate a PostgreSQL query to answer this question: {question}\n\nReturn ONLY the SQL query. Start with SELECT. No backticks, no explanation."
    })

    sql, usage = call_claude(messages, system=SCHEMA_CONTEXT)

    # Clean up the SQL
    sql = re.sub(r"^```(?:sql)?\s*", "", sql)
    sql = re.sub(r"\s*```$", "", sql)
    sql = sql.strip()

    return sql, usage


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

    # Block dangerous keywords
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
                 "CREATE", "GRANT", "REVOKE", "COPY", "--", "/*"]
    for kw in forbidden:
        if re.search(rf"\b{kw}\b", sql, re.IGNORECASE):
            raise ValueError(f"Forbidden keyword in SQL: {kw}")

    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Wrap in a LIMIT if there isn't one
            if "LIMIT" not in sql.upper():
                sql_with_limit = sql.rstrip(";") + f" LIMIT {max_rows}"
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

        # Step 1: Translate NL → SQL
        sql, sql_usage = nl_to_sql(question, conv.get("turns", []))

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
            "usage": {
                "sql_input_tokens": sql_usage.get("input_tokens", 0),
                "sql_output_tokens": sql_usage.get("output_tokens", 0),
                "summary_input_tokens": summary_usage.get("input_tokens", 0),
                "summary_output_tokens": summary_usage.get("output_tokens", 0),
            }
        }
        conv["turns"].append(turn)
        save_conversation(conv)

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
            "usage": turn["usage"],
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
        ]
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
    })


if __name__ == "__main__":
    port = CONFIG["app"].get("port", 5100)
    print(f"\n  EDGAR DATA CHATBOT")
    print(f"  ─────────────────")
    print(f"  Running on http://localhost:{port}")
    print(f"  Schema: {CONFIG['database']['schema']}")
    print(f"  Model:  {CONFIG['anthropic']['model']}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
