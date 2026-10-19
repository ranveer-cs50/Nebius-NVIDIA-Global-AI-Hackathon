
"""
Adaptive Research Agent
Tavily + Nebius-hosted NVIDIA Nemotron + temporary SQLite memory

Install:
    pip install openai requests python-dotenv beautifulsoup4

Required .env:
    TAVILY_API_KEY=your_tavily_key
    NEBIUS_API_KEY=your_nebius_key
    NEBIUS_MODEL=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B

Run:
    python research_agent.py
"""

import os
import re
import json
import time
import hashlib
import sqlite3
from datetime import datetime, timezone
from urllib.parse import urlparse, urldefrag, parse_qsl, urlencode, urlunparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# 1. CONFIGURATION
# ============================================================


load_dotenv()

TAVILY_KEY = os.getenv("TAVILY_API_KEY", "").strip()
NEBIUS_KEY = os.getenv("NEBIUS_API_KEY", "").strip()

if not TAVILY_KEY or not NEBIUS_KEY:
    raise ValueError("Missing TAVILY_API_KEY or NEBIUS_API_KEY in your .env file.")

BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"

DB_PATH = "research_memory.sqlite3"


MAX_ROUNDS = int(os.getenv("MAX_ROUNDS", "5"))
MAX_TOTAL_QUERIES = int(os.getenv("MAX_TOTAL_QUERIES", "20"))
MAX_QUERIES_PER_ROUND = int(os.getenv("MAX_QUERIES_PER_ROUND", "4"))
RESULTS_PER_QUERY = int(os.getenv("RESULTS_PER_QUERY", "6"))
MAX_PAGES = int(os.getenv("MAX_PAGES", "30"))
MAX_PAGE_CHARS = int(os.getenv("MAX_PAGE_CHARS", "10000"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "22000"))
REQUEST_TIMEOUT = 20

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"

if not TAVILY_KEY or not NEBIUS_KEY:
    raise SystemExit(
        "Missing API keys. Set TAVILY_API_KEY and NEBIUS_API_KEY in .env."
    )

client = OpenAI(
    base_url=BASE_URL,
    api_key=NEBIUS_KEY,
)

http = requests.Session()
http.headers.update({
    "User-Agent": "AdaptiveResearchAgent/1.0"
})


# ============================================================
# 2. GENERAL HELPERS
# ============================================================

def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def digest(text):
    """Stable hash for exact deduplication."""
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def clean_text(text):
    text = (text or "").replace("\x00", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def canonical_url(url):
    """Normalize URLs and remove common tracking parameters."""
    url = (url or "").strip()

    if not url:
        return ""

    url, _ = urldefrag(url)
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return ""

    tracking = {
        "utm_source", "utm_medium", "utm_campaign",
        "utm_content", "utm_term", "gclid", "fbclid"
    }

    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in tracking
    ]

    return urlunparse((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        parsed.path or "/",
        "",
        urlencode(query, doseq=True),
        ""
    ))


def ask_model_json(system_prompt, user_prompt, max_tokens=2500):
    """Call Nemotron and parse its JSON response."""
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = (response.choices[0].message.content or "").strip()
    content = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        content,
        flags=re.IGNORECASE,
    )

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")

        if start >= 0 and end > start:
            return json.loads(content[start:end + 1])

        raise ValueError(
            f"Model did not return valid JSON: {content[:300]}"
        )


# ============================================================
# 3. TEMPORARY RESEARCH MEMORY
# ============================================================

def connect_db():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")

    db.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            question TEXT NOT NULL,
            objectives TEXT NOT NULL,
            angles TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            rounds INTEGER DEFAULT 0,
            coverage REAL DEFAULT 0,
            report TEXT
        );

        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            query TEXT NOT NULL,
            query_hash TEXT NOT NULL,
            round_no INTEGER NOT NULL,
            purpose TEXT,
            status TEXT NOT NULL DEFAULT 'planned',
            result_count INTEGER DEFAULT 0,
            FOREIGN KEY(session_id) REFERENCES sessions(id)
                ON DELETE CASCADE,
            UNIQUE(session_id, query_hash)
        );

        CREATE TABLE IF NOT EXISTS sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            url TEXT NOT NULL,
            title TEXT,
            snippet TEXT,
            content TEXT,
            domain TEXT,
            published_date TEXT,
            content_hash TEXT,
            fetch_status TEXT,
            FOREIGN KEY(session_id) REFERENCES sessions(id)
                ON DELETE CASCADE,
            UNIQUE(session_id, url)
        );

        CREATE TABLE IF NOT EXISTS findings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            finding TEXT NOT NULL,
            finding_hash TEXT NOT NULL,
            source_ids TEXT NOT NULL,
            kind TEXT,
            FOREIGN KEY(session_id) REFERENCES sessions(id)
                ON DELETE CASCADE,
            UNIQUE(session_id, finding_hash)
        );

        CREATE TABLE IF NOT EXISTS round_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            round_no INTEGER NOT NULL,
            coverage REAL,
            gaps TEXT,
            stop_reason TEXT,
            FOREIGN KEY(session_id) REFERENCES sessions(id)
                ON DELETE CASCADE
        );
    """)

    db.commit()
    return db


def create_session(db, question, plan):
    session_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_" + digest(question + str(time.time()))[:8]
    )

    db.execute("""
        INSERT INTO sessions
        (id, question, objectives, angles, status, created_at)
        VALUES (?, ?, ?, ?, 'running', ?)
    """, (
        session_id,
        question,
        json.dumps(plan.get("objectives", [])),
        json.dumps(plan.get("angles", [])),
        now(),
    ))

    db.commit()
    return session_id


def save_query(db, sid, query, round_no, purpose):
    query = re.sub(r"\s+", " ", (query or "").strip())

    if not query:
        return False

    try:
        db.execute("""
            INSERT INTO queries
            (session_id, query, query_hash, round_no, purpose, status)
            VALUES (?, ?, ?, ?, ?, 'planned')
        """, (
            sid, query, digest(query), round_no, purpose
        ))
        db.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def completed_queries(db, sid):
    rows = db.execute("""
        SELECT query FROM queries
        WHERE session_id = ? AND status = 'done'
        ORDER BY id
    """, (sid,)).fetchall()

    return [row["query"] for row in rows]


def choose_new_queries(db, sid, candidates, limit):
    """Reject previously recorded queries, including failed attempts."""
    rows = db.execute("""
        SELECT query_hash FROM queries WHERE session_id = ?
    """, (sid,)).fetchall()

    seen = {row["query_hash"] for row in rows}
    chosen = []

    for item in candidates:
        if isinstance(item, str):
            item = {"query": item, "purpose": "Research follow-up"}

        query = re.sub(
            r"\s+", " ", str(item.get("query", "")).strip()
        )

        if not query or digest(query) in seen:
            continue

        seen.add(digest(query))
        chosen.append({
            "query": query,
            "purpose": str(item.get("purpose", "Investigate a gap"))
        })

        if len(chosen) >= limit:
            break

    return chosen


# ============================================================
# 4. INITIAL RESEARCH PLANNING
# ============================================================

def make_research_plan(question):
    system = """
You are an autonomous research planner.
Return only valid JSON. Break the user's question into concrete objectives
and diverse investigative angles. Include primary documents, technical
evidence, competing explanations, counterarguments, and uncertainty where
relevant. Avoid repeating the same search in different words.
"""

    prompt = f"""
Research question:
{question}

Create a substantial research plan. Return this JSON structure:

{{
  "objectives": [
    "6-10 specific questions the research must answer"
  ],
  "angles": [
    "8-12 distinct research directions"
  ],
  "initial_queries": [
    {{
      "query": "specific search query",
      "purpose": "what evidence it seeks"
    }}
  ]
}}

Generate 8-12 diverse search queries. Search for primary evidence where
possible, alternative explanations, counterevidence, and source quality.
Do not assume the premise of the question is true.
"""

    plan = ask_model_json(system, prompt, 3000)

    plan.setdefault("objectives", [])
    plan.setdefault("angles", [])
    plan.setdefault("initial_queries", [])

    return plan


# ============================================================
# 5. TAVILY SEARCH AND PAGE EXTRACTION
# ============================================================

def tavily_search(query):
    response = http.post(
        TAVILY_SEARCH_URL,
        json={
            "api_key": TAVILY_KEY,
            "query": query,
            "topic": "general",
            "search_depth": "advanced",
            "max_results": RESULTS_PER_QUERY,
            "include_answer": False,
            "include_raw_content": False,
        },
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()
    return response.json().get("results", [])


def tavily_extract(urls):
    """Attempt to retrieve full page text through Tavily Extract."""
    if not urls:
        return {}

    extracted = {}

    try:
        response = http.post(
            TAVILY_EXTRACT_URL,
            json={
                "api_key": TAVILY_KEY,
                "urls": urls,
                "extract_depth": "advanced",
                "format": "text",
            },
            timeout=35,
        )

        response.raise_for_status()
        payload = response.json()

        for item in payload.get("results", []):
            key = canonical_url(item.get("url", ""))
            content = clean_text(
                item.get("raw_content") or item.get("content") or ""
            )

            if key and content:
                extracted[key] = content[:MAX_PAGE_CHARS]

    except Exception as exc:
        print(f"  Tavily Extract warning: {exc}")

    return extracted


def direct_fetch(url):
    """
    Fallback HTML extraction. Best-effort robots.txt check.
    If access permission cannot be determined, do not fetch directly.
    """
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    try:
        robots_response = http.get(robots_url, timeout=5)

        if robots_response.status_code == 404:
            allowed = True
        elif robots_response.status_code >= 400:
            allowed = False
        else:
            parser = RobotFileParser()
            parser.parse(robots_response.text.splitlines())
            allowed = parser.can_fetch("AdaptiveResearchAgent", url)

        if not allowed:
            return "", "direct_fetch_disallowed"

    except requests.RequestException:
        return "", "robots_check_failed"

    try:
        response = http.get(
            url, timeout=REQUEST_TIMEOUT, allow_redirects=True
        )

        if response.status_code >= 400:
            return "", f"http_{response.status_code}"

        content_type = response.headers.get("Content-Type", "").lower()

        if "html" not in content_type and "text/plain" not in content_type:
            return "", "unsupported_content_type"

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup([
            "script", "style", "nav", "footer", "header",
            "aside", "noscript", "form"
        ]):
            tag.decompose()

        main = soup.find("main") or soup.find("article") or soup.body or soup
        content = clean_text(main.get_text("\n", strip=True))

        return content[:MAX_PAGE_CHARS], "direct_fetch"

    except requests.RequestException as exc:
        return "", f"fetch_error_{type(exc).__name__}"


def collect_results(db, sid, query):
    """
    Save every new URL returned by Tavily, then extract content for newly
    discovered pages within the configured page budget.
    """
    results = tavily_search(query)
    candidates = []

    for item in results:
        original_url = item.get("url", "")
        url = canonical_url(original_url)

        if not url:
            continue

        existing = db.execute("""
            SELECT id, snippet, title FROM sources
            WHERE session_id = ? AND url = ?
        """, (sid, url)).fetchone()

        snippet = clean_text(item.get("content", ""))[:3000]
        title = clean_text(item.get("title", ""))[:500]

        if existing:
            # Enrich an existing record without counting a duplicate URL.
            if len(snippet) > len(existing["snippet"] or ""):
                db.execute("""
                    UPDATE sources SET snippet = ?,
                    title = CASE WHEN ? != '' THEN ? ELSE title END
                    WHERE id = ?
                """, (snippet, title, title, existing["id"]))
            continue

        domain = urlparse(url).netloc

        cursor = db.execute("""
            INSERT INTO sources
            (session_id, url, title, snippet, domain, published_date,
             fetch_status)
            VALUES (?, ?, ?, ?, ?, ?, 'search_result_only')
        """, (
            sid, url, title, snippet, domain,
            item.get("published_date") or item.get("date") or ""
        ))

        candidates.append({
            "id": cursor.lastrowid,
            "url": url,
        })

    db.commit()

    used_pages = db.execute("""
        SELECT COUNT(*) FROM sources
        WHERE session_id = ? AND content IS NOT NULL AND content != ''
    """, (sid,)).fetchone()[0]

    candidates = candidates[:max(0, MAX_PAGES - used_pages)]
    urls = [item["url"] for item in candidates]
    extracted = tavily_extract(urls)

    for item in candidates:
        content = extracted.get(item["url"], "")
        status = "tavily_extract" if content else ""

        if not content:
            content, status = direct_fetch(item["url"])

        db.execute("""
            UPDATE sources
            SET content = ?, content_hash = ?, fetch_status = ?
            WHERE id = ? AND session_id = ?
        """, (
            content or None,
            digest(content) if content else None,
            status or "search_result_only",
            item["id"],
            sid,
        ))

        time.sleep(0.2)

    db.execute("""
        UPDATE queries SET status = 'done', result_count = ?
        WHERE session_id = ? AND query_hash = ?
    """, (len(results), sid, digest(query)))

    db.commit()

    return len(results), len(candidates)


# ============================================================
# 6. EXTRACT AND DEDUPLICATE UNIQUE FINDINGS
# ============================================================

def extract_findings(db, sid, question, objectives, after_id):
    rows = db.execute("""
        SELECT id, title, url, domain, snippet, content
        FROM sources
        WHERE session_id = ? AND id > ?
        ORDER BY id
    """, (sid, after_id)).fetchall()

    if not rows:
        return after_id

    for start in range(0, len(rows), 5):
        batch = rows[start:start + 5]

        source_data = [{
            "id": row["id"],
            "title": row["title"],
            "url": row["url"],
            "domain": row["domain"],
            "text": (row["content"] or row["snippet"] or "")[:4500],
        } for row in batch]

        system = """
You are an evidence extraction assistant.
Return only valid JSON. Extract atomic, useful findings from the supplied
source text. Preserve disagreements instead of merging them. Do not infer
beyond the source. Distinguish reported allegations, established facts,
counterevidence, uncertainties, and useful research leads.
Each finding must cite supplied source IDs. Ignore boilerplate and
repeated wording. Do not invent facts or source IDs.
"""

        prompt = f"""
Research question: {question}

Objectives:
{json.dumps(objectives, ensure_ascii=False)}

Sources:
{json.dumps(source_data, ensure_ascii=False)}

Return:
{{
  "findings": [
    {{
      "text": "one concise, distinct finding",
      "source_ids": [1],
      "kind": "fact|reported_claim|counterevidence|uncertainty|lead"
    }}
  ]
}}

Extract at most 8 useful findings. Prefer new information, specific details,
dates, evidence, counterarguments, and unanswered questions.
"""

        try:
            response = ask_model_json(system, prompt, 2200)
        except Exception as exc:
            print(f"  Finding extraction warning: {exc}")
            continue

        valid_ids = {row["id"] for row in batch}

        for item in response.get("findings", []):
            text = re.sub(r"\s+", " ", str(item.get("text", "")).strip())
            source_ids = item.get("source_ids", [])

            # Models sometimes serialize IDs as strings; normalize safely.
            try:
                source_ids = [int(value) for value in source_ids]
            except (TypeError, ValueError):
                continue

            if len(text) < 20 or not source_ids:
                continue

            if not set(source_ids).issubset(valid_ids):
                continue

            key = digest(text)

            try:
                db.execute("""
                    INSERT INTO findings
                    (session_id, finding, finding_hash, source_ids, kind)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    sid, text, key,
                    json.dumps(sorted(set(source_ids))),
                    str(item.get("kind", "reported_claim")),
                ))

            except sqlite3.IntegrityError:
                # Same finding seen again: preserve any additional source links.
                old = db.execute("""
                    SELECT id, source_ids FROM findings
                    WHERE session_id = ? AND finding_hash = ?
                """, (sid, key)).fetchone()

                if old:
                    merged = set(json.loads(old["source_ids"]))
                    merged.update(source_ids)

                    db.execute("""
                        UPDATE findings SET source_ids = ? WHERE id = ?
                    """, (json.dumps(sorted(merged)), old["id"]))

    db.commit()
    return max(row["id"] for row in rows)


def evidence_context(db, sid):
    sources = db.execute("""
        SELECT id, title, url, domain, published_date, content, snippet,
               fetch_status
        FROM sources WHERE session_id = ?
        ORDER BY id
    """, (sid,)).fetchall()

    findings = db.execute("""
        SELECT id, finding, source_ids, kind
        FROM findings WHERE session_id = ?
        ORDER BY id
    """, (sid,)).fetchall()

    parts = ["SOURCE INVENTORY"]

    for source in sources:
        body = source["content"] or source["snippet"] or "(No text extracted)"

        parts.append(
            f"[S{source['id']}] {source['title']}\n"
            f"URL: {source['url']}\n"
            f"Domain: {source['domain']}\n"
            f"Date: {source['published_date'] or 'unknown'}\n"
            f"Extraction: {source['fetch_status']}\n"
            f"Text: {body[:1800]}"
        )

    parts.append("\nUNIQUE FINDINGS")

    for finding in findings:
        parts.append(
            f"[F{finding['id']}] {finding['finding']}\n"
            f"Type: {finding['kind']}; source IDs: {finding['source_ids']}"
        )

    return "\n\n".join(parts)[:MAX_CONTEXT_CHARS]


# ============================================================
# 7. ADAPTIVE RESEARCH PLANNER
# ============================================================

def plan_next_round(question, objectives, angles, context, previous, round_no):
    system = """
You control an autonomous web research process.
Return only valid JSON. Examine the accumulated evidence, identify the
most important unanswered questions, and propose genuinely new searches.
Do not repeat existing queries or generate superficial rewordings.
Seek primary evidence, independent corroboration, counterevidence, technical
details, and alternative explanations as appropriate.
A repeated claim across copied articles is not independent corroboration.
Coverage is a subjective estimate of objective coverage, not answer accuracy.
"""

    prompt = f"""
Question: {question}
Round: {round_no}

Objectives:
{json.dumps(objectives, ensure_ascii=False)}

Research angles:
{json.dumps(angles, ensure_ascii=False)}

Queries already executed:
{json.dumps(previous[-60:], ensure_ascii=False)}

Accumulated evidence:
{context}

Return JSON:
{{
  "coverage_percent": 0,
  "covered_objectives": [],
  "gaps": ["specific unresolved questions"],
  "new_angles": ["new research angles discovered"],
  "next_queries": [
    {{"query": "specific query", "purpose": "what it investigates"}}
  ],
  "should_stop": false,
  "stop_reason": "reason"
}}

If important gaps remain, provide 4-8 useful candidate queries.
Set should_stop true only when the evidence is reasonably sufficient or
further research has low expected value. Never stop merely because multiple
sources repeat the same conclusion.
"""

    return ask_model_json(system, prompt, 3000)


# ============================================================
# 8. FINAL REPORT
# ============================================================

def generate_report(question, objectives, context, coverage, gaps):
    system = """
You are a careful research analyst. Write a well-organized report based
only on the supplied research record. Cite factual claims with source IDs
such as [S4]. Never invent evidence or citations.
Separate primary sources, secondary reporting, allegations, corroboration,
conflicts, and unknowns. Repeated claims are not automatically independent
confirmation. Explicitly acknowledge missing or inaccessible sources.
"""

    prompt = f"""
Research question: {question}

Objectives:
{json.dumps(objectives, ensure_ascii=False)}

Estimated research coverage: {coverage}%
Remaining gaps:
{json.dumps(gaps, ensure_ascii=False)}

Research record:
{context}

Write:
1. Executive summary and direct answer.
2. Main findings with inline [S#] source references.
3. Evidence supporting each important conclusion.
4. Conflicting evidence and alternative explanations.
5. What remains uncertain and what would change the conclusion.
6. Source list with IDs, titles, and URLs.

Do not treat the estimated coverage as a probability of correctness.
If the record is insufficient, say so explicitly.
"""

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        max_tokens=4000,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )

    return (response.choices[0].message.content or "").strip()


# ============================================================
# 9. END-TO-END AGENT LOOP
# ============================================================

def run_research(question):
    db = connect_db()
    sid = None

    try:
        print("\n[1] Building the research plan...")
        plan = make_research_plan(question)

        objectives = [
            str(item) for item in plan.get("objectives", [])
            if str(item).strip()
        ]
        angles = [
            str(item) for item in plan.get("angles", [])
            if str(item).strip()
        ]

        sid = create_session(db, question, plan)

        print("\nRESEARCH OBJECTIVES")
        for index, objective in enumerate(objectives, start=1):
            print(f"{index}. {objective}")

        print(f"\nResearch session: {sid}")
        print(f"Memory database: {DB_PATH}")

        initial_queries = choose_new_queries(
            db, sid,
            plan.get("initial_queries", []),
            min(MAX_QUERIES_PER_ROUND * 2, MAX_TOTAL_QUERIES),
        )

        if not initial_queries:
            initial_queries = [{
                "query": question,
                "purpose": "Initial discovery",
            }]

        total_queries = 0
        last_source_id = 0
        coverage = 0
        gaps = []

        for round_no in range(1, MAX_ROUNDS + 1):
            if total_queries >= MAX_TOTAL_QUERIES:
                print("\nTotal query budget reached.")
                break

            print(f"\n{'=' * 15} ROUND {round_no}/{MAX_ROUNDS} {'=' * 15}")

            if round_no == 1:
                batch = initial_queries
            else:
                context = evidence_context(db, sid)
                previous = completed_queries(db, sid)

                print("Nemotron is analyzing collected evidence and research gaps...")

                try:
                    decision = plan_next_round(
                        question, objectives, angles,
                        context, previous, round_no,
                    )
                except Exception as exc:
                    print(f"Follow-up planning failed: {exc}")
                    break

                try:
                    coverage = max(
                        0, min(100, float(decision.get("coverage_percent", 0)))
                    )
                except (TypeError, ValueError):
                    coverage = 0

                gaps = [
                    str(item) for item in decision.get("gaps", [])
                ]

                for angle in decision.get("new_angles", []):
                    if str(angle).strip() and str(angle) not in angles:
                        angles.append(str(angle))

                db.execute("""
                    UPDATE sessions SET angles = ?, coverage = ?
                    WHERE id = ?
                """, (json.dumps(angles), coverage, sid))

                db.execute("""
                    INSERT INTO round_notes
                    (session_id, round_no, coverage, gaps, stop_reason)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    sid, round_no, coverage,
                    json.dumps(gaps),
                    str(decision.get("stop_reason", "")),
                ))
                db.commit()

                if decision.get("should_stop") and coverage >= 80:
                    print("Research objectives appear sufficiently covered.")
                    print(decision.get("stop_reason", ""))
                    break

                remaining = MAX_TOTAL_QUERIES - total_queries
                batch = choose_new_queries(
                    db, sid, decision.get("next_queries", []),
                    min(MAX_QUERIES_PER_ROUND, remaining),
                )

            if not batch:
                print("No new non-duplicate queries available. Stopping.")
                break

            for item in batch:
                if total_queries >= MAX_TOTAL_QUERIES:
                    break

                query = item["query"]
                purpose = item.get("purpose", "")

                if not save_query(db, sid, query, round_no, purpose):
                    continue

                print(f"\nSearching: {query}")
                print(f"Purpose: {purpose}")

                try:
                    result_count, new_sources = collect_results(db, sid, query)
                    total_queries += 1

                    print(f"Search results: {result_count}")
                    print(f"New unique URLs: {new_sources}")

                except Exception as exc:
                    total_queries += 1
                    db.execute("""
                        UPDATE queries SET status = 'failed'
                        WHERE session_id = ? AND query_hash = ?
                    """, (sid, digest(query)))
                    db.commit()

                    print(f"Search failed: {type(exc).__name__}: {exc}")

                time.sleep(0.3)

            print("\nExtracting distinct findings...")
            last_source_id = extract_findings(
                db, sid, question, objectives, last_source_id
            )

            stats = db.execute("""
                SELECT
                  (SELECT COUNT(*) FROM queries
                   WHERE session_id = ? AND status = 'done') AS searches,
                  (SELECT COUNT(*) FROM sources
                   WHERE session_id = ?) AS sources,
                  (SELECT COUNT(*) FROM sources
                   WHERE session_id = ? AND content IS NOT NULL
                   AND content != '') AS pages,
                  (SELECT COUNT(*) FROM findings
                   WHERE session_id = ?) AS findings
            """, (sid, sid, sid, sid)).fetchone()

            print(
                f"Memory: {stats['searches']} completed searches | "
                f"{stats['sources']} unique URLs | "
                f"{stats['pages']} pages with content | "
                f"{stats['findings']} unique findings"
            )

            db.execute("""
                UPDATE sessions SET rounds = ? WHERE id = ?
            """, (round_no, sid))
            db.commit()

        # One final evaluation informs the report about remaining gaps.
        print("\n[Final evaluation] Assessing coverage and unresolved questions...")

        context = evidence_context(db, sid)

        try:
            final_eval = plan_next_round(
                question, objectives, angles, context,
                completed_queries(db, sid), MAX_ROUNDS + 1,
            )
            coverage = max(
                0, min(100, float(final_eval.get("coverage_percent", coverage)))
            )
            gaps = [
                str(item) for item in final_eval.get("gaps", gaps)
            ]
        except Exception as exc:
            print(f"Final coverage evaluation failed: {exc}")

        print("\nGenerating the evidence-aware report...")
        context = evidence_context(db, sid)
        report = generate_report(
            question, objectives, context, coverage, gaps
        )

        db.execute("""
            UPDATE sessions
            SET status = 'completed', report = ?, coverage = ?
            WHERE id = ?
        """, (report, coverage, sid))
        db.commit()

        print("\n" + "=" * 70)
        print("FINAL RESEARCH REPORT")
        print("=" * 70)
        print(report)

        print("\n" + "=" * 70)
        print("SESSION SUMMARY")
        print("=" * 70)

        print(f"Session ID: {sid}")
        print(f"Completed searches: {total_queries}")

        for table, label in [
            ("sources", "Unique URLs"),
            ("findings", "Unique findings"),
        ]:
            count = db.execute(
                f"SELECT COUNT(*) FROM {table} WHERE session_id = ?",
                (sid,),
            ).fetchone()[0]
            print(f"{label}: {count}")

        page_count = db.execute("""
            SELECT COUNT(*) FROM sources
            WHERE session_id = ? AND content IS NOT NULL AND content != ''
        """, (sid,)).fetchone()[0]

        print(f"Pages with extracted text: {page_count}")
        print(f"Estimated objective coverage: {coverage}% (not accuracy)")
        print(f"Remaining gaps: {json.dumps(gaps, ensure_ascii=False)}")
        print(f"SQLite memory: {DB_PATH}")
        print("Session data is retained in SQLite for inspection.")

        return report

    finally:
        db.close()


if __name__ == "__main__":
    print("=" * 70)
    print("ADAPTIVE WEB RESEARCH AGENT")
    print("Tavily + Nebius/NVIDIA Nemotron + SQLite")
    print("=" * 70)

    print(
        f"Limits: {MAX_ROUNDS} rounds | "
        f"{MAX_TOTAL_QUERIES} queries maximum | "
        f"{MAX_QUERIES_PER_ROUND} queries per follow-up round | "
        f"{RESULTS_PER_QUERY} results per query | "
        f"{MAX_PAGES} pages maximum"
    )

    question = input("\nWhat would you like to research? ").strip()

    if question:
        try:
            run_research(question)
        except KeyboardInterrupt:
            print("\nInterrupted. Previously committed research data remains in SQLite.")
        except Exception as exc:
            print(f"\nResearch failed: {type(exc).__name__}: {exc}")
            print("Check your API keys, model ID, network, and API limits.")
            raise
    else:
        print("No question entered.")
