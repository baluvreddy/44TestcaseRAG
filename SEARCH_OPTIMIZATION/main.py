import os
import io
import time
import uuid
import math
import logging
import numpy as np
import pandas as pd
import google.generativeai as genai

from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# --- Load env ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "multilevel_test_cases_cloud"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- App ---
app = FastAPI(
    title="Intelligent Test Case Search API (Enhanced)",
    description="Upload, enrich, and search test cases with hybrid search, reranking, and diversity.",
    version="1.1.0"
)

templates = Jinja2Templates(directory=".")
# (Optional) serve static if you have a frontend
# app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Global models/clients (initialized on startup) ---
embedding_model: SentenceTransformer = None
qdrant_client: QdrantClient = None

# --- Simple in-memory TTL cache for search queries ---
SEARCH_CACHE: Dict[str, Tuple[float, Any]] = {}
CACHE_TTL_SECONDS = 60 * 5  # 5 minutes; tune as needed

# --- Configurable params ---
CANDIDATES_TO_RETRIEVE = 15   # initial Qdrant retrieval
FINAL_RESULTS = 3             # results returned to user
GEMINI_RERANK_ENABLED = True  # toggle rerank with Gemini
QUERY_EXPANSION_ENABLED = True
DIVERSITY_ENFORCE = True
DIVERSITY_PER_FEATURE = True  # ensure top results span features
GEMINI_RATE_LIMIT_SLEEP = 0.5  # seconds between Gemini calls (if multiple)

# --- Utility functions ---


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def cache_get(query: str):
    entry = SEARCH_CACHE.get(query)
    if not entry:
        return None
    ts, value = entry
    if time.time() - ts > CACHE_TTL_SECONDS:
        del SEARCH_CACHE[query]
        return None
    return value


def cache_set(query: str, value: Any):
    SEARCH_CACHE[query] = (time.time(), value)


def safe_parse_lines(text: str) -> List[str]:
    return [l.strip() for l in text.splitlines() if l.strip()]

def cleanup_na_test_cases():
    """
    Delete all test cases in Qdrant where Test Case ID is 'NA' or empty.
    This version passes the models.Filter object directly to the points_selector.
    """
    try:
        # Call 1: Delete all points where 'Test Case ID' is "NA"
        logger.info("Attempting to delete points with Test Case ID = 'NA'...")
        
        # FIX: Create the Filter object.
        na_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="Test Case ID",
                    match=models.MatchValue(value="NA"),
                )
            ]
        )
        # FIX: Pass the filter directly, without the FilterSelector wrapper.
        qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=na_filter
        )
        logger.info("Deletion of 'NA' test cases completed.")

        # Call 2: Delete all points where 'Test Case ID' is an empty string
        logger.info("Attempting to delete points with empty Test Case ID...")

        # FIX: Create the Filter object.
        empty_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="Test Case ID",
                    match=models.MatchValue(value=""),
                )
            ]
        )
        # FIX: Pass the filter directly here as well.
        qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=empty_filter
        )
        logger.info("Deletion of empty ID test cases completed.")
        
        return {"message": "Cleanup of invalid test cases complete."}
        
    except Exception as e:
        logger.error(f"Cleanup NA test cases failed: {e}")
        return {"message": f"Cleanup failed: {e}"}


# --- Gemini helpers (query expansion + reranking) ---


def configure_gemini():
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        logger.info("Gemini configured.")
    except Exception as e:
        logger.error(f"Failed to configure Gemini: {e}")


def expand_query_with_gemini(query: str) -> List[str]:
    """
    Use Gemini to produce a small list of paraphrases/synonyms for query expansion.
    Fallback: return [query].
    """
    if not QUERY_EXPANSION_ENABLED:
        return [query]

    # If API key not set, fallback gracefully
    if not GOOGLE_API_KEY:
        logger.warning("GOOGLE_API_KEY not set; skipping Gemini expansion.")
        return [query]

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"""
You are an assistant that expands short search queries into useful paraphrases and synonyms for software test-case search.
Return only a comma-separated single line of 5 short paraphrases/keywords (no numbering).
Query: "{query}"
Output example: login error, authentication failure, invalid credentials, sign-in fails, cannot login
"""
        response = model.generate_content(prompt)
        text = response.text.strip()
        # Parse comma-separated
        parts = [p.strip() for p in text.replace("\n", ",").split(",") if p.strip()]
        # always include original query as first
        expansions = [query] + parts
        logger.info(f"Query expansion for '{query}': {expansions[:6]}")
        time.sleep(GEMINI_RERANK_ENABLED and GEMINI_RATE_LIMIT_SLEEP or 0)  # be nice with rate limits
        return expansions[:6]  # limit number of expansions
    except Exception as e:
        logger.error(f"Gemini expansion failed: {e}")
        return [query]


def rerank_with_gemini(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ask Gemini to rerank the candidates. Candidates should be a list of dicts with id, feature, description.
    Return candidates in new order (top-first). If Gemini fails, return original order.
    """
    if not GEMINI_RERANK_ENABLED:
        return candidates

    if not GOOGLE_API_KEY:
        logger.warning("GOOGLE_API_KEY not set; skipping Gemini rerank.")
        return candidates

    # Prepare prompt with candidates
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt_lines = [
            "You are a helpful assistant. Re-rank the following test cases by relevance to the query.",
            f"Query: \"{query}\"",
            "Return only a newline-separated list of candidate IDs (the 'id' field) in best-to-worst order. Do not add extra commentary."
        ]
        prompt_lines.append("\nCandidates:")
        for c in candidates:
            brief = c.get("description") or c.get("summary") or ""
            # Keep candidates compact
            prompt_lines.append(f"{c['id']} | Feature: {c.get('feature','N/A')} | Desc: {brief[:220]}")
        prompt = "\n".join(prompt_lines)

        response = model.generate_content(prompt)
        text = response.text.strip()
        lines = safe_parse_lines(text)
        # lines may be IDs separated by spaces or with numbering; try to extract ids
        ordered_ids = []
        for l in lines:
            # split on spaces and punctuation, find token matching a candidate id
            token = l.split()[0].strip().strip(".").strip("-")
            ordered_ids.append(token)

        # Map back to candidate dicts
        id_to_c = {c['id']: c for c in candidates}
        ordered = [id_to_c[i] for i in ordered_ids if i in id_to_c]
        # If Gemini returned partial order, append missing ones preserving original order
        seen = set([c['id'] for c in ordered])
        ordered += [c for c in candidates if c['id'] not in seen]
        logger.info("Rerank: Gemini provided an ordering.")
        time.sleep(GEMINI_RATE_LIMIT_SLEEP)
        return ordered
    except Exception as e:
        logger.error(f"Gemini rerank failed: {e}")
        return candidates


# --- Startup: init embedding model & Qdrant client and Gemini configuration ---


@app.on_event("startup")
async def startup_event():
    global embedding_model, qdrant_client
    logger.info("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info("Embedding model loaded.")

    logger.info("Connecting to Qdrant Cloud...")
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=30
    )
    logger.info("Connected to Qdrant.")

    logger.info("Configuring Gemini (if key present)...")
    configure_gemini()

    # ensure collection exists (recreate if missing)
    try:
        existing = qdrant_client.get_collections().collections
        names = [c.name for c in existing]
        if COLLECTION_NAME not in names:
            logger.info(f"Creating collection {COLLECTION_NAME}...")
            qdrant_client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=embedding_model.get_sentence_embedding_dimension(),
                    distance=models.Distance.COSINE
                ),
            )
            logger.info("Collection created.")
    except Exception as e:
        logger.error(f"Qdrant collection check/create failed: {e}")


# --- Serve frontend ---


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# --- Enhanced upload: store multiple embeddings per test case ---


@app.post("/api/upload")
async def upload_and_process_file(file: UploadFile = File(...)):
    if not file.filename.endswith(('.csv', '.xlsx')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV or XLSX file.")

    try:
        contents = await file.read()
        buffer = io.BytesIO(contents)
        
        # More robust CSV reading for your data format
        if file.filename.endswith('.csv'):
            df = pd.read_csv(buffer, encoding='utf-8', quotechar='"', skipinitialspace=True)
        else:
            df = pd.read_excel(buffer)
            
        # Log the original dataframe info
        logger.info(f"Original DataFrame shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Use astype(str) to safely handle all columns and replace 'nan'
        df = df.astype(str).replace('nan', '').replace('NaN', '').replace('None', '')
        
        # Log some sample data
        logger.info(f"Sample of first few rows:")
        for idx, row in df.head(10).iterrows():
            logger.info(f"Row {idx}: ID='{row.get('Test Case ID', '')}', Step No='{row.get('Step No.', '')}', Test Step='{str(row.get('Test Step', ''))[:50]}...'")
            
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading file: {e}")

    # **CRITICAL FIX**: Forward-fill the Test Case ID column
    # In your CSV, only the first row of each test case has the ID, subsequent rows are empty
    # We need to fill down the Test Case ID to all rows of the same test case
    df['Test Case ID'] = df['Test Case ID'].replace('', pd.NA).fillna(method='ffill')
    
    logger.info("After forward-filling Test Case ID:")
    for idx, row in df.head(10).iterrows():
        logger.info(f"Row {idx}: ID='{row.get('Test Case ID', '')}', Step No='{row.get('Step No.', '')}', Test Step='{str(row.get('Test Step', ''))[:50]}...'")

    # **IMPROVEMENT**: Filter out invalid Test Case IDs before grouping
    df = df[df["Test Case ID"].str.strip() != '']
    df = df[df["Test Case ID"].str.strip().str.upper() != 'NA']
    
    if df.empty:
        return {"message": "No valid test cases found to process in the file."}

    points_to_upsert = []
    grouped = df.groupby("Test Case ID")
    logger.info(f"Processing {len(grouped)} unique test cases...")

    for test_case_id, group in grouped:
        # Get basic info from first row (these should be same across all rows for same test case)
        feature = str(group["Feature"].iloc[0]) if "Feature" in group.columns else ""
        description = str(group["Test Case Description"].iloc[0]) if "Test Case Description" in group.columns else ""
        prerequisites = str(group["Pre-requisites"].iloc[0]) if "Pre-requisites" in group.columns else ""

        # FIXED: Properly combine ALL steps from ALL rows in the group
        steps_list = []
        logger.info(f"Processing test case: {test_case_id}, rows in group: {len(group)}")
        
        for idx, row in group.iterrows():
            # Get step information from each row
            step_no = str(row.get('Step No.', '')).strip()
            test_step = str(row.get('Test Step', '')).strip()
            expected_result = str(row.get('Expected Result', '')).strip()
            
            # Debug log for each row
            logger.info(f"Row {idx}: Step No='{step_no}', Test Step='{test_step[:50]}...', Expected='{expected_result[:50]}...'")
            
            # Only add non-empty test steps and skip 'nan' values
            if test_step and test_step.lower() not in ['nan', '', 'none', 'null']:
                # Format: "Step X: [Test Step] → Expected: [Expected Result]"
                if step_no and step_no.lower() not in ['nan', '', 'none', 'null']:
                    formatted_step = f"Step {step_no}: {test_step}"
                else:
                    # If no step number, just use the test step
                    formatted_step = f"{test_step}"
                
                if expected_result and expected_result.lower() not in ['nan', '', 'none', 'null']:
                    formatted_step += f" → Expected: {expected_result}"
                
                steps_list.append(formatted_step)
                logger.info(f"Added step: {formatted_step[:100]}...")
            else:
                logger.info(f"Skipped empty/invalid step: '{test_step}'")
        
        # Combine all steps with double newlines for better readability in the modal
        steps_combined = "\n\n".join(steps_list) if steps_list else ""
        
        # Debug logging to verify steps are being combined correctly
        logger.info(f"Test Case ID: {test_case_id}, Steps found: {len(steps_list)}")
        if len(steps_list) > 1:
            logger.info(f"First step: {steps_list[0][:100]}...")
            logger.info(f"Last step: {steps_list[-1][:100]}...")
            logger.info(f"Combined length: {len(steps_combined)} characters")
            logger.info(f"Combined steps preview: {steps_combined[:200]}...")
        elif len(steps_list) == 1:
            logger.info(f"Single step: {steps_list[0][:100]}...")
        else:
            logger.warning(f"No valid steps found for test case: {test_case_id}")

        # Continue with enrichment
        enrichment = get_gemini_enrichment(description, feature)
        summary = enrichment.get("summary", "")
        keywords = enrichment.get("keywords", [])

        desc_emb = embedding_model.encode(description).tolist() if description else None
        steps_emb = embedding_model.encode(steps_combined).tolist() if steps_combined else None
        summary_emb = embedding_model.encode(summary).tolist() if summary else None

        # Safer way to average embeddings, handles cases where some fields are empty
        valid_embeddings = [np.array(e) for e in [desc_emb, steps_emb, summary_emb] if e is not None]
        main_vector = np.mean(valid_embeddings, axis=0).tolist() if valid_embeddings else embedding_model.encode("").tolist()

        payload = {
            "Test Case ID": test_case_id,
            "Feature": feature,
            "Test Case Description": description,
            "Pre-requisites": prerequisites,
            "Steps": steps_combined,
            "TestCaseSummary": summary,
            "TestCaseKeywords": keywords,
            "desc_embedding": desc_emb,
            "steps_embedding": steps_emb,
            "summary_embedding": summary_emb
        }

        point_id = str(uuid.uuid4())
        points_to_upsert.append(
            models.PointStruct(
                id=point_id,
                vector=main_vector,
                payload=payload
            )
        )
        time.sleep(0.15)

    if not points_to_upsert:
        return {"message": "No valid test cases found to process in the file."}

    try:
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points_to_upsert, wait=True)
        logger.info(f"Upserted {len(points_to_upsert)} points.")
        return {"message": f"Successfully processed and stored {len(points_to_upsert)} test cases."}
    except Exception as e:
        logger.error(f"Error upserting: {e}")
        raise HTTPException(status_code=500, detail=f"Error storing data: {e}")


# --- Get all ---


@app.get("/api/get-all")
async def get_all_test_cases():
    try:
        all_points, _ = qdrant_client.scroll(collection_name=COLLECTION_NAME, limit=100, with_payload=True)
        test_cases = []
        for p in all_points:
            # Create a dictionary that includes the point's ID and its entire payload
            # This makes its structure identical to the search results
            full_test_case_data = p.payload if isinstance(p.payload, dict) else {}
            full_test_case_data['id'] = p.id
            test_cases.append(full_test_case_data)
            
        return {"test_cases": test_cases}
    except Exception as e:
        logger.error(f"Error retrieving: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while retrieving data: {e}")

# --- Delete all (recreate) ---


@app.post("/api/delete-all")
async def delete_all_data():
    try:
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=embedding_model.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE
            ),
        )
        logger.info("Recreated collection; data cleared.")
        return {"message": "All test case data has been successfully deleted."}
    except Exception as e:
        logger.error(f"Failed to delete data: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while deleting data: {e}")


# --- NEW: Update endpoint ---

class UpdateTestCaseRequest(BaseModel):
    feature: str
    description: str
    prerequisites: str
    steps: str

@app.put("/api/update/{point_id}")
async def update_test_case(point_id: str, update_data: UpdateTestCaseRequest):
    try:
        # First, get the existing point to preserve other fields
        existing_points = qdrant_client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id],
            with_payload=True
        )
        
        if not existing_points:
            raise HTTPException(status_code=404, detail="Test case not found")
        
        existing_payload = existing_points[0].payload
        
        # Update the specific fields
        updated_payload = existing_payload.copy()
        updated_payload.update({
            "Feature": update_data.feature,
            "Test Case Description": update_data.description,
            "Pre-requisites": update_data.prerequisites,
            "Steps": update_data.steps,
        })
        
        # Re-generate embeddings for updated content
        desc_emb = embedding_model.encode(update_data.description).tolist() if update_data.description else None
        steps_emb = embedding_model.encode(update_data.steps).tolist() if update_data.steps else None
        
        # Get existing summary embedding or regenerate if needed
        summary = existing_payload.get("TestCaseSummary", "")
        summary_emb = existing_payload.get("summary_embedding")
        if not summary_emb and summary:
            summary_emb = embedding_model.encode(summary).tolist()
        
        # Update embeddings in payload
        updated_payload.update({
            "desc_embedding": desc_emb,
            "steps_embedding": steps_emb,
            "summary_embedding": summary_emb
        })
        
        # Recalculate main vector
        valid_embeddings = [np.array(e) for e in [desc_emb, steps_emb, summary_emb] if e is not None]
        main_vector = np.mean(valid_embeddings, axis=0).tolist() if valid_embeddings else embedding_model.encode("").tolist()
        
        # Update the point in Qdrant
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=main_vector,
                    payload=updated_payload
                )
            ],
            wait=True
        )
        
        logger.info(f"Updated test case {point_id}")
        return {"message": "Test case updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating test case: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating test case: {e}")


# --- Enhanced search endpoint ---


@app.post("/api/search")
async def search_test_cases(request: Request):
    data = await request.json()
    raw_query = data.get("query")
    filter_feature = data.get("feature")  # optional filter: only search within this feature
    if not raw_query:
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    # Check cache
    cache_key = f"{raw_query}::feature={filter_feature}"
    cached = cache_get(cache_key)
    if cached:
        logger.info("Returning cached results.")
        return cached

    logger.info(f"Search query: {raw_query} (feature filter: {filter_feature})")

    # 1) Query expansion (paraphrases/synonyms)
    expansions = expand_query_with_gemini(raw_query) if QUERY_EXPANSION_ENABLED else [raw_query]
    # create combined query string for embedding
    combined_query = " ".join(expansions)
    query_vector = embedding_model.encode(combined_query).tolist()

    # 2) Basic vector search in Qdrant to get candidates
    try:
        # optionally apply feature filter
        q_filter = None
        if filter_feature:
            q_filter = models.Filter(
                must=[models.FieldCondition(
                    key="Feature",
                    match=models.MatchValue(value=filter_feature)
                )]
            )

        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=CANDIDATES_TO_RETRIEVE,
            with_payload=True,
            query_filter=q_filter
        )
    except Exception as e:
        logger.error(f"Qdrant search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    # 3) Hybrid local reranking: combine Qdrant score with field-level cosine boosts and keyword matches
    candidates = []
    for res in search_results:
        payload = res.payload if isinstance(res.payload, dict) else {}
        # base score from Qdrant (cosine or other)
        base_score = float(res.score) if hasattr(res, "score") else 0.0

        # local boosts using stored embeddings (desc/steps/summary)
        desc_emb = np.array(payload.get("desc_embedding", []), dtype=float) if payload.get("desc_embedding") else None
        steps_emb = np.array(payload.get("steps_embedding", []), dtype=float) if payload.get("steps_embedding") else None
        summary_emb = np.array(payload.get("summary_embedding", []), dtype=float) if payload.get("summary_embedding") else None

        # compute similarities
        sim_desc = cosine_similarity(np.array(query_vector), desc_emb) if desc_emb is not None else 0.0
        sim_steps = cosine_similarity(np.array(query_vector), steps_emb) if steps_emb is not None else 0.0
        sim_summary = cosine_similarity(np.array(query_vector), summary_emb) if summary_emb is not None else 0.0

        # keyword exact match boost (if any expansion or raw query tokens in keywords/payload text)
        keywords = [k.lower() for k in payload.get("TestCaseKeywords", [])] if payload.get("TestCaseKeywords") else []
        text_fields = " ".join([
            str(payload.get("Feature", "")),
            str(payload.get("Test Case Description", "")),
            str(payload.get("Steps", "")),
        ]).lower()

        token_boost = 0.0
        for term in expansions:
            t = term.lower().strip()
            if not t:
                continue
            if t in text_fields:
                token_boost += 0.12  # tune
            if t in keywords:
                token_boost += 0.18  # stronger if present in extracted keywords

        # weighted aggregation: base_score (from qdrant) is high importance; field sims moderate; token boost small
        # Note: Qdrant cosine scores can be in [-1,1] depending on config â€" normalize conservatively
        local_score = (0.6 * ((base_score + 1) / 2)) + (0.25 * max(sim_desc, sim_steps, sim_summary)) + token_boost
        # Keep also raw similarity fields for debugging/reranking
        candidates.append({
            "id": res.id,
            "raw_score": base_score,
            "local_score": local_score,
            "feature": payload.get("Feature", "N/A"),
            "test_case_id": payload.get("Test Case ID", "NA"),
            "description": payload.get("Test Case Description", ""),
            "summary": payload.get("TestCaseSummary", ""),
            "keywords": payload.get("TestCaseKeywords", []),
            "payload": payload
        })

    # 4) Sort by local_score descending and pick top candidates for reranking
    candidates.sort(key=lambda x: x["local_score"], reverse=True)
    top_candidates = candidates[:CANDIDATES_TO_RETRIEVE]

    # 5) LLM rerank (Gemini) - optional, reorders the top candidates for final precision
    reranked = top_candidates
    try:
        if GEMINI_RERANK_ENABLED and GOOGLE_API_KEY:
            reranked = rerank_with_gemini(raw_query, top_candidates)
    except Exception as e:
        logger.error(f"Rerank step failed: {e}")
        reranked = top_candidates

    # 6) Diversity enforcement: ensure top results are not duplicates; prefer distinct features
    final_list = []
    seen_features = set()
    for cand in reranked:
        if len(final_list) >= FINAL_RESULTS:
            break
        if DIVERSITY_ENFORCE and DIVERSITY_PER_FEATURE:
            feat = (cand.get("feature") or "N/A")
            if feat in seen_features:
                # skip duplicates of same feature unless no choice
                continue
            final_list.append(cand)
            seen_features.add(feat)
        else:
            final_list.append(cand)

    # If diversity removed too many, fill up with remaining reranked candidates
    if len(final_list) < FINAL_RESULTS:
        for cand in reranked:
            if cand in final_list:
                continue
            final_list.append(cand)
            if len(final_list) >= FINAL_RESULTS:
                break

    # build response (convert local_score to a percentage-like probability)
    response_items = []
    for c in final_list:
        score_pct = round(min(max(c["local_score"], 0.0), 1.0) * 100, 2)
        # We now include the full payload so the frontend can edit all fields
        full_payload = c.get("payload", {})
        response_items.append({
            "id": c["id"],
            "probability": score_pct,
            "test_case_id": full_payload.get("Test Case ID", "NA"),
            "feature": full_payload.get("Feature", "N/A"),
            "description": full_payload.get("Test Case Description", ""),
            "prerequisites": full_payload.get("Pre-requisites", ""), # This line is crucial
            "steps": full_payload.get("Steps", ""), # This line is also crucial
            "summary": full_payload.get("TestCaseSummary", ""),
            "keywords": full_payload.get("TestCaseKeywords", [])
        })

    result = {"results": response_items}
    cache_set(cache_key, result)
    logger.info(f"Returning {len(response_items)} results for query '{raw_query}'")
    return result


# --- Gemini enrichment helper (used in upload) ---


def get_gemini_enrichment(test_case_description: str, feature: str) -> dict:
    """
    Uses Google Gemini to generate enriched metadata:
    - 3-5 line detailed summary
    - 15-20+ dynamic keywords
    - intent and risks
    """
    if not GOOGLE_API_KEY:
        logger.warning("Google API key not present; using simple heuristics for enrichment.")
        summary = (test_case_description[:600] + '...') if len(test_case_description) > 600 else test_case_description
        tokens = [t.strip().lower() for t in test_case_description.replace('.', ' ').split() if len(t) > 3]
        keywords = list(dict.fromkeys(tokens))[:20]  # fallback approx 20
        return {
            "summary": summary,
            "keywords": keywords,
            "intent": "Heuristic intent extraction not available.",
            "risks": []
        }

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = f"""
Analyze the following software test case and generate enriched metadata:

Feature: "{feature}"
Test Case Description: "{test_case_description}"

Output format (exactly):

Summary: <3-5 lines clearly explaining purpose, prerequisites, process, and expected outcome in detail>
Keywords: <15-20+ diverse keywords/phrases (allow more if meaningful), include synonyms, variations, error cases, and domain-specific terms>
Intent: <short description of the primary validation or business logic tested>
Risks: <comma-separated list of potential risks, edge cases, or failure scenarios this test addresses>
"""

        response = model.generate_content(prompt)
        text = response.text.strip()

        # Defaults
        summary, intent = "", ""
        keywords, risks = [], []

        for line in text.splitlines():
            lower = line.lower()
            if lower.startswith("summary:"):
                summary = line.split(":", 1)[1].strip()
            elif lower.startswith("keywords:"):
                raw_kw = line.split(":", 1)[1]
                keywords = [k.strip() for k in raw_kw.split(",") if k.strip()]
            elif lower.startswith("intent:"):
                intent = line.split(":", 1)[1].strip()
            elif lower.startswith("risks:"):
                raw_risks = line.split(":", 1)[1]
                risks = [r.strip() for r in raw_risks.split(",") if r.strip()]

        # Safety: enforce at least 15 keywords if Gemini under-delivers
        if len(keywords) < 15:
            tokens = [t.strip().lower() for t in test_case_description.split() if len(t) > 3]
            extra_kw = list(dict.fromkeys(tokens))
            keywords.extend(extra_kw[:20 - len(keywords)])

        time.sleep(0.2)  # avoid API rate spike
        return {
            "summary": summary,
            "keywords": keywords,
            "intent": intent,
            "risks": risks
        }

    except Exception as e:
        logger.error(f"Error calling Gemini: {e}")
        return {"summary": "", "keywords": [], "intent": "", "risks": []}