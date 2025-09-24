import os
import io
import time
import uuid
import logging
import numpy as np
import pandas as pd
import google.generativeai as genai

from fastapi import Body, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi import Body


# --- Load env ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING")

DB_NAME = "test_case_db"
COLLECTION_NAME = "multilevel_test_cases_mongo"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_INDEX_NAME = "vector_index"

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- App ---
app = FastAPI(
    title="Intelligent Test Case Search API (MongoDB Edition)",
    description="Upload, enrich, and search test cases with MongoDB Atlas Vector Search.",
    version="1.2.0",
)

templates = Jinja2Templates(directory=".")

# --- Global models/clients (initialized on startup) ---
embedding_model: SentenceTransformer = None
mongo_client: MongoClient = None
db_collection = None  # <-- NEW: Global handle for the collection

# --- Simple in-memory TTL cache for search queries ---
SEARCH_CACHE: Dict[str, Tuple[float, Any]] = {}
CACHE_TTL_SECONDS = 60 * 5  # 5 minutes

# --- Configurable params ---
CANDIDATES_TO_RETRIEVE = 15
FINAL_RESULTS = 3
GEMINI_RERANK_ENABLED = True
QUERY_EXPANSION_ENABLED = True
DIVERSITY_ENFORCE = True
DIVERSITY_PER_FEATURE = True
GEMINI_RATE_LIMIT_SLEEP = 0.5

# --- Utility functions (unchanged) ---
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0


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


# --- Gemini helpers (unchanged, but could be adapted for JSON output for more robustness) ---
def configure_gemini():
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        logger.info("Gemini configured.")
    except Exception as e:
        logger.error(f"Failed to configure Gemini: {e}")


def expand_query_with_gemini(query: str) -> List[str]:
    if not QUERY_EXPANSION_ENABLED or not GOOGLE_API_KEY:
        if not GOOGLE_API_KEY:
            logger.warning("GOOGLE_API_KEY not set; skipping Gemini expansion.")
        return [query]
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""
You are an assistant that expands short search queries into useful paraphrases and synonyms for software test-case search.
Return only a comma-separated single line of 5 short paraphrases/keywords (no numbering).
Query: "{query}"
Output example: login error, authentication failure, invalid credentials, sign-in fails, cannot login
"""
        response = model.generate_content(prompt)
        text = response.text.strip()
        parts = [p.strip() for p in text.replace("\n", ",").split(",") if p.strip()]
        expansions = [query] + parts
        logger.info(f"Query expansion for '{query}': {expansions[:6]}")
        time.sleep(GEMINI_RATE_LIMIT_SLEEP if GEMINI_RERANK_ENABLED else 0)
        return expansions[:6]
    except Exception as e:
        logger.error(f"Gemini expansion failed: {e}")
        return [query]


def rerank_with_gemini(
    query: str, candidates: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if not GEMINI_RERANK_ENABLED or not GOOGLE_API_KEY:
        if not GOOGLE_API_KEY:
            logger.warning("GOOGLE_API_KEY not set; skipping Gemini rerank.")
        return candidates
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt_lines = [
            "You are a helpful assistant. Re-rank the following test cases by relevance to the query.",
            f'Query: "{query}"',
            "Return only a newline-separated list of candidate IDs (the '_id' field) in best-to-worst order. Do not add extra commentary.",
        ]
        prompt_lines.append("\nCandidates:")
        for c in candidates:
            brief = c.get("description") or c.get("summary") or ""
            prompt_lines.append(
                f"{c['_id']} | Feature: {c.get('feature','N/A')} | Desc: {brief[:220]}"
            )
        prompt = "\n".join(prompt_lines)

        response = model.generate_content(prompt)
        text = response.text.strip()
        lines = safe_parse_lines(text)
        ordered_ids = [l.split()[0].strip().strip(".").strip("-") for l in lines]

        id_to_c = {c["_id"]: c for c in candidates}
        ordered = [id_to_c[i] for i in ordered_ids if i in id_to_c]
        seen = {c["_id"] for c in ordered}
        ordered += [c for c in candidates if c["_id"] not in seen]
        logger.info("Rerank: Gemini provided an ordering.")
        time.sleep(GEMINI_RATE_LIMIT_SLEEP)
        return ordered
    except Exception as e:
        logger.error(f"Gemini rerank failed: {e}")
        return candidates


def get_gemini_enrichment(test_case_description: str, feature: str) -> dict:
    # This function remains unchanged but is called during the upload process.
    if not GOOGLE_API_KEY:
        logger.warning(
            "Google API key not present; using simple heuristics for enrichment."
        )
        summary = (
            (test_case_description[:600] + "...")
            if len(test_case_description) > 600
            else test_case_description
        )
        tokens = [
            t.strip().lower()
            for t in test_case_description.replace(".", " ").split()
            if len(t) > 3
        ]
        keywords = list(dict.fromkeys(tokens))[:20]
        return {"summary": summary, "keywords": keywords}
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""
Analyze the following software test case and generate enriched metadata:
Feature: "{feature}"
Test Case Description: "{test_case_description}"
Output format (exactly):
Summary: <3-5 lines clearly explaining purpose and process>
Keywords: <15-20+ diverse keywords/phrases, comma-separated>
"""
        response = model.generate_content(prompt)
        text = response.text.strip()
        summary, keywords = "", []
        for line in text.splitlines():
            if line.lower().startswith("summary:"):
                summary = line.split(":", 1)[1].strip()
            elif line.lower().startswith("keywords:"):
                raw_kw = line.split(":", 1)[1]
                keywords = [k.strip() for k in raw_kw.split(",") if k.strip()]
        time.sleep(0.2)
        return {"summary": summary, "keywords": keywords}
    except Exception as e:
        logger.error(f"Error calling Gemini: {e}")
        return {"summary": "", "keywords": []}


from motor.motor_asyncio import AsyncIOMotorClient
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model, mongo_client, db_collection

    # --- Startup ---
    logger.info("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info("✅ Embedding model loaded.")

    logger.info("Connecting to MongoDB Atlas (async)...")
    try:
        mongo_client = AsyncIOMotorClient(
            MONGO_CONNECTION_STRING,
            serverSelectionTimeoutMS=5000,
            tls=True,  # use TLS/SSL explicitly
        )
        # Ping MongoDB to confirm connection
        await mongo_client.admin.command("ping")
        db_collection = mongo_client[DB_NAME][COLLECTION_NAME]
        logger.info(f"✅ Connected to MongoDB | Collection='{COLLECTION_NAME}'")
        logger.warning("⚠️ Ensure a Vector Search Index is created in MongoDB Atlas.")
    except Exception as e:
        logger.error(f"❌ Failed to connect to MongoDB: {e}", exc_info=True)
        mongo_client, db_collection = None, None

    logger.info("Configuring Gemini (if key present)...")
    configure_gemini()

    yield

    # --- Shutdown ---
    if mongo_client:
        mongo_client.close()
        logger.info("🔒 MongoDB connection closed.")
    logger.info("👋 Lifespan shutdown complete.")


app = FastAPI(lifespan=lifespan)

# --- Serve frontend ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/upload")
async def upload_and_process_file(file: UploadFile = File(...)):
    if not file.filename.endswith((".csv", ".xlsx")):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a CSV or XLSX file.",
        )

    # ✅ Correct MongoDB check
    if db_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB connection not available.")

    try:
        contents = await file.read()
        buffer = io.BytesIO(contents)
        if file.filename.endswith(".csv"):
            df = pd.read_csv(buffer, encoding="utf-8")
        else:
            df = pd.read_excel(buffer)

        # Normalize and clean
        df = df.astype(str).replace(["nan", "NaN"], "")
        df["Test Case ID"] = (
            df["Test Case ID"].replace("", pd.NA).fillna(method="ffill")
        )
        df.dropna(subset=["Test Case ID"], inplace=True)
        df = df[
            df["Test Case ID"].str.strip().str.upper().ne("NA")
            & df["Test Case ID"].str.strip().ne("")
        ]
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading file: {e}")

    documents_to_insert = []
    grouped = df.groupby("Test Case ID")
    logger.info(f"Processing {len(grouped)} unique test cases...")

    for test_case_id, group in grouped:
        feature = str(group["Feature"].iloc[0])
        description = str(group["Test Case Description"].iloc[0])
        prerequisites = str(group["Pre-requisites"].iloc[0])

        # Build steps
        steps_list = []
        for _, row in group.iterrows():
            step_no = str(row.get("Step No.", "")).strip()
            test_step = str(row.get("Test Step", "")).strip()
            expected_result = str(row.get("Expected Result", "")).strip()
            if test_step:
                formatted_step = (
                    f"Step {step_no}: {test_step}" if step_no else test_step
                )
                if expected_result:
                    formatted_step += f" → Expected: {expected_result}"
                steps_list.append(formatted_step)
        steps_combined = "\n\n".join(steps_list)

        # Gemini enrichment
        enrichment = get_gemini_enrichment(description, feature)
        summary = enrichment.get("summary", "")
        keywords = enrichment.get("keywords", [])

        # Embeddings
        desc_emb = embedding_model.encode(description).tolist() if description else None
        steps_emb = (
            embedding_model.encode(steps_combined).tolist() if steps_combined else None
        )
        summary_emb = embedding_model.encode(summary).tolist() if summary else None

        valid_embeddings = [
            np.array(e) for e in [desc_emb, steps_emb, summary_emb] if e is not None
        ]
        main_vector = (
            np.mean(valid_embeddings, axis=0).tolist()
            if valid_embeddings
            else embedding_model.encode("").tolist()
        )

        # Document structure
        document = {
            "_id": str(uuid.uuid4()),  # Explicit ID
            "Test Case ID": test_case_id,
            "Feature": feature,
            "Test Case Description": description,
            "Pre-requisites": prerequisites,
            "Steps": steps_combined,
            "TestCaseSummary": summary,
            "TestCaseKeywords": keywords,
            "desc_embedding": desc_emb,
            "steps_embedding": steps_emb,
            "summary_embedding": summary_emb,
            "main_vector": main_vector,  # Vector to be indexed
        }
        documents_to_insert.append(document)

    if not documents_to_insert:
        return {"message": "No valid test cases found to process in the file."}

    try:
        # ⚡ Async insert using Motor
        result = await db_collection.insert_many(documents_to_insert)
        logger.info(f"Inserted {len(result.inserted_ids)} documents.")
        return {
            "message": f"Successfully processed and stored {len(result.inserted_ids)} test cases."
        }
    except Exception as e:
        logger.error(f"Error inserting into MongoDB: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error storing data: {e}")


# --- Get All with Pagination ---
@app.get("/api/get-all")
async def get_all_test_cases(
    skip: int = 0, limit: int = 50, sort_by: str = "created_at", order: int = -1
):
    """
    Retrieve test cases with pagination and sorting.
    - skip: number of documents to skip (default: 0)
    - limit: number of documents to return (default: 50)
    - sort_by: field to sort on (default: created_at)
    - order: -1 for descending, 1 for ascending (default: -1)
    """
    if db_collection is None:  # ✅ Correct None check
        raise HTTPException(status_code=503, detail="MongoDB connection not available.")

    try:
        # Projection to exclude large/vector fields
        projection = {
            "_id": 1,
            "main_vector": 0,
            "desc_embedding": 0,
            "steps_embedding": 0,
            "summary_embedding": 0,
        }

        # Validate order
        sort_order = -1 if order < 0 else 1

        # Async cursor using Motor
        cursor = (
            db_collection.find({}, projection)
            .sort(sort_by, sort_order)
            .skip(skip)
            .limit(limit)
        )

        test_cases = []
        async for doc in cursor:
            doc["id"] = str(doc["_id"])
            del doc["_id"]  # replace ObjectId with string
            test_cases.append(doc)

        return {
            "success": True,
            "count": len(test_cases),
            "skip": skip,
            "limit": limit,
            "test_cases": test_cases,
        }

    except Exception as e:
        logger.error(f"Error retrieving test cases from MongoDB: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="An error occurred while retrieving data."
        )


# --- Delete all (recreate) ---
@app.post("/api/delete-all")
async def delete_all_data(confirm: bool = False):
    """
    Delete all test case data from the MongoDB collection.
    - confirm: must be True to actually delete data (safety check).
    """
    if db_collection is None:  # ✅ Correct None check
        raise HTTPException(status_code=503, detail="MongoDB connection not available.")

    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Confirmation required. Pass ?confirm=true to delete all data.",
        )

    try:
        # Drop the entire collection asynchronously
        await db_collection.drop()
        logger.warning(f"⚠️ Dropped collection '{COLLECTION_NAME}'; all data cleared.")

        return {
            "success": True,
            "collection": COLLECTION_NAME,
            "message": f"All test case data in '{COLLECTION_NAME}' has been successfully deleted.",
        }

    except Exception as e:
        logger.error(
            f"❌ Failed to delete all data from '{COLLECTION_NAME}': {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail="An error occurred while deleting data."
        )


# --- NEW: Update endpoint ---
class UpdateTestCaseRequest(BaseModel):
    feature: str | None = None
    description: str | None = None
    prerequisites: str | None = None
    steps: str | None = None


@app.put("/api/update/{doc_id}")
async def update_test_case(doc_id: str, update_data: UpdateTestCaseRequest = Body(...)):
    if db_collection is None:  # ✅ Correct None check
        raise HTTPException(status_code=503, detail="MongoDB connection not available.")

    try:
        # 🔍 Fetch existing document
        existing_doc = await db_collection.find_one({"_id": doc_id})
        if existing_doc is None:
            raise HTTPException(status_code=404, detail="Test case not found")

        # 📝 Update only provided fields
        updated_doc = existing_doc.copy()
        if update_data.feature is not None:
            updated_doc["Feature"] = update_data.feature
        if update_data.description is not None:
            updated_doc["Test Case Description"] = update_data.description
        if update_data.prerequisites is not None:
            updated_doc["Pre-requisites"] = update_data.prerequisites
        if update_data.steps is not None:
            updated_doc["Steps"] = update_data.steps

        # 🔮 Re-enrich if description or feature changed
        if update_data.description or update_data.feature:
            enrichment = get_gemini_enrichment(
                updated_doc.get("Test Case Description", ""),
                updated_doc.get("Feature", ""),
            )
            updated_doc["TestCaseSummary"] = enrichment.get("summary", "")
            updated_doc["TestCaseKeywords"] = enrichment.get("keywords", [])

        # 🎯 Recompute embeddings
        desc_emb = embedding_model.encode(
            updated_doc.get("Test Case Description", "")
        ).tolist()
        steps_emb = embedding_model.encode(updated_doc.get("Steps", "")).tolist()
        summary_emb = embedding_model.encode(
            updated_doc.get("TestCaseSummary", "")
        ).tolist()

        valid_embeddings = [
            np.array(e) for e in [desc_emb, steps_emb, summary_emb] if e is not None
        ]
        main_vector = (
            np.mean(valid_embeddings, axis=0).tolist()
            if valid_embeddings
            else embedding_model.encode("").tolist()
        )

        updated_doc.update(
            {
                "desc_embedding": desc_emb,
                "steps_embedding": steps_emb,
                "summary_embedding": summary_emb,
                "main_vector": main_vector,
            }
        )

        # 💾 Replace with updated version asynchronously
        await db_collection.replace_one({"_id": doc_id}, updated_doc)

        logger.info(f"✅ Updated test case {doc_id}")

        # 🧹 Return updated document without embeddings
        response_doc = updated_doc.copy()
        for field in [
            "desc_embedding",
            "steps_embedding",
            "summary_embedding",
            "main_vector",
        ]:
            response_doc.pop(field, None)

        return {
            "success": True,
            "message": f"Test case {doc_id} updated successfully",
            "updated_test_case": response_doc,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error updating test case {doc_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="An error occurred while updating the test case."
        )


# --- Enhanced async search endpoint ---
@app.post("/api/search")
async def search_test_cases(request: Request):
    if db_collection is None:  # ✅ Correct None check
        raise HTTPException(status_code=503, detail="MongoDB connection not available.")

    data = await request.json()
    raw_query = (data.get("query") or "").strip()
    filter_feature = (data.get("feature") or "").strip() or None

    if not raw_query:
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    # --- Cache check ---
    cache_key = f"{raw_query}::feature={filter_feature}"
    if cached := cache_get(cache_key):
        logger.info(
            f"⚡ Returning cached results for '{raw_query}' (feature={filter_feature})"
        )
        return {**cached, "from_cache": True}

    logger.info(f"🔍 Search query='{raw_query}' | feature filter='{filter_feature}'")

    # --- Expand Query with Gemini ---
    expansions = expand_query_with_gemini(raw_query)
    combined_query = " ".join(expansions)
    query_vector = embedding_model.encode(combined_query).tolist()

    # --- MongoDB Vector Search ---
    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": "main_vector",
                "queryVector": query_vector,
                "numCandidates": 150,
                "limit": CANDIDATES_TO_RETRIEVE,
            }
        },
        {"$project": {"score": {"$meta": "vectorSearchScore"}, "document": "$$ROOT"}},
    ]
    if filter_feature:
        pipeline[0]["$vectorSearch"]["filter"] = {"Feature": {"$eq": filter_feature}}

    try:
        search_results = await db_collection.aggregate(pipeline).to_list(
            length=CANDIDATES_TO_RETRIEVE
        )
    except Exception as e:
        logger.error(f"❌ MongoDB vector search error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Search failed due to database error."
        )

    # --- Local reranking ---
    candidates = []
    for res in search_results:
        payload = res.get("document", {})
        base_score = float(res.get("score", 0.0))

        def safe_sim(vec1, vec2):
            if not vec1 or not vec2:
                return 0.0
            return cosine_similarity(np.array(vec1), np.array(vec2))

        desc_emb = payload.get("desc_embedding", [])
        steps_emb = payload.get("steps_embedding", [])
        summary_emb = payload.get("summary_embedding", [])

        sim_desc = safe_sim(query_vector, desc_emb)
        sim_steps = safe_sim(query_vector, steps_emb)
        sim_summary = safe_sim(query_vector, summary_emb)

        keywords = [k.lower() for k in payload.get("TestCaseKeywords", [])]
        text_fields = f'{payload.get("Feature", "")} {payload.get("Test Case Description", "")} {payload.get("Steps", "")}'.lower()

        token_boost = sum(
            0.12 for term in expansions if term.lower().strip() in text_fields
        )
        token_boost += sum(
            0.18 for term in expansions if term.lower().strip() in keywords
        )

        local_score = (
            (0.6 * base_score)
            + (0.25 * max(sim_desc, sim_steps, sim_summary))
            + token_boost
        )

        candidates.append(
            {
                "_id": payload.get("_id"),
                "raw_score": base_score,
                "local_score": local_score,
                "feature": payload.get("Feature", "N/A"),
                "test_case_id": payload.get("Test Case ID", "NA"),
                "description": payload.get("Test Case Description", ""),
                "summary": payload.get("TestCaseSummary", ""),
                "keywords": payload.get("TestCaseKeywords", []),
                "payload": payload,
            }
        )

    # --- Rerank with Gemini ---
    candidates.sort(key=lambda x: x["local_score"], reverse=True)
    top_candidates = candidates[:CANDIDATES_TO_RETRIEVE]
    reranked = rerank_with_gemini(raw_query, top_candidates)

    # --- Apply diversity ---
    final_list = []
    seen_features = set()
    for cand in reranked:
        if len(final_list) >= FINAL_RESULTS:
            break
        if DIVERSITY_ENFORCE and DIVERSITY_PER_FEATURE:
            feat = cand.get("feature") or "N/A"
            if feat in seen_features:
                continue
            final_list.append(cand)
            seen_features.add(feat)
        else:
            final_list.append(cand)

    # --- Backfill if needed ---
    if len(final_list) < FINAL_RESULTS:
        seen_ids = {c["_id"] for c in final_list}
        for cand in reranked:
            if len(final_list) >= FINAL_RESULTS:
                break
            if cand["_id"] not in seen_ids:
                final_list.append(cand)

    # --- Build response ---
    response_items = []
    for c in final_list:
        score_pct = round(min(max(c["local_score"], 0.0), 1.0) * 100, 2)
        payload = c.get("payload", {})
        response_items.append(
            {
                "id": c["_id"],
                "probability": score_pct,
                "test_case_id": payload.get("Test Case ID", "NA"),
                "feature": payload.get("Feature", "N/A"),
                "description": payload.get("Test Case Description", ""),
                "prerequisites": payload.get("Pre-requisites", ""),
                "steps": payload.get("Steps", ""),
                "summary": payload.get("TestCaseSummary", ""),
                "keywords": payload.get("TestCaseKeywords", []),
            }
        )

    result = {
        "query": raw_query,
        "feature_filter": filter_feature,
        "results_count": len(response_items),
        "results": response_items,
        "from_cache": False,
    }

    cache_set(cache_key, result)
    logger.info(f"✅ Returning {len(response_items)} results for query '{raw_query}'")
    return result


# --- End of file ---
