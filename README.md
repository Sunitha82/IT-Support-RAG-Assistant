# IT Support RAG Assistant

An IT support assistant that answers questions using company documents, displays
supporting sources, masks recognised personal information, and traces requests
with Langfuse.

## What users can ask

The current knowledge base covers:

- VPN troubleshooting and internal application access.
- Employee network password policies.
- Software-access requests and approvals.
- Remote-working requirements.
- IT helpdesk support.

**Network passwords and Wi-Fi passwords are different topics.** The current
documents contain employee network password guidance, but no Wi-Fi password policy.
The welcome screen displays topics from the current index and provides clickable
example questions.

## How it works

```text
User question
    |
Input guardrail
    |
Input PII masking
    |
Multi-query semantic retrieval
    |
Reciprocal Rank Fusion -> Cross-encoder reranking
    |
Filtering and parent-section selection
    |
Retrieved-context PII masking
    |
Azure OpenAI generates an answer
    |
Output PII masking
    |
Answer, sources, retrieval details and trace ID
```

If no evidence passes the retrieval filters, the app returns an
insufficient-information message and skips the model call.

## Technology stack

| Component | Technology |
| --- | --- |
| Frontend | Streamlit |
| Backend | FastAPI |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector storage | FAISS; Chroma also supported for standard retrieval |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Answer generation | Azure OpenAI |
| PII detection and masking | Microsoft Presidio |
| Observability | Langfuse |
| Hosting | Azure Container Apps |
| Container images | Azure Container Registry |
| Credentials | Azure Key Vault and managed identity |
| Build and deployment | Azure DevOps pipeline |

## Search methods

The enhanced pipeline uses **multi-query semantic vector retrieval**, rather than
BM25 or keyword-plus-vector hybrid search.

1. Create embeddings representing the meaning of the question.
2. Search the original question and related variants.
3. Apply explicit IT terminology expansion where applicable.
4. Combine the ranked result lists using Reciprocal Rank Fusion (RRF).
5. Rerank candidate chunks using a cross-encoder.
6. Filter weak results and return their complete parent sections.

In the current parent-child implementation, FAISS stores the vectors; search
reconstructs them and calculates exact cosine scores locally. IT terminology
expansion uses explicit phrase mappings, not an LLM-generated rewrite.

### Understanding RAG Insights

| Field | Meaning |
| --- | --- |
| Matched child | Small document chunk found during search |
| Parent / Chunk ID | Larger section supplied as context |
| Similarity | Cosine similarity of the matched child vector |
| Distance | `1 - cosine similarity`; lower means closer |
| RRF score | Combined ranking signal across query variants |
| Reranker score | Cross-encoder relevance score |
| Candidate count | Number of possible matches considered |
| Top K | Maximum number of final results requested |

**These scores are not answer-confidence percentages.** The UI similarity
threshold changes highlighting only; it does not change retrieval or model
context. The cosine score retained during fusion can come from a search variant;
it is not a new similarity calculation against the entire parent section.

## Project structure

```text
app/
  api/              API routes, authentication and response schemas
  core/             Configuration, logging and Key Vault integration
  guardrails/       Input validation and PII masking
  ingestion/        Document loading and chunking
  retrieval/        Vector stores, multi-query retrieval and reranking
  llm/              Azure OpenAI client and prompts
  observability/    Langfuse integration
  services/         RAG orchestration
  ui/               Streamlit interface, styling and topic catalogue

data/               Source documents
scripts/            Indexing, validation and pause/resume utilities
tests/              Automated tests
evaluation/         Evaluation utilities
.streamlit/         UI theme configuration
azure-pipelines.yml Azure DevOps build and deployment pipeline
```

## Local setup

### Prerequisites

- Python 3.13.
- Docker Desktop for container-based development.
- Azure CLI for Azure authentication and lifecycle commands.
- Access to the configured Azure OpenAI deployment.

Run the commands below from the project root in PowerShell.

### Install dependencies

For a new checkout (reuse your existing virtual environment if already set up):

```powershell
py -3.13 -m venv .venv

.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m spacy download en_core_web_lg
```

CPU-only PyTorch avoids downloading unnecessary CUDA packages. The first setup
downloads sizeable model files and can take several minutes.

### Configure the environment

Create `.env` from `.env.example` **only if you do not already have one**:

```powershell
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
}
```

Configure the values for your deployment:

```dotenv
USE_AZURE_KEY_VAULT=false

VECTOR_STORE_PROVIDER=faiss
RETRIEVAL_MODE=enhanced

RAG_API_KEY=<your-local-api-key>

AZURE_OPENAI_ENDPOINT=<your-endpoint>
AZURE_OPENAI_DEPLOYMENT=<your-deployment>
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_API_VERSION=2024-10-21

LANGFUSE_ENABLED=false
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_PUBLIC_KEY=<your-project-public-key>
LANGFUSE_SECRET_KEY=<your-project-secret-key>
```

Never commit `.env` or real credentials. The UI and API must use the same RAG API
key. Enable Langfuse only after its configuration is complete.

### Index the documents

For local development using credentials from `.env`:

```powershell
$env:USE_AZURE_KEY_VAULT = "false"
$env:DOCUMENT_SOURCE = "local"
$env:VECTOR_STORE_PROVIDER = "faiss"

.\.venv\Scripts\python.exe -m scripts.index_documents
.\.venv\Scripts\python.exe -m scripts.index_parent_child
```

The first command builds the selected standard store. Enhanced retrieval uses
the separate parent-child index built by the second command.

### Start the API

In the same terminal:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

### Start the UI

In another terminal, launch Streamlit with `.env` loaded:

```powershell
$env:USE_AZURE_KEY_VAULT = "false"
$env:RAG_API_URL = "http://127.0.0.1:8000"
.\.venv\Scripts\python.exe -c "from dotenv import load_dotenv; load_dotenv(); from streamlit.web.cli import main; import sys; sys.argv=['streamlit','run','app/ui/streamlit_app.py','--server.port=8501']; main()"
```

Open:

- UI: <http://localhost:8501>
- API documentation: <http://localhost:8000/docs>

The UI does not independently load `.env` when launched with a plain
`streamlit run` command; the wrapper above loads it before starting Streamlit.

## Docker Compose

The existing Compose configuration is tailored to this project:

- API: <http://localhost:8001>
- UI: <http://localhost:8501>
- UI-to-API connection: `http://api:8000`.
- API-side Key Vault fetching is explicitly enabled.
- Enhanced retrieval uses the bind-mounted `faiss_azure_test_db` directory.

Before starting, configure Key Vault authentication, including the tenant and
client credentials needed by your chosen identity. The UI's `RAG_API_KEY` from
`.env` must match the API key fetched from Key Vault. The Compose file overrides
`USE_AZURE_KEY_VAULT` to `true`, regardless of the `.env` development setting.

The mounted `faiss_azure_test_db` must contain `parents.json` and its valid
`children` FAISS index. A new empty bind mount will not contain the index baked
into `/app/faiss_parent_child_db`.

To generate the mounted parent-child index locally using the configured source:

```powershell
$env:PARENT_CHILD_FAISS_DIR = Join-Path $PWD "faiss_azure_test_db"
.\.venv\Scripts\python.exe -m scripts.index_parent_child
Remove-Item Env:PARENT_CHILD_FAISS_DIR
```

Then start the containers:

```powershell
docker compose up --build -d
docker compose ps
```

Stop the local containers while retaining volumes:

```powershell
docker compose down
```

## API endpoints

| Endpoint | Purpose | Authentication |
| --- | --- | --- |
| `GET /health` | Confirms the API process responds | None |
| `GET /ready` | Checks the selected knowledge-base index | None |
| `GET /topics` | Lists topics from indexed documents | `X-API-Key` |
| `POST /ask` | Retrieves evidence and generates an answer | `X-API-Key` |

**A successful `/health` response alone does not prove retrieval or answer
generation works.** Readiness checks the index, not every model or external
dependency. Test a supported question after startup.

## Adding or updating documents

Uploading documents to Blob Storage alone does **not** update the deployed
knowledge base. The current deployment builds indexes into the API image from
the local `data/` directory.

For the existing workflow:

1. Add or update UTF-8 `.txt` documents in `data/`.
2. Commit the changes to Azure Repos through a branch and pull request.
3. Let the pipeline rebuild the indexes and images.
4. Deploy and verify retrieval using representative questions.

This is **document ingestion and indexing**, not model training.

Both indexing scripts support `DOCUMENT_SOURCE=azure_blob`. Their current Blob
source is `stitragdocs2026` / `it-support-documents`. That path needs explicit
configuration, authentication and execution; there is no automatic upload watcher.
The Dockerfile currently sets `DOCUMENT_SOURCE=local` for its build-time indexing.

For document removal or significant restructuring, use a fresh index build to
avoid retaining stale entries in an existing local store. For newly added topic
documents, review their title, example question and description in
`app/ui/topic_catalog.py`; unknown indexed sources receive a generic entry.

## Azure deployment

The pipeline builds from **Azure Repos -> IT-RAG-Demo -> `main`**.

The GitHub repository and Azure Repos repository are separate copies. Pushing
to GitHub does not automatically update Azure Repos.

Check your remotes:

```powershell
git remote -v
```

If the Azure remote is missing, add it once:

```powershell
git remote add azure https://dev.azure.com/AzureKans/IT-RAG-Demo/_git/IT-RAG-Demo
```

Commit only the intended files and push a reviewed branch:

```powershell
git push -u azure <your-branch-name>
```

Create a pull request into `main`. Completing the merge triggers the configured
pipeline. Avoid blindly staging the whole working directory: it may contain
local outputs, diagrams or unrelated work.

### Pipeline flow

```text
Checkout Azure Repos main
    -> Build API and UI images
    -> Push images to Azure Container Registry
    -> Configure Langfuse Key Vault references
    -> Update API and UI Container Apps
```

Images use the Azure DevOps build ID as their tag. Verify the source commit and
the final API/UI image tags after each deployment. The current pipeline updates
existing apps; it is not a complete infrastructure-provisioning script.

### Existing Azure resources

| Resource | Name |
| --- | --- |
| Resource group | `rag-demo-rg` |
| Container Apps environment | `cae-it-rag-demo` |
| API app | `ca-it-rag-api` |
| UI app | `ca-it-rag-ui` |
| Container registry | `acritrag2026` |
| Key Vault | `kv-it-rag-2026` |
| Managed identity | `id-it-rag-acr-pull` |

The API has internal ingress; the UI has external ingress. The pipeline configures
the API with 1 CPU and 2 GiB RAM and keeps one replica available while active.

### Required Key Vault secrets

- `RAG-API-KEY`
- `AZURE-OPENAI-KEY`
- `langfuse-public-key`
- `langfuse-secret-key`

The assigned managed identity must have permission to read the referenced
secrets. The Langfuse keys must belong to the same project and region.

In Azure, `USE_AZURE_KEY_VAULT=false` disables **Python-side** fetching. Container
Apps still resolves its configured Key Vault secret references and supplies the
values as environment variables.

The UI uses `RAG_API_URL` and `RAG_API_KEY`. Legacy `API_URL` / `API_KEY` variables
do not configure the current UI.

## Pause and resume Azure apps

Pausing deactivates revisions without deleting apps, the environment, identities,
secrets or URLs. It does not recreate deleted resources.

### Check status

```powershell
az login
.\scripts\Manage-RagDemo.ps1 -Action Status
```

### Pause

```powershell
.\scripts\Manage-RagDemo.ps1 -Action Pause -WhatIf
.\scripts\Manage-RagDemo.ps1 -Action Pause
```

### Resume

```powershell
.\scripts\Manage-RagDemo.ps1 -Action Resume
```

The script stops the UI before the API. On resume, it starts the API first,
waits for its health status, then starts and checks the UI. Ask a supported VPN
question to verify the complete flow after resuming.

Keep `.local/rag-demo/state.json`. It stores revision metadata, not secret
values. Do not commit it. A new checkout will not have this machine-local file.
The script refuses to resume an outdated snapshot after a newer deployment; a
fresh Pause can capture the currently active revisions instead.

### Latest saved working deployment

As of **16 November 2026**, both apps were paused with no replicas:

| App | Saved revision | Image tag |
| --- | --- | --- |
| API | `ca-it-rag-api--0000015` | `684` |
| UI | `ca-it-rag-ui--0000009` | `684` |

These values change after later deployments. Prefer the saved state and live
status over old revision numbers in screenshots.

Manual activation is also possible through **Revisions and replicas -> Inactive
revisions**. Start the matching API revision first, followed by the UI. To view
inactive revisions from the CLI, include `--all` when listing revisions.

Avoid running deployment pipelines while paused: a deployment can create an
active revision and restart compute usage. Allow 15-30 minutes before a demo for
startup and a manual test.

The apps use the Consumption profile. Inactive revisions stop running replicas;
registry, storage, Key Vault and logging resources remain and can still incur
their normal charges. This is not a promise of a zero total Azure bill.

## PII and guardrails

- Recognised PII is masked in the question before retrieval and model use.
- Retrieved evidence is masked before model use and display.
- Generated answers are checked again before being returned.
- The UI stores and displays the API's masked question.
- Recognised prompt-injection patterns are rejected before retrieval.
- Unsupported questions return a fallback when no evidence passes filtering.

PII detection is best effort. Pattern-based injection checks are not a complete
security defence. Input masking occurs on the server after submission; it does
not prevent the original text from reaching the API.

The app does not reset passwords, grant access, change settings or create tickets
for the user. It explains guidance from the available documents.

## Langfuse

The configured destination is <https://cloud.langfuse.com>.

For a successful request:

1. Open **Privacy checks and tracing** in the UI.
2. Copy its trace ID.
3. Find that exact trace in the correct Langfuse project.
4. Inspect stage counts, outputs and durations.

The UI's **queued** status does not guarantee delivery. Traces record counts,
timings, status and identifiers, not raw question, document or answer text. Token
usage and cost tracking are not currently instrumented.

Optional local connectivity test, after configuring credentials:

```powershell
.\.venv\Scripts\python.exe test_langfuse_trace.py --send-test
```

This explicitly sends a metadata-only test trace. Running the script without
the flag sends nothing.

### Trace stages

`it-support-rag-request` is the root observation. Its child observations include:

- `input-guardrail`
- `input-pii-masking`
- `retrieval`, containing multi-query search, reranking and parent selection
- `context-pii-masking`
- `azure-openai-generation`
- `output-pii-masking`

Blocked requests stop at the guardrail. Requests with no evidence skip generation.
An Error status for a deliberately blocked input is not necessarily an outage.
Nested observation durations should not be added to parent durations as if they
were independent operations.

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Additional utilities:

- `scripts/evaluate_demo_retrieval.py`: real-model retrieval regression checks.
- `scripts/verify_demo_api.py`: offline API smoke test using real retrieval and
  masking, controlled LLM output, and in-memory tracing.

The model files must already be cached for offline execution. Unit/service tests
use mocked external clients or an in-memory trace exporter instead of production
requests. The smoke test requires the Langfuse dependency and a valid local index.

The latest UI validation completed in Linux with **114 tests passed and one
platform-specific test skipped**. A subsequent targeted UI/topic rerun passed
all eight tests. The latest header-spacing fix was also checked visually.

If Windows blocks an optional native dependency, use an approved Linux/WSL
environment or container-based validation. Do not disable system security
protections. An unrelated optional pytest plugin can also fail before project
tests start; isolate the test environment or disable optional plugin autoloading
for that test process.

## Eight presentation scenarios

1. Direct VPN question and document grounding.
2. Paraphrased VPN question.
3. Multi-query search, RRF, reranking and parent-child retrieval.
4. Unsupported-question fallback.
5. Prompt-injection rejection.
6. Input PII masking.
7. Output PII masking.
8. Langfuse traceability and timing.

The live output-masking check found zero detections. Actual replacement of a
fictional email in generated output was verified separately using controlled
local tests. A stage running with zero detections is not proof that every kind
of PII was absent.

## Evaluation metrics

The walkthrough verified selected behaviours, not a complete benchmark.

| Metric | Current position |
| --- | --- |
| Retrieval Recall@k | Expected sources checked for selected questions; aggregate recall not calculated |
| Answer correctness | Selected answers manually compared with documents |
| Faithfulness | Selected answers inspected against retrieved context |
| Answer relevancy | Manually reviewed for selected questions |
| Fallback behaviour | Unsupported example verified to skip model generation |
| Response latency | Individual traces inspected; one request took 1.98 seconds |

A formal evaluation needs a verified dataset, labelled sources, reference
answers, a consistent scoring method, and repeated latency measurements including
median/p95 and cold/warm starts. Evaluation utilities exist in `evaluation/`, but
they were not reviewed or run as a formal benchmark during this walkthrough.

**Cosine, RRF and reranker scores are not substitutes for these evaluation
metrics.**

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| UI works, but questions fail | API logs, active revision, authentication and `/ready` |
| Knowledge base unavailable | Correct index path, index contents and directory permissions |
| Embedding model unavailable offline | Correct image, `HF_HOME`, and baked-in model cache |
| Old behaviour after deployment | Image tag, active revision, traffic routing and pipeline source commit |
| Topic cards unavailable | API `/topics`, UI API URL and API-key configuration; use Refresh available topics |
| Resume says a revision is missing | Include inactive revisions using `--all`; inspect the saved state |
| No Langfuse trace | Enable flag, project keys, base URL, Key Vault access and exporter logs |
| Guardrail trace shows Error | May be the deliberate rejection; inspect the stage and UI message |
| UI says BELOW threshold | Display highlighting only; inspect backend filtering and evidence |
| Welcome tagline is clipped | Confirm the current UI image includes the updated header spacing |

Do not rebuild, delete resources or change credentials before identifying the
failing stage. Use the request's support reference or trace ID to correlate the
failure with logs.
