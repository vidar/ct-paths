# Contentstack Reference Graph

Two ways to view the reference tree for a Contentstack content type:

1. **`app/index.html`** — a single-page Contentstack Developer Hub app (Full Page location) that fetches content types live from the stack and renders the tree in-browser.
2. **`generate_ref_graphs.py`** — an offline script that builds the same tree from an exported content-types JSON file and writes a standalone HTML file.

## Full Page App (recommended)

A single-file HTML app (`app/index.html`) that runs inside Contentstack. No build step, no server code — just static hosting.

### What it does

On load, it:

1. Initializes `@contentstack/app-sdk` inside the Contentstack iframe.
2. Fetches all content types via `fullPage.stack.getContentTypes()` (paginated, with `include_global_field_schema: true`).
3. Walks the schemas to extract reference edges and builds the full tree.
4. Renders an interactive D3 tree with a picker for the root content type.

### Register it in Developer Hub

1. Go to Contentstack Developer Hub → **Create App**.
2. Add a **Full Page** UI location.
3. Set the base URL to wherever you host `app/`.
4. Install the app on your stack and open it from the sidebar.

For local development, serve the folder from any static server, e.g.:

```bash
cd app
python3 -m http.server 3000
```

Then point the app's base URL at `http://localhost:3000` (or a tunnel URL for HTTPS).

### Controls

- **Root** — pick any content type; types with a `url` field are marked with `★`
- **Depth** — set the visible depth; use +/- or type a number
- **Expand All / Collapse All** — jump to max or min depth
- **Top-Down / Left-Right** — switch tree orientation
- **Ignore** — exclude up to 5 content types from the tree (click a tag to remove)
- **Leaf counter** — shown while the tree is uniformly expanded to a fixed depth
- **Pan and zoom** — scroll to zoom, drag to pan; *Reset View* re-fits the tree

### Node colors

- **Blue** — root node
- **Green** — intermediate node (has further outgoing references)
- **Purple** — leaf node (no outgoing references)
- **Red dashed** — cycle (already visited on this path)

## Offline script

`generate_ref_graphs.py` remains available for ad-hoc use without installing the app. It takes an exported content-types JSON and writes a self-contained HTML file.

### Prerequisites

- Python 3.10+
- matplotlib (`pip install -r requirements.txt`)

### Export content types

```bash
curl --location 'https://eu-api.contentstack.com/v3/content_types?include_count=false&include_global_field_schema=true&include_branch=false' \
  --header 'api_key: <apikey>' \
  --header 'authorization: <managementtoken>' \
  --header 'Content-Type: application/json'
```

The script accepts either the full API response (with a `content_types` wrapper) or a plain JSON array.

### Usage

```bash
python generate_ref_graphs.py --input types.json --output ./graphs
```

Outputs `graphs/page_wrapper_reference_tree.html` and `graphs/_index.md`.

## How reference extraction works

Both the app and the script recursively walk each content type's schema:

| Field type | `data_type` | Where refs are found |
|---|---|---|
| Direct reference | `reference` | `reference_to` array (ignores `sys_assets`) |
| Group field | `group` | Recurses into nested `schema` |
| Modular blocks | `blocks` | Recurses into each block's `schema` |
| Global field | `global_field` | Recurses into inline `schema` (requires `include_global_field_schema`) |

Cycles (e.g., Page → Hero Banner → Page) are detected and rendered as dashed red nodes without further expansion.
