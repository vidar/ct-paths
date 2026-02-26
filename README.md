# Contentstack Reference Graph Generator

Generates PNG reference tree diagrams for Contentstack content types. Each diagram shows the full reference tree from a content type's perspective — what it references, what those reference, and so on down to the leaf nodes.

Only content types that have a `url` field in their schema get a diagram generated (these are the routable page types).

## Prerequisites

- Python 3.10+
- matplotlib (`pip install matplotlib`)

## Fetching content types

Export your content types from the Contentstack Management API:

```bash
curl --location 'https://eu-api.contentstack.com/v3/content_types?include_count=false&include_global_field_schema=true&include_branch=false' \
--header 'api_key: <apikey>' \
--header 'authorization: <managementtoken>' \
--header 'Content-Type: application/json' \
```

The script accepts either the full API response (with a `content_types` wrapper) or a plain JSON array of content types.

## Usage

```bash
python generate_ref_graphs.py --input types.json --output ./graphs
```

This will:

1. Parse all content types and discover reference edges (including references nested inside groups, modular blocks, and global fields)
2. Filter to only content types whose schema contains a `uid: "url"` field
3. Build a reference tree for each, detecting and marking cycles
4. Render one PNG per content type into the output directory
5. Generate a `_index.md` summary with stats and links to each diagram

## Output

- `graphs/{ct_uid}_reference_tree.png` — one diagram per content type with a URL field
- `graphs/_index.md` — summary table of all generated diagrams

## How it works

The script recursively walks each content type's schema to extract reference edges:

| Field type | `data_type` | Where refs are found |
|---|---|---|
| Direct reference | `reference` | `reference_to` array (ignores `sys_assets`) |
| Group field | `group` | Recurses into nested `schema` |
| Modular blocks | `blocks` | Recurses into each block's `schema` |
| Global field | `global_field` | Recurses into inline `schema` |

For each content type with a URL field, it builds a tree by following outgoing references. Cycles (e.g., Page -> Hero Banner -> Page) are detected and rendered as dashed red nodes without further expansion.

### Node colors

- **Blue** — Root node (the content type being diagrammed)
- **Green** — Intermediate node (has further outgoing references)
- **Purple** — Leaf node (no outgoing references)
- **Red dashed** — Cycle (already visited on this path)
