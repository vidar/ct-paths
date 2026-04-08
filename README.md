# Contentstack Reference Graph Generator

Generates an interactive HTML reference tree for the `page_wrapper` content type. The diagram shows the full reference chain — what it references, what those reference, and so on — with click-to-expand navigation and depth controls.

## Prerequisites

- Python 3.10+
- matplotlib (`pip install matplotlib`)

## Fetching content types

Export your content types from the Contentstack Management API:

```bash
curl --location 'https://eu-api.contentstack.com/v3/content_types?include_count=false&include_global_field_schema=true&include_branch=false' \
--header 'api_key: <apikey>' \
--header 'authorization: <managementtoken>' \
--header 'Content-Type: application/json'
```

The script accepts either the full API response (with a `content_types` wrapper) or a plain JSON array of content types.

## Usage

```bash
python generate_ref_graphs.py --input types.json --output ./graphs
```

This will:

1. Parse all content types and discover reference edges (including references nested inside groups, modular blocks, and global fields)
2. Build the full reference tree for `page_wrapper`, detecting and marking cycles
3. Generate an interactive HTML file and a `_index.md` summary

## Output

- `graphs/page_wrapper_reference_tree.html` — interactive tree diagram
- `graphs/_index.md` — summary table with stats

## Interactive HTML features

- **Click to expand/collapse** — click any intermediate node to toggle its children
- **Depth control** — set a global depth with the +/- buttons or type a number; the whole tree expands or contracts to that level
- **Leaf counter** — shows the number of leaf nodes at the current depth setting (hides when nodes are manually expanded/collapsed, since the tree no longer reflects a uniform depth)
- **Ignore filter** — select up to 5 content types to exclude from the tree entirely, both visually and for leaf counting; click a tag to remove it
- **Pan and zoom** — scroll to zoom, drag to pan
- **Reset View** — re-centers and fits the tree to the viewport
- **Adaptive spacing** — deeper levels of the tree are rendered more compactly to keep the diagram from stretching too wide

## How it works

The script recursively walks each content type's schema to extract reference edges:

| Field type | `data_type` | Where refs are found |
|---|---|---|
| Direct reference | `reference` | `reference_to` array (ignores `sys_assets`) |
| Group field | `group` | Recurses into nested `schema` |
| Modular blocks | `blocks` | Recurses into each block's `schema` |
| Global field | `global_field` | Recurses into inline `schema` |

For each content type, it builds a tree by following outgoing references. Cycles (e.g., Page -> Hero Banner -> Page) are detected and rendered as dashed red nodes without further expansion.

### Node colors

- **Blue** — Root node (the content type being diagrammed)
- **Green** — Intermediate node (has further outgoing references)
- **Purple** — Leaf node (no outgoing references)
- **Red dashed** — Cycle (already visited on this path)
