## Run

```bash
python parse.py
```

This generates both `./src/assets/docs.json` (for the frontend) and `./public/docs.md` (for AI agents).

To skip markdown generation:

```bash
python parse.py -m ""
```

## Notes

### Site config

The docs site is environment-driven. `vite` now generates `public/robots.txt`, `public/sitemap.xml`, and
`public/llms.txt`, and injects canonical/Open Graph/Twitter metadata into `index.html` from environment
variables at build time.

- in `ptspy`, it defaults to production (`https://ptspy.org`, indexable)

```bash
pnpm build
```

Variables:

- `VITE_SITE_URL`: Deployed host for this build
- `VITE_CANONICAL_URL`: Canonical public host
- `VITE_INDEXABLE`: `true` for production, `false` for staging
- `VITE_REPO_URL`: Repository URL used in generated `llms.txt`

1. Check `base` path in `vite.config.ts`

2. Some basic markdown elements are pre-parsed and incorporated in the JSON as objects. They are:

| Markdown                   | JSON                                      |
| -------------------------- | ----------------------------------------- |
| \`x=1\`                    | `{ "code": "x=1" }`                       |
| `![alt text](image_url)`   | `{ "image": ["image_url", "alt_text"] }`  |
| `[link text](url)`         | `{ "link": ["url", "link text"] }`        |
| `[link text](`reference`)` | `{ "ref": ["link text", ["reference"]] }` |

3. `Data` section lets you add arbitrary key/value pairs to the docstring. For example: "internal: true" can make a function as an internal function

```
Data:
    internal: true
    key2: value2
```
