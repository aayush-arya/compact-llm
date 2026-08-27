# Frontend — CompactLLM

Next.js (App Router) + TypeScript + Tailwind v4 + shadcn/ui. The UI for the
resume/JD relevance scorer: a live scorer, a base-vs-fine-tuned playground, the
evaluation dashboard, dataset stats, and request history.

Design system ported from the CompactLLM Figma (dark-first, violet primary,
Inter + JetBrains Mono). Tokens live in [`app/globals.css`](app/globals.css).

## Develop

```bash
npm install
npm run dev            # http://localhost:3000
```

Point it at the backend with `NEXT_PUBLIC_API_URL` (defaults to
`http://localhost:8000`):

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

The backend's CORS accepts any `localhost` port by default, so the dev server
connects even if Next picks a non-3000 port.

## Routes

| Route         | Backend it talks to        |
|---------------|----------------------------|
| `/`           | `/health`, `/datasets/stats`, `/eval/benchmark`, `/history` |
| `/score`      | `/score` (SSE stream)      |
| `/playground` | `/compare`                 |
| `/evaluation` | `/eval/benchmark`          |
| `/datasets`   | `/datasets/stats`          |
| `/history`    | `/history`                 |

## Build

```bash
npm run build && npm start
```

Deploys to Vercel as-is; set `NEXT_PUBLIC_API_URL` to the deployed backend URL.
