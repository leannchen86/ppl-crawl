# Crawlers

Scripts for crawling data from Diffbot's Knowledge Graph API.

## Setup

1. Copy `.env.example` to `.env` in the project root:
   ```bash
   cp .env.example .env
   ```

2. Add your Diffbot API token to `.env`:
   ```
   DIFFBOT_TOKEN=your_actual_token_here
   ```

## Usage

### crawl.ts
Fetches person entities with images from Diffbot's Knowledge Graph.

```bash
bun run crawlers/crawl.ts
```

### crawl_img.ts
Downloads images for entities from the JSON file.

```bash
bun run crawlers/crawl_img.ts
```
