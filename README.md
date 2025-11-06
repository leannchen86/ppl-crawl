For Moo

- Install Bun: `curl -fsSL https://bun.com/install | bash`
- Get JSON data from Diffbot: `bun crawl.ts` (don't share that script; your API key is in it). This'll save into `entities.json`.
- Get images data from entities.json: `bun crawl_img.ts` (change the range of entities pictures to crawl in that file, if needed). This'll save into `images/`.
