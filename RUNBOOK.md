# Runbook: VPN/Proxy IP Classifier API

How the API is deployed, how to redeploy it, how to roll back to an earlier image, and how to check that it's healthy.

## How deployment works

```
push to main ─► CI: lint ─┐
                  test ───┴─► build image ─► smoke-test it ─► push to GHCR ─► deploy job ─► Render pulls that tag
                                                             sha-<commit>, latest          (deploy hook + imgURL)
```

- **Image registry:** `ghcr.io/uzair-khiiba/vpn-classifier-mlops`. Every push to `main` produces `sha-<7-char commit>` and moves `latest`. A `v*` git tag also produces an image with that tag, such as `v2.0-milestone2`.
- **Host:** Render web service `vpn-proxy-api` (free plan, Singapore), created from [render.yaml](render.yaml). Render runs the image that CI built and tested; it never builds the code itself.
- **What's live:** `GET /health` returns `"build": "<commit>"`, the commit the running image was built from. Compare it with the tag you expect.
- **Health:** Render checks `/health` before switching traffic to a new deploy. If the new container never answers, Render keeps the previous one running. The image's Docker `HEALTHCHECK` also requires `"status": "ok"`, meaning the model loaded.

## One-time setup

1. **Make the image public.** After the first CI run on `main`: GitHub → your profile → Packages → `vpn-classifier-mlops` → Package settings → Change visibility → Public. Otherwise Render can't pull it. Alternatively, add a GHCR registry credential in Render using a personal access token with `read:packages`.
2. **Create the service.** In Render, choose New → Blueprint, pick this repo, and apply `render.yaml`. Or use New → Web Service → Existing image → `ghcr.io/uzair-khiiba/vpn-classifier-mlops:latest`, with health check path `/health`.
3. **Connect CI to Render.** Copy the deploy hook from Render (service → Settings → Deploy Hook). In GitHub, go to repo → Settings → Secrets and variables → Actions:
   - Secret `RENDER_DEPLOY_HOOK_URL` = the deploy hook URL.
   - Variable `SERVICE_URL` = `https://<service>.onrender.com`. Optional: with it, CI waits until `/health` reports the new build.

Without the secret, the deploy job prints a warning and skips the deploy; the rest of CI still runs.

### The dashboard service

The dashboard (`vpn-proxy-dashboard`, image `ghcr.io/uzair-khiiba/vpn-classifier-dashboard`, built from `ui.Dockerfile`) is set up the same way:

1. After CI's first run on `main` with the dashboard job, make the `vpn-classifier-dashboard` package public.
2. In Render, open the Blueprint and **Sync** it so it picks up the second service in `render.yaml`. Or create it with New → Web Service → Existing image → `ghcr.io/uzair-khiiba/vpn-classifier-dashboard:latest`, health check path `/_stcore/health`, and environment variable `API_BASE_URL=https://vpn-proxy-api.onrender.com`.
3. Add the dashboard's deploy hook to the `production` environment as the secret `RENDER_DASHBOARD_DEPLOY_HOOK_URL`.

Redeploying and rolling back work exactly as for the API: use the dashboard's hook and its `sha-<commit>` tags. The two services are independent, so rolling back one doesn't touch the other.

## Redeploy

| Situation | Do this |
|---|---|
| Normal release | Merge to `main`. CI tests, builds, pushes `sha-<commit>` and deploys it. |
| Re-run the same version (e.g. after a Render incident) | GitHub → Actions → the run on `main` → **Re-run all jobs**. Or Render → Manual Deploy → *Deploy latest reference*. |
| Deploy a specific tag by hand | `curl -X POST "$RENDER_DEPLOY_HOOK_URL&imgURL=ghcr.io%2Fuzair-khiiba%2Fvpn-classifier-mlops%3Asha-abc1234"` |
| New model | Promote it (`python -m experiments.promote_model`). This changes the committed `models/model.joblib`. Then merge to `main` as normal: the model is copied into the image. |

Then verify:

```bash
URL=https://<service>.onrender.com
curl -s $URL/health      # "status":"ok", "model_loaded":true, "build":"<commit you expect>"
curl -s -X POST $URL/predict -H "Content-Type: application/json" -d @examples/vpn_ip.json
```

## Roll back to a previous image tag

Use this when a deploy breaks the service (5xx errors, wrong predictions, `/health` degraded) and you need the last good version back now. Fix the code afterwards.

1. **Find the last good tag.** `/metrics` and the Render logs show when errors started. List the tags:
   ```bash
   gh api /users/uzair-khiiba/packages/container/vpn-classifier-mlops/versions \
     --jq '.[] | "\(.created_at)  \(.metadata.container.tags | join(", "))"' | head
   ```
   Or check GitHub → Packages → `vpn-classifier-mlops`, or the commit history of `main`. Each commit's image is `sha-<its 7-char hash>`.
2. **Deploy that tag.** Use one of these:
   - **Deploy hook (fastest):**
     ```bash
     TAG=sha-abc1234        # or v2.0-milestone2
     curl -X POST "$RENDER_DEPLOY_HOOK_URL&imgURL=$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=''))" ghcr.io/uzair-khiiba/vpn-classifier-mlops:$TAG)"
     ```
   - **Render dashboard:** service → Settings → Image URL → change the tag to `:sha-abc1234` → Save, then Manual Deploy. Or service → Events → pick an earlier successful deploy → **Rollback**.
3. **Verify:** `curl -s $URL/health` shows `"build":"abc1234"` and `"status":"ok"`.
4. **Stop the next push from re-deploying the bad code.** Revert the bad commit on `main` (`git revert <sha>`, through a PR). The CI run for the revert deploys a fixed image. Don't re-tag `latest` by hand; the next push to `main` moves it.

### Locally, or on a plain Docker host

```bash
docker pull ghcr.io/uzair-khiiba/vpn-classifier-mlops:sha-abc1234
docker rm -f vpn-proxy-api
docker run -d --name vpn-proxy-api -p 8000:8000 ghcr.io/uzair-khiiba/vpn-classifier-mlops:sha-abc1234
```

## Monitoring

| Where | What you see |
|---|---|
| `GET /metrics` | Since the process started: `total_requests`, `total_errors` (5xx), `avg_latency_ms`, predictions per class, and per endpoint the count, errors, average, p95 and max latency. Counters reset on restart or redeploy. |
| Render → Logs | One line per request: `request method=POST path=/predict status=200 latency_ms=4.21`, plus uvicorn's access log and any model-loading errors |
| Response header | `X-Process-Time-Ms` on every response |
| `GET /health` | Model loaded, lookup data loaded, model version, running build |

```bash
watch -n 5 "curl -s $URL/metrics | python3 -m json.tool | head -20"
```

**Things to check:**
- `total_errors` is above 0.
- `/health` is `degraded`: the model file is missing or invalid; the reason is in the logs.
- `avg_latency_ms` for `POST /predict` goes well above ~25 ms. That's the locally measured figure; Render's free CPU is slower.

## Known behaviour

- **Cold start.** Render's free instances sleep after 15 minutes without traffic. The first request after that takes about 30–60 s. Open `/health` a minute before a demo.
- **Metrics reset** when the instance sleeps or redeploys, because they live in memory.
- **First SHAP request** (`POST /explain/shap`) takes a few hundred ms longer while SHAP loads.
