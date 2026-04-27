# Netlify Deployment Guide for Bunoraa Frontend

## Overview

This guide explains how to deploy the Bunoraa frontend application to Netlify, connecting it with the HuggingFace-hosted backend API.

## Prerequisites

1. Netlify account with access to deploy
2. Git repository connected to Netlify
3. Backend API deployed on HuggingFace or another server
4. Required environment variables documented

## Environment Variables

### For Netlify Production Deployment

Set these environment variables in your Netlify Site Settings → Environment:

```bash
NEXT_PUBLIC_API_BASE_URL=https://bunoraa-api.hf.space/api/v1
NEXT_INTERNAL_API_BASE_URL=https://bunoraa-api.hf.space/api/v1
NEXT_PUBLIC_MEDIA_BASE_URL=https://bunoraa-api.hf.space/media
NEXT_PUBLIC_SITE_URL=https://bunoraa.netlify.app
NEXT_PUBLIC_WS_BASE_URL=wss://bunoraa-api.hf.space
NEXT_PUBLIC_WS_ENABLED=true
NEXT_DISABLE_PRERENDER=true
NEXT_DISABLE_BUILD_PRERENDER=true
NEXT_IMAGE_UNOPTIMIZED=false
NEXT_PUBLIC_CLOUDFLARE_BEACON_TOKEN=99cd4569fd314a31bb530d46e16f26c9
NEXT_PUBLIC_VAPID_PUBLIC_KEY=BFbhEXdK2Jp5YZijD8PwFvRrtgK87GbOQps12XCGyXhkE_4r-VTXdrH8VkxjV7gnrzQ6kYyezfJ-sOa5OIj-_Gc
NEXT_FRONTEND_ORIGIN=https://bunoraa.netlify.app
NEXT_API_PROXY_TARGET=https://bunoraa-api.hf.space
NODE_VERSION=20
```

### Important Notes on Environment Variables

- **NEXT_PUBLIC_*** variables: Exposed to browser, safe to include production URLs
- **NEXT_INTERNAL_API_BASE_URL**: Used during build, should point to backend API
- **NEXT_FRONTEND_ORIGIN**: Used for CORS and redirects
- Never hardcode sensitive keys in source code (they're now removed in favor of environment variables)

## Deployment Steps

### 1. Connect Repository to Netlify

- Go to Netlify.com → New site from Git
- Select your GitHub/GitLab/Bitbucket repository
- Choose the branch to deploy (e.g., `main`)

### 2. Configure Build Settings

The `netlify.toml` file handles most configuration, but verify:

- **Base directory**: `frontend`
- **Build command**: `npm run build`
- **Publish directory**: `.next`

### 3. Set Environment Variables

1. Go to Site Settings → Environment
2. Add all the variables listed in the "For Netlify Production Deployment" section above
3. **CRITICAL**: These variables must be set BEFORE the first build to prevent secrets detection

### 4. Deploy

- Commit your changes to the main branch
- Netlify will automatically trigger a build
- Monitor the build logs in the Netlify UI

## Troubleshooting

### Build Fails with "Secrets Scanning Found Secrets"

This occurs when hardcoded production URLs are detected in the code:

- **Solution**: Ensure all production URLs are set as environment variables in Netlify
- The code now uses dynamic environment variables with safe fallbacks
- Remove any `.env.production` file from Git if it contains real credentials

### Build Fails with "Cannot Find Module"

- Check that `NODE_VERSION=20` is set in environment variables
- Clear the build cache in Netlify: Site settings → Build & deploy → Clear cache

### CORS Errors in Production

- Verify `NEXT_FRONTEND_ORIGIN` matches your deployed domain
- Ensure Django backend has this domain in `ALLOWED_ORIGINS`
- Check that backend's `NEXT_FRONTEND_ORIGIN` setting includes the Netlify domain

### WebSocket Errors

- Verify `NEXT_PUBLIC_WS_BASE_URL` points to your backend
- For HuggingFace Spaces: Use `wss://` protocol
- Ensure WebSocket support is enabled on your backend

### Media Not Loading

- Check `NEXT_PUBLIC_MEDIA_BASE_URL` is correct
- Verify CORS headers are set correctly on the media server
- Test media URL directly in browser

## Performance Optimization

### Image Optimization

The frontend uses Next.js Image Optimization:

- Local development: `NEXT_IMAGE_UNOPTIMIZED=true` (faster builds)
- Production: `NEXT_IMAGE_UNOPTIMIZED=false` (optimized delivery)

### Caching Headers

`netlify.toml` includes cache headers for:

- `/_next/*` files (31536000s = 1 year)
- `/static/*` files (31536000s = 1 year)
- Other assets (browser default)

## Local Development

### Setup

```bash
cd frontend
npm install
cp .env.example .env.local  # If not already done
npm run dev
```

### Environment for Local Backend

Set these in `.env.local` when running Django backend on localhost:8000:

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/api/v1
NEXT_INTERNAL_API_BASE_URL=http://localhost:8000/api/v1
NEXT_PUBLIC_MEDIA_BASE_URL=http://localhost:8000/media
NEXT_FRONTEND_ORIGIN=http://localhost:3000
NEXT_API_PROXY_TARGET=http://localhost:8000
```

## Backend Integration

### HuggingFace Deployment

If backend is on HuggingFace Spaces:

- URL format: `https://username-space-name.hf.space`
- Use this URL for all `NEXT_INTERNAL_API_BASE_URL`, `NEXT_PUBLIC_API_BASE_URL`, etc.
- Ensure HuggingFace Space allows CORS requests

### CORS Configuration

The frontend needs backend to accept requests from:

- `https://bunoraa.netlify.app` (Netlify)
- Your custom domain if applicable
- `http://localhost:3000` (development)

Update Django `ALLOWED_ORIGINS` setting accordingly.

## Monitoring & Maintenance

### Check Build Logs

- Netlify Dashboard → Deploys → Click on a deploy → Deploy log
- Look for:
  - Build errors
  - Dependencies not installing
  - Environment variable warnings

### Performance Monitoring

- Netlify Analytics (if enabled)
- Lighthouse scores in production
- Core Web Vitals

## Rollback Procedures

If a deployment fails:

1. Go to Netlify Dashboard → Deploys
2. Find the last successful deployment
3. Click the three dots → Publish deploy
4. The previous version becomes live immediately

## CI/CD Integration

Netlify automatically:

- Triggers on Git push to deployed branch
- Runs the build command
- Publishes if successful
- Can integrate with pull request previews

## Custom Domain Setup

1. Go to Site Settings → Domain management
2. Add your custom domain
3. Update DNS records (or use Netlify DNS)
4. Wait for DNS propagation
5. Update environment variable `NEXT_PUBLIC_SITE_URL` to match

## Security Considerations

- ✅ No hardcoded secrets in source code
- ✅ Environment variables handled by Netlify
- ✅ SSL/TLS automatically configured
- ✅ Headers configured in `netlify.toml`
- ⚠️ Ensure `NEXT_PUBLIC_*` variables don't contain sensitive data
- ⚠️ Keep private keys (like API keys) in Netlify environment only

## Need Help?

- Check Netlify docs: https://docs.netlify.com/
- Next.js deployment guide: https://nextjs.org/docs/deployment/netlify
- See DEPLOYMENT.md in the repository root for backend deployment
