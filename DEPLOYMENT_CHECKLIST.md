# Netlify Deployment Checklist

## Pre-Deployment Steps

### Code Changes ✅
- [x] Removed hardcoded `https://bunoraa.com` from `lib/seo.ts`
- [x] Removed hardcoded `https://bunoraa.com` from `components/contact/ContactPageClient.tsx`
- [x] Removed hardcoded `media.bunoraa.com` fallback pattern from `next.config.ts`
- [x] Created `netlify.toml` configuration file
- [x] Created `.env.example` documentation file
- [x] All hardcoded production URLs replaced with safe fallbacks

### Configuration Files
- [ ] Review and commit all changes to Git
- [ ] Push changes to your main branch

## Netlify Setup Steps

### 1. Create/Link Netlify Project
```
Go to https://app.netlify.com
Click "New site from Git"
Select your repository
Connect your GitHub/GitLab/Bitbucket account
Choose the branch to deploy (main/master)
```

### 2. Configure Build Settings in Netlify UI

Set these values:
- **Base directory**: `frontend`
- **Build command**: `npm run build`
- **Publish directory**: `.next`

### 3. Add Environment Variables in Netlify

Go to: **Site Settings → Environment variables**

Add **only these required variables** (update URLs according to your HuggingFace backend):

```
NEXT_PUBLIC_API_BASE_URL = https://your-backend-space.hf.space/api/v1
NEXT_INTERNAL_API_BASE_URL = https://your-backend-space.hf.space/api/v1
NEXT_PUBLIC_MEDIA_BASE_URL = https://your-backend-space.hf.space/media
NEXT_PUBLIC_SITE_URL = https://bunoraa.netlify.app
NEXT_PUBLIC_WS_BASE_URL = wss://your-backend-space.hf.space
NODE_VERSION = 20
```

**Optional** (only if using these services):

```
NEXT_PUBLIC_CLOUDFLARE_BEACON_TOKEN = 99cd4569fd314a31bb530d46e16f26c9
NEXT_PUBLIC_VAPID_PUBLIC_KEY = BFbhEXdK2Jp5YZijD8PwFvRrtgK87GbOQps12XCGyXhkE_4r-VTXdrH8VkxjV7gnrzQ6kYyezfJ-sOa5OIj-_Gc
```

**IMPORTANT**: Set these BEFORE deploying to avoid build failures

### 4. Trigger First Deployment

```
Click "Deploy"
Wait for build to complete
Check deploy logs if build fails
```

## Backend (HuggingFace) Setup

### Django CORS Configuration

Update your Django backend `settings.py`:

```python
# Allow Netlify frontend
CORS_ALLOWED_ORIGINS = [
    "https://bunoraa.netlify.app",
    "http://localhost:3000",  # For local development
]

# Also set NEXT_FRONTEND_ORIGIN in Django settings
NEXT_FRONTEND_ORIGIN = "https://bunoraa.netlify.app"
```

### HuggingFace Space Configuration

1. Create a HuggingFace Space (Docker/Docker Compose)
2. Deploy your Django backend there
3. Note the Space URL: `https://username-space.hf.space`
4. Update Netlify environment variables with this URL

## Verification Steps

### After First Successful Deploy

- [ ] Visit https://bunoraa.netlify.app
- [ ] Check Console for errors (F12)
- [ ] Test API calls work (Network tab)
- [ ] Check if images load correctly
- [ ] Test a form submission (Contact page)
- [ ] Check WebSocket connection (if applicable)

### If Build Fails: Common Issues

1. **"Secrets Scanning Found Secrets"**
   - Verify all environment variables are set in Netlify
   - Clear build cache: Site settings → Build & deploy → Clear cache
   - Retry deploy

2. **"Cannot Find Module"**
   - Ensure `NODE_VERSION=20` is set
   - Check `frontend/package.json` exists
   - Clear cache and retry

3. **"API requests failing"**
   - Verify `NEXT_INTERNAL_API_BASE_URL` points to correct backend
   - Check backend CORS configuration
   - Test backend URL directly in browser

## Continuous Deployment

After first successful deploy:

1. **Make code changes locally**
2. **Push to main branch**: `git push origin main`
3. **Netlify automatically builds and deploys**
4. **Check Netlify Dashboard → Deploys**

## Local Development

To test locally before deploying:

```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:3000`

Ensure Django backend is running on `http://localhost:8000`

## Rollback Instructions

If deployment breaks production:

1. Go to Netlify Dashboard
2. Click "Deploys"
3. Find the last successful deployment
4. Click the three dots (⋯) menu
5. Select "Publish deploy"

Reverted! The previous version is now live.

## Performance Optimization (Optional)

- Enable Netlify Analytics: Site settings → Analytics
- Monitor Core Web Vitals
- Consider adding image optimization service

## Security Checklist

- [x] No hardcoded secrets in source code
- [ ] `.env.production` not committed to Git
- [ ] Environment variables set in Netlify only
- [ ] CORS properly configured on backend
- [ ] HTTPS enabled (automatic with Netlify)
- [ ] Security headers configured in `netlify.toml`

## Support & Documentation

- Netlify Deployment Issues: See [NETLIFY_DEPLOYMENT.md](../NETLIFY_DEPLOYMENT.md)
- Next.js Docs: https://nextjs.org/docs
- Netlify Docs: https://docs.netlify.com/

## Quick Links

- Netlify Dashboard: https://app.netlify.com
- HuggingFace Spaces: https://huggingface.co/spaces
- Project Repository: [Your repo URL]
