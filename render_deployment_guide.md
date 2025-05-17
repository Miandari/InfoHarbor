# Dastyar Assistant API - Render Deployment Guide

This guide provides instructions for deploying the Dastyar Assistant API on Render with performance optimizations for low-latency access.

## Deployment Overview

The Dastyar API is configured to run on Render as a Web Service with a linked Redis database for caching and state management. This architecture provides the best balance of performance, reliability, and cost.

## Prerequisites

- A Render account (https://render.com)
- Your API keys (OpenAI, Tavily, ListenNotes)
- Git repository with your Dastyar code

## Deployment Steps

### 1. Deploy using Render Blueprint

The simplest way to deploy is using Render's Blueprint feature:

1. Fork or push your Dastyar repository to GitHub
2. In Render, go to "Blueprints" and click "New Blueprint Instance"
3. Connect your GitHub repository
4. Render will detect the `render.yaml` file and set up all services automatically

### 2. Manual Service Setup

If you prefer manual setup:

1. **Create Redis Service:**
   - In Render dashboard, select "New" → "Redis"
   - Name it "dastyar-redis"
   - Choose an appropriate plan (at least 256MB)
   - Create the service

2. **Create Web Service:**
   - In Render dashboard, select "New" → "Web Service"
   - Connect your repository
   - Name it "dastyar-api"
   - Set Build Command: `pip install -r requirements.txt`
   - Set Start Command: `uvicorn api:app --host 0.0.0.0 --port $PORT`
   - Under "Advanced" → "Environment", add all required environment variables
   - Link the Redis service via environment variable

## Performance Optimization

### 1. Render-Specific Configuration

The following settings are optimized for Render hosting:

#### Service Configuration:

- **Plan Selection:** Choose at least "Standard" plan for production use
- **Region:** Select the region closest to your API consumers
- **Auto-scaling:** Enable for handling variable loads
- **Health Check Path:** Set to `/health` for monitoring

#### Environment Variables:

```
# API Keys
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
LISTENNOTES_API_KEY=your_listennotes_key

# Render Configuration
PORT=10000  # Render will override this
RENDER_REDIS_URL=[will be auto-set when linked]

# API Security
API_KEYS=your_api_key1,your_api_key2
ALLOWED_ORIGINS=https://your-frontend-domain.com,https://www.your-frontend-domain.com

# Performance Tuning
API_RATE_LIMIT=200
MAX_CONVERSATION_HISTORY=50
WORKERS=4  # CPU cores available on your Render instance
```

### 2. Render-Optimized Redis Usage

The API is configured to use Render's managed Redis service for:

- Session storage
- Response caching
- Rate limiting tracking
- Tool configuration storage

This minimizes database access and improves response time.

### 3. Scaling and Performance Monitoring

Enable Render's metrics and monitoring:

1. In your service dashboard, go to "Metrics"
2. Monitor response times, error rates, and resource usage
3. Set up alerts for performance degradation

### 4. Content Delivery Network (CDN)

For even faster global access, place a CDN in front of your Render deployment:

1. Configure your Render service to be the origin server
2. Set appropriate cache headers in the FastAPI responses
3. Configure your CDN to respect cache directives

## Deployment Verification

After deployment, test your API with:

```bash
curl https://your-render-service.onrender.com/health
```

You should receive a response indicating the service is healthy.

## Common Optimization Issues

### Slow Cold Starts

Render free and starter plans will spin down after periods of inactivity, causing cold starts:

1. **Solution:** Upgrade to a "Standard" plan for always-on service
2. **Workaround:** Set up a simple cron job to ping your service every 5 minutes

### Redis Connection Errors

If your service can't connect to Redis:

1. Ensure the Redis instance is running
2. Verify the environment variable is correctly linked
3. Check Redis connection limits (you may need to upgrade your Redis plan)

### High Latency

If you're experiencing high latency:

1. Monitor query times in the API logs
2. Check Redis connection pooling settings
3. Consider placing your Render and Redis instances in the same region
4. Enable request compression in both the API and client

## Real-time Monitoring Dashboard

For production deployments, create a monitoring dashboard:

1. In the Render dashboard, go to "Dashboards" → "Create"
2. Add metrics for:
   - Response time
   - Error rate
   - CPU/Memory usage
   - Redis connections

## Conclusion

With these optimizations, your Dastyar API hosted on Render will provide fast, reliable responses even under load. The combination of Redis caching, proper connection pooling, and smart client integration results in a high-performance system suitable for production use.