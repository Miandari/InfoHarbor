"""
FastAPI application for the Elderly Assistant Agent
"""
import logging
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from src.api.routes import router
from src.settings import settings


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/app.log') if not settings.debug else logging.StreamHandler()
    ]
)

logger = logging.getLogger("api.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Elderly Assistant Agent API")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Agent name: {settings.agent_name}")
    
    # Create data directory if it doesn't exist
    import os
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Elderly Assistant Agent API")


# Create FastAPI app
app = FastAPI(
    title="Elderly Assistant Agent API",
    description="A caring AI assistant designed specifically for elderly users",
    version="1.0.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else [settings.render_external_url] if settings.render_external_url else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    
    if settings.debug:
        import traceback
        traceback.print_exc()
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "An error occurred"
        }
    )


# Include routes
app.include_router(router)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "name": "Elderly Assistant Agent API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs" if settings.debug else "disabled",
        "elder_mode": settings.elder_mode
    }


if __name__ == "__main__":
    uvicorn.run(
        "src.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )