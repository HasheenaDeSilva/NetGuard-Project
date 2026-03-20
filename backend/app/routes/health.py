from fastapi import APIRouter

# Create router group for health-related endpoints
router = APIRouter(tags=["Health"])


@router.get("/health")
def health():
    # Simple API health check endpoint
    # Used by frontend to show API Online / Offline status
    return {
        "status": "ok",
        "service": "NetGuard API",
    }