"""
Minimal Anthropic Messages API proxy using LiteLLM SDK.

Accepts Anthropic Messages API requests on /v1/messages and routes them
through LiteLLM to SAP AI Core (or any other LiteLLM-supported provider).

Uses litellm.anthropic.messages.acreate() which handles Anthropic format
natively — no format translation, no Pydantic serialization bugs.
"""

import json
import logging
import os
import sys

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

import litellm

# Drop unsupported params (e.g., 'thinking') instead of raising errors.
# Claude Code CLI sends params that not all providers support.
litellm.drop_params = True

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("litellm-proxy")

app = FastAPI()

# The LiteLLM model to route requests to (e.g., "sap/anthropic--claude-4.6-sonnet")
LITELLM_MODEL = os.environ.get("LITELLM_MODEL", "sap/anthropic--claude-4.6-sonnet")


@app.get("/health/readiness")
async def health_readiness():
    return {"status": "ok"}


@app.post("/v1/messages")
async def messages(request: Request):
    """
    Proxy Anthropic Messages API requests through LiteLLM SDK.

    Claude Code CLI sends requests here (via ANTHROPIC_BASE_URL).
    We forward them to LiteLLM which routes to SAP AI Core.
    """
    body = await request.json()

    # Override model with our LiteLLM model (e.g., sap/anthropic--claude-4.6-sonnet)
    # Claude Code CLI sends the model it was configured with, but we need
    # the LiteLLM provider-prefixed model name for routing.
    original_model = body.get("model", "unknown")
    body["model"] = LITELLM_MODEL

    logger.info(
        f"Proxying request: {original_model} -> {LITELLM_MODEL}, stream={body.get('stream', False)}"
    )

    is_streaming = body.get("stream", False)

    try:
        if is_streaming:
            return await _handle_streaming(body)
        else:
            return await _handle_non_streaming(body)
    except Exception as e:
        logger.exception(f"Error proxying request: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": str(e),
                },
            },
        )


async def _handle_non_streaming(body: dict) -> JSONResponse:
    """Handle non-streaming Anthropic Messages API request."""
    body.pop("stream", None)

    response = await litellm.anthropic.messages.acreate(**body)

    # litellm.anthropic.messages.acreate() returns Anthropic-format response
    if hasattr(response, "model_dump"):
        result = response.model_dump()
    elif isinstance(response, dict):
        result = response
    else:
        result = json.loads(str(response))

    return JSONResponse(content=result)


async def _handle_streaming(body: dict) -> StreamingResponse:
    """Handle streaming Anthropic Messages API request."""
    body["stream"] = True

    response = await litellm.anthropic.messages.acreate(**body)

    async def event_generator():
        """Yield SSE events from LiteLLM streaming response."""
        try:
            async for chunk in response:
                # Each chunk from litellm.anthropic.messages.acreate(stream=True)
                # is already in Anthropic SSE format
                if hasattr(chunk, "model_dump"):
                    data = chunk.model_dump()
                elif isinstance(chunk, dict):
                    data = chunk
                else:
                    data = json.loads(str(chunk))

                event_type = data.get("type", "content_block_delta")
                yield f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
        except Exception as e:
            logger.exception(f"Streaming error: {e}")
            error_data = {
                "type": "error",
                "error": {"type": "api_error", "message": str(e)},
            }
            yield f"event: error\ndata: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("LITELLM_PROXY_PORT", "4000"))
    logger.info(f"Starting LiteLLM proxy on port {port}, model={LITELLM_MODEL}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
