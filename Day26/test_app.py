import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app import app

client = TestClient(app)

def test_hello_route():
    """Test the successful GET / route."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello World"}

def test_error_route():
    """Test the GET /error route that raises HTTPException."""
    response = client.get("/error")
    assert response.status_code == 400
    assert "detail" in response.json()
    assert response.json()["detail"] == "This is a test bad request"

def test_nonexistent_route():
    """Test a non-existent route to cover 404 logging."""
    response = client.get("/nonexistent")
    assert response.status_code == 404

def test_middleware_exception_handling():
    import asyncio
    from starlette.requests import Request
    async def run_test():
        scope = {
            'type': 'http',
            'method': 'GET',
            'path': '/',
            'query_string': b'',
            'headers': [],
            'client': ('127.0.0.1', 8000),
        }
        request = Request(scope)
        async def mock_call_next(request):
            raise Exception("Test exception")
        from app import log_requests
        with pytest.raises(Exception, match="Test exception"):
            await log_requests(request, mock_call_next)
    asyncio.run(run_test())


