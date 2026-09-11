from backend.shared.health import HealthResponse, build_health_response


def test_build_health_response_defaults_to_ok() -> None:
    response = build_health_response("test-service")

    assert isinstance(response, HealthResponse)
    assert response.status == "ok"
    assert response.service == "test-service"
    assert response.environment == "development"


def test_build_health_response_uses_environment() -> None:
    response = build_health_response("test-service", "testing")

    assert response.environment == "testing"
