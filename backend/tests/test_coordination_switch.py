def test_disabled_coordinator_does_not_register_task_routes(monkeypatch):
    from app import main

    monkeypatch.setattr(main._settings, "COORDINATION_ENABLED", False)
    app = main.create_app()
    assert not any(path.startswith("/agent/tasks") for path in app.openapi()["paths"])


def test_forgeflow_address_is_not_assumed_to_be_the_rag_service():
    from app.config import Settings

    assert Settings.model_fields["FORGEFLOW_BASE_URL"].default == ""
