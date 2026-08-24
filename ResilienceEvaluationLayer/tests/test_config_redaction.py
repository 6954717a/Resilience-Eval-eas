from omegaconf import OmegaConf

from habitat_llm.config_redaction import (
    REDACTED_VALUE,
    redacted_config_yaml,
    write_redacted_config_copy,
)


def test_redacted_config_yaml_is_recursive_and_does_not_mutate_config(
    monkeypatch,
):
    monkeypatch.setenv("TEST_CONFIG_SECRET", "resolved-secret-value")
    config = OmegaConf.create(
        {
            "llm_api_key": "plain-api-key",
            "nested": [
                {
                    "accessToken": "plain-token",
                    "headers": {"Authorization": "Bearer private"},
                    "sort_key": "non-secret-sort-order",
                }
            ],
            "clientSecret": "${oc.env:TEST_CONFIG_SECRET}",
            "llm_base_url": "https://example.invalid/v1",
        }
    )

    rendered = redacted_config_yaml(config, resolve=True)
    persisted = OmegaConf.create(rendered)

    assert persisted.llm_api_key == REDACTED_VALUE
    assert persisted.nested[0].accessToken == REDACTED_VALUE
    assert persisted.nested[0].headers.Authorization == REDACTED_VALUE
    assert persisted.clientSecret == REDACTED_VALUE
    assert persisted.nested[0].sort_key == "non-secret-sort-order"
    assert persisted.llm_base_url == "https://example.invalid/v1"
    assert "plain-api-key" not in rendered
    assert "plain-token" not in rendered
    assert "resolved-secret-value" not in rendered

    assert config.llm_api_key == "plain-api-key"
    assert config.nested[0].accessToken == "plain-token"
    assert config.clientSecret == "resolved-secret-value"


def test_unresolved_secret_interpolation_is_replaced_without_resolution(
    monkeypatch,
):
    monkeypatch.setenv("TEST_CONFIG_SECRET", "must-not-be-persisted")
    config = OmegaConf.create(
        {
            "openai_api_key": "${oc.env:TEST_CONFIG_SECRET}",
            "ordinary_key_name": "kept",
        }
    )

    rendered = redacted_config_yaml(config)
    persisted = OmegaConf.create(rendered)

    assert persisted.openai_api_key == REDACTED_VALUE
    assert persisted.ordinary_key_name == "kept"
    assert "TEST_CONFIG_SECRET" not in rendered
    assert "must-not-be-persisted" not in rendered


def test_runtime_yaml_uses_null_and_never_resolves_sensitive_field():
    resolver_called = False

    def sensitive_resolver():
        nonlocal resolver_called
        resolver_called = True
        raise RuntimeError("sensitive resolver must not run")

    OmegaConf.register_new_resolver(
        "test_sensitive_config_resolver",
        sensitive_resolver,
        replace=True,
    )
    config = OmegaConf.create(
        {
            "llm_api_key": "${test_sensitive_config_resolver:}",
            "ordinary_value": "kept",
        }
    )

    rendered = redacted_config_yaml(
        config,
        resolve=True,
        replacement=None,
    )
    persisted = OmegaConf.create(rendered)

    assert persisted.llm_api_key is None
    assert persisted.ordinary_value == "kept"
    original = OmegaConf.to_container(config, resolve=False)
    assert original["llm_api_key"] == "${test_sensitive_config_resolver:}"
    assert not resolver_called
    assert "test_sensitive_config_resolver" not in rendered
    assert "sensitive resolver must not run" not in rendered


def test_write_redacted_config_copy_never_copies_external_secret(tmp_path):
    source = tmp_path / "server" / "config.yaml"
    destination = tmp_path / "results" / "config_rlm.yaml"
    source.parent.mkdir()
    destination.parent.mkdir()
    source.write_text(
        "llm_api_key: external-secret\n"
        "nested:\n"
        "  authorization: Bearer external-token\n"
        "  model: public-model-name\n",
        encoding="utf-8",
    )

    written = write_redacted_config_copy(source, destination)
    persisted = OmegaConf.load(destination)

    assert written == destination
    assert persisted.llm_api_key == REDACTED_VALUE
    assert persisted.nested.authorization == REDACTED_VALUE
    assert persisted.nested.model == "public-model-name"
    assert "external-secret" not in destination.read_text(encoding="utf-8")
    assert "external-token" not in destination.read_text(encoding="utf-8")
    assert "external-secret" in source.read_text(encoding="utf-8")
