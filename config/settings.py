from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class MemoryConfig(BaseModel):
    backend: str = "chromadb"
    path: str = "./data/memory"
    collection: str = "enaya_memory"
    embedding_model: str = "text-embedding-3-small"
    max_history: int = 50


class GatewayConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8765
    enable_websocket: bool = True
    enable_http: bool = True
    enable_mcp: bool = True
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


class BackendConfig(BaseModel):
    default: str = "local"
    available: list[str] = Field(
        default_factory=lambda: ["local", "docker", "ssh", "subprocess", "kubernetes", "cloud"]
    )


class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: str | None = Field(default=None, env="OPENAI_API_KEY")
    base_url: str | None = Field(default=None, env="OPENAI_BASE_URL")
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 60


class AgentConfig(BaseModel):
    model: str = "gpt-4o"
    max_iterations: int = 25
    timeout: int = 300
    temperature: float = 0.7


class PluginConfig(BaseModel):
    auto_discover: bool = True
    paths: list[str] = Field(default_factory=list)
    registry_url: str | None = None


class EnayaConfig(BaseSettings):
    project_name: str = "Enaya Agent"
    debug: bool = False
    log_level: str = "INFO"
    data_dir: Path = Path("./data")
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    backend: BackendConfig = Field(default_factory=BackendConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    plugins: PluginConfig = Field(default_factory=PluginConfig)

    class Config:
        env_prefix = "ENAYA_"
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_config() -> EnayaConfig:
    return EnayaConfig()
