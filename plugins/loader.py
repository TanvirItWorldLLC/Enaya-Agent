from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class PluginManifest:
    name: str
    version: str
    description: str
    entry_point: str
    dependencies: list[str] = field(default_factory=list)
    hooks: list[str] = field(default_factory=list)
    config_schema: dict[str, Any] = field(default_factory=dict)


class PluginLoader:
    def __init__(self, plugin_paths: list[str] | None = None) -> None:
        self.plugin_paths = plugin_paths or ["./plugins"]
        self._plugins: dict[str, Any] = {}
        self._manifests: dict[str, PluginManifest] = {}
        logger.info("plugin_loader_initialized", paths=self.plugin_paths)

    async def discover(self) -> list[PluginManifest]:
        manifests = []
        for path in self.plugin_paths:
            p = Path(path)
            if not p.exists():
                continue
            for plugin_dir in p.iterdir():
                if plugin_dir.is_dir():
                    manifest_path = plugin_dir / "plugin.json"
                    if manifest_path.exists():
                        with open(manifest_path) as f:
                            data = json.load(f)
                        manifest = PluginManifest(**data)
                        manifests.append(manifest)
                        self._manifests[manifest.name] = manifest
        logger.info("plugins_discovered", count=len(manifests))
        return manifests

    async def load(self, manifest: PluginManifest) -> Any:
        for path in self.plugin_paths:
            p = Path(path) / manifest.name / manifest.entry_point
            if p.exists():
                spec = importlib.util.spec_from_file_location(manifest.name, p)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    plugin_class = getattr(module, "EnayaPlugin", None)
                    if not plugin_class:
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            if (
                                isinstance(attr, type)
                                and hasattr(attr, "name")
                                and hasattr(attr, "version")
                                and hasattr(attr, "initialize")
                            ):
                                plugin_class = attr
                                break
                    if plugin_class:
                        plugin = plugin_class()
                        self._plugins[manifest.name] = plugin
                        logger.info("plugin_loaded", name=manifest.name, version=manifest.version)
                        return plugin
        raise RuntimeError(f"Could not load plugin {manifest.name}")

    async def load_all(self) -> list[Any]:
        manifests = await self.discover()
        plugins = []
        for m in manifests:
            try:
                p = await self.load(m)
                plugins.append(p)
            except Exception as e:
                logger.error("plugin_load_failed", name=m.name, error=str(e))
        return plugins

    async def initialize_all(self, context: dict[str, Any]) -> None:
        for name, plugin in self._plugins.items():
            try:
                await plugin.initialize(context)
                logger.info("plugin_initialized", name=name)
            except Exception as e:
                logger.error("plugin_init_failed", name=name, error=str(e))

    async def shutdown_all(self) -> None:
        for name, plugin in self._plugins.items():
            try:
                await plugin.shutdown()
                logger.info("plugin_shutdown", name=name)
            except Exception as e:
                logger.error("plugin_shutdown_failed", name=name, error=str(e))

    def get(self, name: str) -> Any | None:
        return self._plugins.get(name)

    def list_plugins(self) -> list[str]:
        return list(self._plugins.keys())
