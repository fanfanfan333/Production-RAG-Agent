"""ForgeFlow adapter — the code specialist of the central coordinator.

Authentication is per request: the caller's own ForgeFlow bearer token is
forwarded verbatim. There is deliberately no admin token, default credential or
username→account mapping in this module. If the caller has not supplied a
token, code execution is reported unavailable instead of borrowing an identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ForgeFlowUnavailable(RuntimeError):
    """ForgeFlow is unconfigured, unreachable, or rejected the user credential."""


class ForgeFlowRejected(RuntimeError):
    """ForgeFlow answered with a non-success status."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"ForgeFlow HTTP {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class ForgeRun:
    run_id: str
    status: str
    outcome: str
    intent: str
    session_id: str
    parent_run_id: str
    artifacts: list[dict]
    codeplane: dict
    errors: list[str]


_TERMINAL = {"completed", "failed", "aborted", "rejected", "interrupted"}
_AWAITING = {"awaiting_approval", "pending_approval"}


class ForgeFlowClient:
    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        timeout_seconds: float = 30.0,
        transport: Any | None = None,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self._token = (token or "").strip()
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.transport = transport

    def _require_ready(self) -> None:
        if not self.base_url:
            raise ForgeFlowUnavailable("未配置 ForgeFlow 服务地址，代码执行不可用")
        if not self._token:
            raise ForgeFlowUnavailable(
                "缺少用户自己的 ForgeFlow 凭证，代码执行不可用（不会借用共享身份）"
            )

    def _client(self):
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise ForgeFlowUnavailable("HTTP 客户端未安装，代码执行不可用") from exc
        return httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            trust_env=False,
            transport=self.transport,
            headers={"Authorization": f"Bearer {self._token}"},
        )

    async def _request(self, method: str, path: str, **kwargs) -> Any:
        self._require_ready()
        try:
            async with self._client() as client:
                response = await client.request(method, path, **kwargs)
        except Exception as exc:  # noqa: BLE001 - typed, credential-free failure
            raise ForgeFlowUnavailable(
                f"ForgeFlow 连接失败: {type(exc).__name__}"
            ) from exc
        if response.status_code in {401, 403}:
            raise ForgeFlowUnavailable("ForgeFlow 凭证无效或已过期")
        if response.status_code >= 400:
            try:
                detail = str(response.json().get("detail") or "")[:300]
            except Exception:  # noqa: BLE001
                detail = response.text[:300]
            raise ForgeFlowRejected(response.status_code, detail)
        try:
            return response.json()
        except ValueError as exc:
            raise ForgeFlowUnavailable("ForgeFlow 返回了无法解析的响应") from exc

    async def create_task(
        self,
        *,
        intent: str,
        workflow_type: str = "generic",
        context: dict | None = None,
        session_id: str = "",
        parent_run_id: str = "",
        request_key: str = "",
    ) -> ForgeRun:
        payload = {
            "intent": intent,
            "workflow_type": workflow_type,
            "context": context or {},
        }
        if session_id:
            payload["session_id"] = session_id
        if parent_run_id:
            payload["parent_run_id"] = parent_run_id
        if request_key:
            payload["request_key"] = request_key
        data = await self._request("POST", "/workspace/tasks", json=payload)
        if not isinstance(data, dict) or not data.get("run_id"):
            raise ForgeFlowUnavailable("ForgeFlow 未返回执行编号")
        return ForgeRun(
            run_id=str(data["run_id"]),
            status=str(data.get("status") or "queued"),
            outcome=str(data.get("outcome") or ""),
            intent=intent,
            session_id=str(data.get("session_id") or ""),
            parent_run_id=str(data.get("parent_run_id") or ""),
            artifacts=[],
            codeplane={},
            errors=[],
        )

    async def get_request(self, request_key: str) -> ForgeRun:
        data = await self._request("GET", f"/workspace/requests/{request_key}")
        if not isinstance(data, dict) or not data.get("run_id"):
            raise ForgeFlowUnavailable("ForgeFlow 未返回执行编号")
        return await self.get_run(str(data["run_id"]))

    async def identity(self) -> dict:
        data = await self._request("GET", "/workspace/identity")
        if not isinstance(data, dict) or not data.get("tenant_id") or not data.get("user_id"):
            raise ForgeFlowUnavailable("ForgeFlow 未返回有效用户身份")
        return {"tenant_id": str(data["tenant_id"]), "user_id": str(data["user_id"])}

    async def get_run(self, run_id: str) -> ForgeRun:
        data = await self._request("GET", f"/runs/{run_id}")
        if not isinstance(data, dict):
            raise ForgeFlowUnavailable("ForgeFlow 运行详情格式无效")
        return ForgeRun(
            run_id=str(data.get("run_id") or run_id),
            status=str(data.get("status") or "unknown"),
            outcome=str(data.get("outcome") or ""),
            intent=str(data.get("intent") or ""),
            session_id=str(data.get("session_id") or ""),
            parent_run_id=str(data.get("parent_run_id") or ""),
            artifacts=list(data.get("artifacts") or []),
            codeplane=dict(data.get("codeplane") or {}),
            errors=[str(item) for item in (data.get("errors") or [])],
        )

    async def abort(self, run_id: str) -> str:
        data = await self._request("POST", f"/runs/{run_id}/abort")
        return str((data or {}).get("status") or "abort_requested")

    async def approve(self, run_id: str, note: str = "") -> dict:
        data = await self._request(
            "POST", f"/codeplane/runs/{run_id}/approve", json={"note": note}
        )
        return dict(data or {})

    async def reject(self, run_id: str, note: str = "") -> dict:
        data = await self._request(
            "POST", f"/codeplane/runs/{run_id}/reject", json={"note": note}
        )
        return dict(data or {})


def forgeflow_state(run: ForgeRun) -> str:
    """Map ForgeFlow run status onto the coordinator's honest task state."""
    status = run.status.lower()
    if status == "completed":
        return "completed"
    if status in _AWAITING:
        return "awaiting_approval"
    if status in {"failed", "rejected", "aborted", "interrupted"}:
        return status
    if status in {"queued", "pending"}:
        return "running"
    if status == "running":
        return "running"
    return "running" if status not in _TERMINAL else "failed"
