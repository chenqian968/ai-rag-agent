"""
全局网络配置模块。

解决的问题:
在部分网络环境(尤其校园网/家庭宽带)下,DNS 会把 IPv6 地址排在 IPv4 前面,
但本机 IPv6 往往不通,导致 httpx 连接时报 `getaddrinfo failed` / `ConnectError`。

本模块在导入时强制 socket 优先使用 IPv4,一劳永逸地规避该问题。
"""

import socket


def _prefer_ipv4() -> None:
    """让 socket.getaddrinfo 优先返回 IPv4 地址。"""
    try:
        # AF_UNSPEC 表示不限定地址族;改为优先 IPv4 的解析策略
        if socket.has_ipv6:
            # 关键:把解析结果里的 IPv6 条目过滤,只保留 IPv4
            # 通过 monkeypatch getaddrinfo,在返回前按 IPv4 优先排序
            _orig_getaddrinfo = socket.getaddrinfo

            def _patched_getaddrinfo(*args, **kwargs):
                results = _orig_getaddrinfo(*args, **kwargs)
                # family=23 (AF_INET6) 排到后面,IPv4 (AF_INET, family=2) 优先
                results = sorted(
                    results,
                    key=lambda r: 0 if r[0] == socket.AF_INET else 1,
                )
                return results

            socket.getaddrinfo = _patched_getaddrinfo
    except Exception:  # noqa: BLE001
        # 打补丁失败也不影响主流程,只是退回默认行为
        pass


_prefer_ipv4()
