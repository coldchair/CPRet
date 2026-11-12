from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import stat
import struct
import threading


MAX_MESSAGE = 4096


def receive_message(connection: socket.socket) -> dict:
    buffer = bytearray()
    while b"\n" not in buffer:
        chunk = connection.recv(min(1024, MAX_MESSAGE + 1 - len(buffer)))
        if not chunk:
            raise ValueError("Incomplete local command")
        buffer.extend(chunk)
        if len(buffer) > MAX_MESSAGE:
            raise ValueError("Local command too large")
    message = json.loads(bytes(buffer).split(b"\n", 1)[0])
    if not isinstance(message, dict):
        raise ValueError("Local command must be a JSON object")
    return message


def private_directory(directory: Path) -> None:
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError("Local admin directory must be owned by this UID with mode 0700")


class LocalAdminServer:
    def __init__(self, path: str | Path, handler) -> None:
        self.path = Path(path).absolute()
        private_directory(self.path.parent)
        self._handler = handler
        self._closed = threading.Event()
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            if self.path.exists() or self.path.is_symlink():
                info = self.path.lstat()
                if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
                    raise ValueError("Refusing to replace an unowned/non-socket admin path")
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                    probe.settimeout(1)
                    try:
                        probe.connect(str(self.path))
                    except ConnectionRefusedError:
                        if self.path.lstat().st_ino != info.st_ino:
                            raise ValueError("Admin socket changed during stale check")
                        self.path.unlink()
                    else:
                        raise ValueError("Local admin socket is already in use")
            self._socket.bind(str(self.path))
            os.chmod(self.path, 0o600)
            self._inode = self.path.stat().st_ino
            self._socket.listen(4)
            self._socket.settimeout(0.5)
        except BaseException:
            self._socket.close()
            raise
        self._thread = threading.Thread(target=self._serve, daemon=True, name="local-retrieval-admin")
        self._thread.start()

    def _serve(self) -> None:
        while not self._closed.is_set():
            try:
                connection, _ = self._socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with connection:
                connection.settimeout(5)
                try:
                    _, uid, _ = struct.unpack("3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                    if uid != os.getuid():
                        raise PermissionError("Only the service UID may issue local commands")
                    response = {"ok": True, "result": self._handler(receive_message(connection))}
                except Exception as error:
                    response = {"ok": False, "error": str(error)}
                try:
                    connection.sendall(json.dumps(response, ensure_ascii=False).encode("utf-8") + b"\n")
                except OSError:
                    pass

    def close(self) -> None:
        self._closed.set()
        self._socket.close()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=2)
        try:
            if self.path.lstat().st_ino == self._inode:
                self.path.unlink()
        except FileNotFoundError:
            pass


def send_command(path: str | Path, message: dict, timeout: float = 300) -> dict:
    encoded = json.dumps(message).encode("utf-8") + b"\n"
    if len(encoded) > MAX_MESSAGE:
        raise ValueError("Local command too large")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        connection.connect(str(path))
        connection.sendall(encoded)
        return receive_message(connection)


def main() -> None:
    parser = argparse.ArgumentParser(description="Same-UID Unix-socket retrieval admin; no HTTP endpoint")
    parser.add_argument("--socket", type=Path, default=Path(__file__).resolve().parents[1] / ".runtime/admin.sock")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    change = commands.add_parser("set-query-length")
    change.add_argument("--tokens", type=int, choices=(2048, 8192), required=True)
    reload_parser = commands.add_parser("reload")
    reload_parser.add_argument("--bundle", type=Path, required=True)
    reload_parser.add_argument("--query-tokens", type=int, choices=(2048, 8192))
    commands.add_parser("rollback")
    build = commands.add_parser("build")
    build.add_argument("--input", type=Path, required=True)
    build.add_argument("--version", required=True)
    build.add_argument("--tokens", type=int, choices=(2048, 8192), default=8192)
    commands.add_parser("cancel-build")
    args = parser.parse_args()
    message = {"command": args.command}
    if args.command == "set-query-length":
        message["tokens"] = args.tokens
    if args.command == "reload":
        message["bundle"] = str(args.bundle.resolve())
        if args.query_tokens is not None:
            message["tokens"] = args.query_tokens
    if args.command == "build":
        message.update(input=str(args.input.resolve()), version=args.version, tokens=args.tokens)
    response = send_command(args.socket, message)
    print(json.dumps(response, ensure_ascii=False, indent=2))
    if not response.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
