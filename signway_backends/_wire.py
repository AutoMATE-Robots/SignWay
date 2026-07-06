"""Length-prefixed pickle framing over a TCP socket. Stdlib only, so it works identically in the
omnivla (3.10) and habitat (3.9) envs. Pickle protocol 4 is compatible with both."""
from __future__ import annotations

import pickle
import struct


def send_msg(sock, obj) -> None:
    data = pickle.dumps(obj, protocol=4)
    sock.sendall(struct.pack(">I", len(data)) + data)


def _recv_all(sock, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed mid-message")
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock):
    (length,) = struct.unpack(">I", _recv_all(sock, 4))
    return pickle.loads(_recv_all(sock, length))
