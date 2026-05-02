"""
Server — saves frames from each client into its own folder.

Usage:
    python server.py --save-path /path/to/frames [--port 8888] [--n 5]

While running, type a new number + Enter to push a new N to all clients.
"""

import socket
import struct
import pickle
import threading
import argparse
import os
import cv2


# ── Shared state ─────────────────────────────────────────────────────────────
clients: list[socket.socket] = []
clients_lock = threading.Lock()
current_n = 5
SAVE_BASE_PATH = r"C:\Users\ADMIN\Downloads\Daleel\cameras"


def broadcast_n(n: int) -> None:
    """Send an updated N to every connected client."""
    msg = struct.pack(">I", n)
    with clients_lock:
        dead = []
        for c in clients:
            try:
                c.sendall(msg)
            except Exception:
                dead.append(c)
        for c in dead:
            clients.remove(c)


def make_client_folder(base_path: str, addr: tuple) -> str:
    """
    Create a folder for a client inside base_path.
    Folder name: client_<IP>_<PORT>
    Returns the folder path.
    """
    folder_name = f"client_{addr[0]}_{addr[1]}"
    folder_path = os.path.join(base_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    print(f"[Server] Created folder: {folder_path}")
    return folder_path


def handle_client(conn: socket.socket, addr: tuple, base_path: str) -> None:
    global current_n
    print(f"[Server] Client connected: {addr}")

    # Create a dedicated folder for this client
    client_folder = make_client_folder(base_path, addr)

    # Send initial N
    conn.sendall(struct.pack(">I", current_n))

    with clients_lock:
        clients.append(conn)

    frames_received = 0
    try:
        while True:
            # Read 4-byte length header
            raw_len = _recv_exact(conn, 4)
            if raw_len is None:
                break
            (payload_len,) = struct.unpack(">I", raw_len)

            # Read payload
            payload = _recv_exact(conn, payload_len)
            if payload is None:
                break

            frame = pickle.loads(payload)
            frames_received += 1

            # Save frame as an image file
            filename = f"frame_{frames_received:06d}.jpg"
            filepath = os.path.join(client_folder, filename)
            cv2.imwrite(filepath, frame)

            print(f"[Server] Saved frame #{frames_received} from {addr}  → {filepath}")

            # Display frame
            cv2.imshow(f"Client {addr}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"[Server] Error with {addr}: {e}")
    finally:
        with clients_lock:
            if conn in clients:
                clients.remove(conn)
        conn.close()
        cv2.destroyAllWindows()
        print(f"[Server] Client {addr} disconnected. Total frames saved: {frames_received}")


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    """Read exactly n bytes from sock, or return None on EOF/error."""
    buf = b""
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except Exception:
            return None
        if not chunk:
            return None
        buf += chunk
    return buf


def console_input_loop() -> None:
    """Let the operator type a new N value at any time."""
    global current_n
    while True:
        try:
            raw = input()
            new_n = int(raw.strip())
            if new_n < 1:
                print("[Server] N must be >= 1")
                continue
            current_n = new_n
            print(f"[Server] Broadcasting new N={current_n} to all clients ...")
            broadcast_n(current_n)
        except ValueError:
            print("[Server] Please enter a valid integer.")
        except (EOFError, KeyboardInterrupt):
            break


def run_server(host: str, port: int, initial_n: int, save_path: str) -> None:
    global current_n
    current_n = initial_n

    # Ensure the base save path exists
    os.makedirs(save_path, exist_ok=True)
    print(f"[Server] Saving frames to: {os.path.abspath(save_path)}")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(5)
    print(f"[Server] Listening on {host}:{port}  (initial N={current_n})")
    print("[Server] Type a number + Enter to push a new N to all clients.\n")

    # Background thread for console input
    t = threading.Thread(target=console_input_loop, daemon=True)
    t.start()

    try:
        while True:
            conn, addr = srv.accept()
            threading.Thread(
                target=handle_client,
                args=(conn, addr, save_path),
                daemon=True
            ).start()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down.")
    finally:
        srv.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",      type=int, default=9999,           help="Port to listen on")
    parser.add_argument("--n",         type=int, default=5,              help="Initial frame-skip value")
    args = parser.parse_args()


    run_server(host="0.0.0.0", port=args.port, initial_n=args.n, save_path=SAVE_BASE_PATH)
