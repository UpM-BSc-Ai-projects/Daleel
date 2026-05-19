import cv2
import numpy as np
import socket
import struct
import pickle
import threading
import time


SERVER_HOST = "192.168.1.18"   # <-- replace with your server's IP
SERVER_PORT = 9999
DEFAULT_N = 5
FRAME_DIFF_THRESHOLD = 12.0  # Mean absolute difference threshold (0-255 scale)


def receive_config(sock: socket.socket) -> int:
    """Receive the initial N value from the server."""
    try:
        raw = sock.recv(4)
        if len(raw) < 4:
            print(f"[Client] Could not read N from server, using default ({DEFAULT_N})")
            return DEFAULT_N
        n = struct.unpack(">I", raw)[0]
        print(f"[Client] Server says: send every {n} frame(s)")
        return n
    except Exception as e:
        print(f"[Client] Error receiving config: {e}. Using default ({DEFAULT_N})")
        return DEFAULT_N


def listen_for_updates(sock: socket.socket, state: dict, stop_event: threading.Event) -> None:
    """Background thread: listens for updated N values from the server."""
    while not stop_event.is_set():
        try:
            raw = sock.recv(4)
            if not raw:
                print("[Client] Server closed the connection.")
                stop_event.set()
                break
            if len(raw) == 4:
                new_n = struct.unpack(">I", raw)[0]
                if new_n > 0 and new_n != state["n"]:
                    print(f"[Client] Server updated N: {state['n']} → {new_n}")
                    state["n"] = new_n
        except Exception as e:
            if not stop_event.is_set():
                print(f"[Client] Listener error: {e}")
            break


def get_frame_difference(frame1, frame2) -> float:
    """Calculate mean absolute difference between two frames."""
    if frame1 is None or frame2 is None:
        return float('inf')
    
    # Convert to grayscale for faster comparison
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate absolute difference
    diff = cv2.absdiff(gray1, gray2)
    
    # Return mean difference
    return np.mean(diff)


def send_frame(sock: socket.socket, frame) -> bool:
    """Serialize and send a single frame. Returns False on failure."""
    try:
        payload = pickle.dumps(frame)
        header = struct.pack(">I", len(payload))
        sock.settimeout(30)
        sock.sendall(header + payload)
        sock.settimeout(None)
        return True
    except Exception as e:
        print(f"[Client] Failed to send frame: {e}")
        return False


def connect_with_retry(host: str, port: int, retries: int = 5, delay: float = 2.0) -> socket.socket:
    """Try to connect to the server, retrying on failure."""
    for attempt in range(1, retries + 1):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            print(f"[Client] Connected to {host}:{port}")
            return sock
        except ConnectionRefusedError:
            print(f"[Client] Connection refused (attempt {attempt}/{retries}). Retrying in {delay}s...")
            time.sleep(delay)
        except Exception as e:
            print(f"[Client] Unexpected error on attempt {attempt}: {e}")
            time.sleep(delay)
    raise RuntimeError(f"[Client] Could not connect to {host}:{port} after {retries} attempts.")


def run_client(host: str = SERVER_HOST, port: int = SERVER_PORT) -> None:
    # ── Connect ────────────────────────────────────────────────────────────
    sock = connect_with_retry(host, port)

    # ── Receive initial N from server ──────────────────────────────────────
    state = {"n": receive_config(sock)}

    # ── Open webcam ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Client] Cannot open webcam. Exiting.")
        sock.close()
        return
    print("[Client] Webcam opened.")

    # ── Start background listener for N updates ────────────────────────────
    stop_event = threading.Event()
    listener = threading.Thread(
        target=listen_for_updates,
        args=(sock, state, stop_event),
        daemon=True
    )
    listener.start()

    # ── Main streaming loop ────────────────────────────────────────────────
    frame_index = 0
    frames_sent = 0
    last_sent_frame = None

    print("[Client] Streaming started. Press Ctrl+C to quit.")
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("[Client] Failed to read from webcam.")
                break

            frame_index += 1

            if frame_index % state["n"] == 0:
                # Check frame difference before sending
                diff = get_frame_difference(frame, last_sent_frame)
                print(f"Difference from last sent frame: {diff:.2f}")
                
                if diff >= FRAME_DIFF_THRESHOLD:
                    ok = send_frame(sock, frame)
                    if not ok:
                        break
                    frames_sent += 1
                    last_sent_frame = frame.copy()
                    print(f"[Client] Sent frame #{frame_index} (diff: {diff:.2f}, total sent: {frames_sent}, N={state['n']})")
                else:
                    print(f"[Client] Frame #{frame_index} skipped (diff: {diff:.2f} < {FRAME_DIFF_THRESHOLD})")

    except KeyboardInterrupt:
        print("\n[Client] Interrupted by user.")
    finally:
        stop_event.set()
        cap.release()
        sock.close()
        print(f"[Client] Done. Total frames sent: {frames_sent}")


if __name__ == "__main__":
    run_client()
