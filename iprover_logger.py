#!/usr/bin/env python3
"""
iprover_logger.py

Standalone script to listen for iProver interactive-mode TCP connections
and record all incoming data both as raw logs and as individual JSON objects.
Each JSON object will be pretty-printed with indentation for readability.
"""
import argparse
import json
import logging
import os
import socket
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Listen for iProver connections and log raw + JSON messages."
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="IP address to bind the logger (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=12300,
        help="TCP port to listen on (default: 12300)"
    )
    parser.add_argument(
        "--raw-log", type=str,
        default=os.path.expanduser("/home/ks/Proof-Guidance-for-Automated-Theorem-Proving-Using-Large-Language-Models/logs/iprover_raw.log"),
        help="File to append raw TCP data"
    )
    parser.add_argument(
        "--json-log", type=str,
        default=os.path.expanduser("/home/ks/Proof-Guidance-for-Automated-Theorem-Proving-Using-Large-Language-Models/logs/iprover_log.jsonl"),
        help="File to append one JSON object per entry, pretty-printed"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity"
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))
    logging.info(f"Starting iProver logger on {args.host}:{args.port}")

    # Open raw log for append
    raw_fd = open(args.raw_log, "a", encoding="utf-8")
    decoder = json.JSONDecoder()
    buffer = ""

    # Start listening
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind((args.host, args.port))
        except OSError as e:
            logging.error(f"Failed to bind to {args.host}:{args.port}: {e}")
            sys.exit(1)
        server.listen(1)
        logging.info("Awaiting iProver connection...")
        conn, addr = server.accept()
        logging.info(f"Connection accepted from {addr[0]}:{addr[1]}")
        try:
            while True:
                data = conn.recv(65536)
                print(f"Received {len(data)} bytes from {addr[0]}:{addr[1]}")
                if not data:
                    logging.info("Peer closed connection")
                    break

                # Decode chunk and write to raw log
                chunk = data.decode("utf-8", errors="ignore")
                print(f"Chunk length: {len(chunk)}")
                print(f"Chunk head: {chunk[:100]}...")
                print(f"Chunk tail: {chunk[-100:]}...")
                raw_fd.write(chunk)
                raw_fd.flush()

                # Accumulate and extract JSON objects from stream
                buffer += chunk
                print(f"Buffer length: {len(buffer)}")
                print(f"Buffer head: {buffer[:100]}...")
                print(f"Buffer tail: {buffer[-100:]}...")
                while True:
                    try:
                        obj, idx = decoder.raw_decode(buffer)
                    except ValueError:
                        logging.debug(f"JSON decode failed, buffer head: {buffer[:100]}")
                        # Not enough data to decode a full JSON object yet
                        break
                    # Append pretty-printed JSON to JSON log
                    with open(args.json_log, "a", encoding="utf-8") as jf:
                        jf.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n\n")
                    logging.debug(f"Logged JSON tag={obj.get('tag')}")
                    # Remove consumed prefix and any leading whitespace
                    buffer = buffer[idx:].lstrip()

        except KeyboardInterrupt:
            logging.warning("Interrupted by user")
        finally:
            raw_fd.close()
            conn.close()
            logging.info("Logger shutting down, files closed.")


if __name__ == "__main__":
    main()
