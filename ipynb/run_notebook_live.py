#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


class LiveNotebookClient(NotebookClient):
    async def async_execute_cell(self, cell, cell_index, execution_count=None, store_history=True):
        if cell.get("cell_type") == "code":
            first_line = next((line.strip() for line in cell.get("source", "").splitlines() if line.strip()), "")
            print(f"\n===== CELL {cell_index + 1} START ===== {first_line[:100]}", flush=True)

        try:
            return await super().async_execute_cell(cell, cell_index, execution_count, store_history)
        finally:
            if cell.get("cell_type") == "code":
                print(f"\n===== CELL {cell_index + 1} END =====", flush=True)

    def process_message(self, msg, cell, cell_index):
        msg_type = msg.get("msg_type")
        content = msg.get("content", {})

        if msg_type == "stream":
            stream = sys.stderr if content.get("name") == "stderr" else sys.stdout
            print(content.get("text", ""), end="", file=stream, flush=True)
        elif msg_type == "error":
            print("\n".join(content.get("traceback", [])), file=sys.stderr, flush=True)
        elif msg_type in {"execute_result", "display_data"}:
            data = content.get("data", {})
            text = data.get("text/plain")
            if text:
                if isinstance(text, list):
                    text = "".join(text)
                print(text, flush=True)

        return super().process_message(msg, cell, cell_index)


def main():
    parser = argparse.ArgumentParser(description="Execute a notebook and stream cell output to stdout.")
    parser.add_argument("input", help="Input notebook path")
    parser.add_argument("--output", required=True, help="Executed notebook output path")
    parser.add_argument("--timeout", type=int, default=-1, help="Cell timeout in seconds; -1 disables timeout")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open(encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    client = LiveNotebookClient(
        notebook,
        timeout=args.timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(Path.cwd())}},
    )

    try:
        client.execute()
    finally:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            nbformat.write(notebook, f)
        print(f"\nSaved executed notebook: {output_path}", flush=True)


if __name__ == "__main__":
    main()
