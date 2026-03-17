import re
import json
import sys
from pathlib import Path

# Regex matching the SpatialRNN training line format
LINE_RE = re.compile(
    r"\[SpatialRNN\] Block (?P<block>\d+)/\d+, "
    r"Batch (?P<batch>\d+)/\d+, "
    r"Loss (?P<loss>\d+\.\d+) "
    r"\(ma\d+ \d+\.\d+\), "
    r"Resp (?P<resp>\d+\.\d+) "
    r"\(ma\d+ \d+\.\d+\), "
    r"Rule (?P<rule>\d+\.\d+) "
    r"\(ma\d+ \d+\.\d+\), "
    r".*?grad_norm (?P<grad>\d+\.\d+)"
)

def parse_log_to_loss_hist(log_text: str):
    loss_hist = []
    for line in log_text.splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue
        block_print = int(m.group("block"))
        batch_print = int(m.group("batch"))

        loss_hist.append(
            dict(
                total=float(m.group("loss")),
                resp=float(m.group("resp")),
                rule=float(m.group("rule")),
                # match train_wcst: block and batch are 0-based floats
                block=float(block_print - 1),
                batch=float(batch_print - 1),
                grad_norm=float(m.group("grad")),
            )
        )
    return loss_hist

def main():
    if len(sys.argv) != 2:
        print("Usage: python parse_spatialrnn_log_to_loss_history.py OUTPUT_PATH.json")
        sys.exit(1)

    out_path = Path(sys.argv[1])
    log_text = sys.stdin.read()
    loss_hist = parse_log_to_loss_hist(log_text)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(loss_hist, f, indent=2)
    print(f"Saved loss history to {out_path} ({len(loss_hist)} entries)")

if __name__ == "__main__":
    main()