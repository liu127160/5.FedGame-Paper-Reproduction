"""Print a readable summary of scalar values in a TensorBoard run."""

import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", help="TensorBoard run directory")
    args = parser.parse_args()

    logdir = Path(args.logdir)
    accumulator = EventAccumulator(str(logdir), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags()

    print("Log directory: {}".format(logdir))
    print("Available tags: {}".format(tags))
    print("\nScalar summaries:")

    scalar_tags = tags.get("scalars", [])
    if not scalar_tags:
        print("  No scalar tags found.")
        return

    for tag in scalar_tags:
        events = accumulator.Scalars(tag)
        values = [float(event.value) for event in events]
        steps = [int(event.step) for event in events]
        print("\n[{}]".format(tag))
        print("  count={}  steps={}..{}".format(len(events), min(steps), max(steps)))
        print("  first={:.6f}  last={:.6f}  min={:.6f}  max={:.6f}".format(
            values[0], values[-1], min(values), max(values)))
        head = [(steps[i], round(values[i], 6)) for i in range(min(5, len(events)))]
        start = max(0, len(events) - 5)
        tail = [(steps[i], round(values[i], 6)) for i in range(start, len(events))]
        print("  first five: {}".format(head))
        print("  last five:  {}".format(tail))


if __name__ == "__main__":
    main()
