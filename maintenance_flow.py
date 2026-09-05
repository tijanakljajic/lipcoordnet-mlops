from pathlib import Path
import subprocess
import sys

from metaflow import FlowSpec, step


class LipCoordNetMaintenanceFlow(FlowSpec):

    @step
    def start(self):
        print("Starting LipCoordNet maintenance flow.")
        self.next(self.validate_checkpoint)

    @step
    def validate_checkpoint(self):
        checkpoints = list(Path("pretrain").glob("*.pt"))

        if len(checkpoints) != 1:
            raise RuntimeError(
                f"Expected exactly one checkpoint, found {len(checkpoints)}."
            )

        checkpoint = checkpoints[0]

        if checkpoint.stat().st_size == 0:
            raise RuntimeError("Checkpoint file is empty.")

        self.checkpoint_path = str(checkpoint)
        self.checkpoint_size_mb = round(checkpoint.stat().st_size / (1024**2), 2)

        print(f"Checkpoint: {self.checkpoint_path}")
        print(f"Size: {self.checkpoint_size_mb} MB")

        self.next(self.evaluate_model)

    @step
    def evaluate_model(self):
        subprocess.run(
            [sys.executable, "evaluate_model.py"],
            check=True,
        )

        self.evaluation_status = "passed"
        self.next(self.end)

    @step
    def end(self):
        print("Maintenance flow completed.")
        print(f"Checkpoint validation: passed")
        print(f"Model evaluation: {self.evaluation_status}")


if __name__ == "__main__":
    LipCoordNetMaintenanceFlow()