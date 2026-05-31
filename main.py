import sys

from pyframe import Pipe

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "media/HCBHD36W0AI3Hz4.jpeg"

    # Local HuggingFace backend (free, bring your own model with model="...")
    result = Pipe(path, backend="local", max_frames=10).run()
    print(f"{result.source}: {result.verdict.value} ({result.max_score:.0%})")

    # AWS Rekognition backend
    # Pipe(path, backend="aws", region="us-east-1", min_confidence=0.8).run()

    # Two-stage cascade: local soft-screen gates the expensive AWS pass
    # Pipe(path, backend="aws", prescreen=True).run()

    # Merge frames into a grid before classifying
    # Pipe(path, backend="aws", use_merged=True, frames_per_batch=2).run()
