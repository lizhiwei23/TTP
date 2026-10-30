"""Command-line entry point for TTP."""
from ttp.eval_ttp import evaluate, parse_args

if __name__ == "__main__":
    evaluate(parse_args())
