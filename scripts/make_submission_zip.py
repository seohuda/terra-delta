#!/usr/bin/env python3
import argparse
from terradelta.submission import make_submission_zip


def main():
    p = argparse.ArgumentParser(description="Build root-layout ZIP locally; never uploads")
    p.add_argument("--source", default="outputs/submission_export")
    p.add_argument("--output", default="outputs/submission.zip")
    a = p.parse_args()
    print(make_submission_zip(a.source, a.output))


if __name__ == "__main__":
    main()
