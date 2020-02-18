"""Gets banking data from:
  https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data

Dataset paper: TODO

Copyright PolyAI Limited.
"""
# TODO: test after the data is hosted publicly
import argparse
import os

import requests

_GITHUB_URL_BASE = ("https://github.com/PolyAI-LDN/task-specific-datasets/tree"
                    "/master/banking_data")
_FILES = ["train.csv", "test.csv", "val.csv", "categories.json"]


def _get_script_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        help="Path to dir where to save train, test, validation, "
             "categories.json",
        required=True
    )
    return parser.parse_args()


def _main():
    flags = _get_script_flags()
    data_dir = flags.data_dir

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    for fname in _FILES:
        print(f"Getting file: {fname}")
        remote_url = _GITHUB_URL_BASE + fname
        local_path = os.path.join(data_dir, fname)

        request = requests.get(remote_url)
        with open(local_path, "w") as f:
            f.write(request.content)


if __name__ == "__main__":
    _main()
