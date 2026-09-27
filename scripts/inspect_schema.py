from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="List SQLite tables for migration verification")
    parser.add_argument("database", type=Path)
    arguments = parser.parse_args()

    connection = sqlite3.connect(arguments.database)
    try:
        rows = connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        print([row[0] for row in rows])
    finally:
        connection.close()


if __name__ == "__main__":
    main()
