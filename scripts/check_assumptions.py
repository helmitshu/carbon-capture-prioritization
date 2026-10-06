"""Monthly staleness check for the carbon capture economics assumptions.

Reads assumptions.py, compares today against each assumption's review
period, and prints the result. Exit code 0 in both cases; the caller
(the monthly cron) decides whether to alert Hakim. Prints ALL CURRENT
when nothing is overdue, otherwise one OVERDUE line per assumption.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assumptions import overdue_assumptions


def main() -> None:
    overdue = overdue_assumptions()
    if not overdue:
        print("ALL CURRENT")
        return
    print(f"OVERDUE: {len(overdue)}")
    for a in overdue:
        print(f"- {a['name']}: {a['status_text']}")


if __name__ == "__main__":
    main()
