import argparse


# ARGUMENT PARSER
description = """
use academy_reports to generate daily and intersession reports
"""

parser = argparse.ArgumentParser(description=description)

parser.add_argument('file',
                    type=str,
                    nargs='*',
                    default='',
                    help="to generate the reports for a specific file")

arg = parser.parse_args()