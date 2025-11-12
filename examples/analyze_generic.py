from financial_report_parser.parser import parse_financial_report
from pprint import pprint

if __name__ == "__main__":
    pdf_path = "your_financial_report.pdf"
    result = parse_financial_report(pdf_path)
    pprint(result)
