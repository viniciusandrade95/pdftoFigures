from financial_report_parser.parser import parse_financial_report
from pprint import pprint
import json

if __name__ == "__main__":
    result = parse_financial_report("financial_report_parser/examples/BCP_2024.pdf")
    
    # Save to JSON file
    with open('financial_report.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)



