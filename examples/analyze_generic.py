import os
from pprint import pprint

from financial_report_parser.llm_client import LLMClient
from financial_report_parser.parser import parse_financial_report

if __name__ == "__main__":
    pdf_path = "your_financial_report.pdf"
    llm_client = None
    base_url = os.environ.get("LLM_BASE_URL")
    if base_url:
        try:
            llm_client = LLMClient(base_url)
        except ValueError as exc:
            print(f"Skipping LLM integration: {exc}")
    result = parse_financial_report(pdf_path, llm_client=llm_client)
    pprint(result)
