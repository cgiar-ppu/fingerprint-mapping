import os
import pandas as pd
from PyPDF2 import PdfReader

def extract_program_proposals_text_to_xlsx():
    """
    Goes to the "ProgramProposals" folder, reads each PDF file, extracts its pages' text,
    and saves it into a new XLSX file where each row corresponds to a PDF,
    and each column in that row corresponds to the text from a specific page.
    """
    folder = "ProgramProposals"

    print(f"Scanning folder '{folder}' for PDF files...")
    if not os.path.isdir(folder):
        print(f"The folder '{folder}' does not exist. Exiting...")
        return

    pdf_files = [file for file in os.listdir(folder) if file.lower().endswith(".pdf")]
    if not pdf_files:
        print("No PDF files found in the folder. Exiting...")
        return

    print(f"Found {len(pdf_files)} PDF file(s). Extracting text now...")
    all_rows = []
    for pdf_file in pdf_files:
        file_path = os.path.join(folder, pdf_file)
        print(f"Reading PDF file: {pdf_file}")
        reader = PdfReader(file_path)
        pages_text = []

        for page in reader.pages:
            text = page.extract_text()
            pages_text.append(text.strip() if text else "")

        row_data = [pdf_file] + pages_text
        all_rows.append(row_data)

    max_cols = max(len(row) for row in all_rows) if all_rows else 0
    for row in all_rows:
        while len(row) < max_cols:
            row.append("")

    column_names = ["Filename"] + [f"Page{i}" for i in range(1, max_cols)]
    df = pd.DataFrame(all_rows, columns=column_names)

    output_file = "program_proposals_extracted.xlsx"
    print(f"Saving extracted data to '{output_file}'...")
    df.to_excel(output_file, index=False)
    print("Extraction complete. Check the current directory for the output file.")

if __name__ == "__main__":
    extract_program_proposals_text_to_xlsx()