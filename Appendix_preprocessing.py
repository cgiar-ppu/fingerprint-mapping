import os
import pdfplumber
import pandas as pd

def extract_tables_with_pdfplumber():
    """
    Goes to the 'ProposalsAppendix' folder, finds each PDF file, extracts tables
    from all pages, and saves each PDF's tables into an Excel file with one sheet
    per detected table.
    """
    folder = "ProposalsAppendix"

    print(f"Scanning folder '{folder}' for PDF files...")
    if not os.path.isdir(folder):
        print(f"The folder '{folder}' does not exist. Exiting...")
        return

    pdf_files = [file for file in os.listdir(folder) if file.lower().endswith(".pdf")]
    if not pdf_files:
        print("No PDF files found in the folder. Exiting...")
        return

    print(f"Found {len(pdf_files)} PDF file(s). Extracting tables with pdfplumber...")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder, pdf_file)
        print(f"Processing PDF file: {pdf_file}")

        # Open the PDF using pdfplumber
        try:
            with pdfplumber.open(pdf_path) as pdf:
                tables_list = []
                
                # Iterate over pages to extract tables
                for page_index, page in enumerate(pdf.pages, start=1):
                    # extract_table() returns a list of lists if a table is found,
                    # or None if no table is detected.
                    table = page.extract_table()

                    if table:
                        # The first row is often treated as a header row
                        # but that depends on your actual PDF's structure.
                        # For a simple approach, treat the first row as column headers
                        columns = table[0]
                        data_rows = table[1:]  # the rest of the rows

                        df = pd.DataFrame(data_rows, columns=columns)
                        # Clean up the dataframe as needed (optional)
                        df.fillna("", inplace=True)

                        # Save the table plus a page reference
                        tables_list.append((page_index, df))
        
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
            continue

        # If no tables were extracted, skip writing an Excel
        if not tables_list:
            print(f"No tables found in {pdf_file}. Skipping...")
            continue
        
        # Create an Excel file with each table on a separate sheet
        output_excel = os.path.splitext(pdf_file)[0] + "_pdfplumber_tables.xlsx"
        output_path = os.path.join(folder, output_excel)
        
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for idx, (page_number, table_df) in enumerate(tables_list, start=1):
                sheet_name = f"Page_{page_number}_Table_{idx}"
                table_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"Extracted tables from '{pdf_file}' and saved to '{output_excel}'.")
    
    print("All PDFs have been processed.")

if __name__ == "__main__":
    extract_tables_with_pdfplumber()