import fitz

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page in enumerate(pdf):
            page_text = page.get_text()
            text += page_text
    return text

pdf_input = input("PDF file paths: ")
pdf_files = [file.strip() for file in pdf_input.split(",")]

all_text = ""
for pdf in pdf_files:
    try:
        text = extract_text_from_pdf(pdf)
        all_text += text + "\n"
    except Exception as e:
        print(f"Error reading {pdf}: {e}")

output_file = "extracted_annual_reports.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(all_text)

print(f"\nText extraction completed. Saved to '{output_file}'.")
