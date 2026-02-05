from pypdf import PdfReader

file_path = "sbert.pdf"

try:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    with open("pdf_content.txt", "w", encoding="utf-8") as f:
        f.write(text)
        
    print("Successfully extracted text to pdf_content.txt")
    print("-" * 20)
    print(text[:1000]) # Print first 1000 chars to verify

except Exception as e:
    print(f"Error reading PDF: {e}")
