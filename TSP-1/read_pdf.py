
import pypdf

def read_pdf(file_path):
    try:
        reader = pypdf.PdfReader(file_path)
        text = ""
        # Read pages 5 to 10 (0-indexed: 4 to 9)
        for i in range(4, 11): 
            if i < len(reader.pages):
                text += f"--- Page {i+1} ---\n"
                text += reader.pages[i].extract_text() + "\n"
        print(text)
    except Exception as e:
        print(f"Error reading PDF: {e}")

if __name__ == "__main__":
    read_pdf("path_planning.pdf")
