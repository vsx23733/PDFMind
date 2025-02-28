# PDF Data Extraction and Structuration Engine

### Overview

This project is an advanced engine designed to process and extract structured, semi-structured, and unstructured data from PDFs while maintaining the highest possible information retention. It then structures the extracted data according to the document's table of contents.

### Features

  - Extracts text, tables, and images from PDFs
  - Uses Deep Learning and OCR for accurate data recognition
  - Detects and reconstructs tables with high precision
  - Assigns extracted content to corresponding sections based on the Table of Contents
  - Outputs structured data in JSON format

### Pipeline Workflow

  - Input PDF: The engine starts with a given PDF file.
  - Split into Pages: Each page is processed individually.
  - Convert to Image: The PDF pages are converted into images for further processing.

### Deep Learning Model Processing:

  - Classifies content as text or table using a simplified ResNet architecture.
  - Outputs label predictions for further extraction.

### Text Extraction:

  - If labeled as text, the extract_text function is used.
  - OCR techniques (Tesseract, OpenCV) are applied for high-accuracy recognition.

### Table Extraction:

  - If labeled as a table, the table_extractor function is used.
  - Table detection is performed using img2table.
  - OCR is used to extract table contents accurately.

### Image Preprocessing and Segmentation:

  - Uses OpenCV to segment images and isolate tables.
  - Inpainting techniques restore missing or occluded data.
  - Assign Data to Titles:
  - Extracts the Table of Contents.
  - Matches extracted data to corresponding sections.

### Output Generation:

  - Structured data is stored as a JSON file.


### Technologies Used:

  - Deep Learning: TensorFlow, ResNet
  - OCR: Tesseract, OpenCV
  - PDF Processing: PdfReader
  - Image Processing: OpenCV
  - Table Extraction: img2table

### Installation

To use this engine, install the required dependencies:
```bash
pip install tensorflow opencv-python pytesseract pdfminer.six img2table
```

### Usage
```bash
from pdf_extractor import PDFExtractor

extractor = PDFExtractor("path/to/pdf.pdf")
data = extractor.process()
print(data)  # Outputs structured JSON
```

### Future Improvements

  Enhance deep learning model accuracy
  Improve table reconstruction for complex layouts
  Support additional document formats (Word, HTML)
  

### Contributors

  - Axel ONOBIONO
  - Asser OMAR

### License

This project is licensed under the MIT License. See the LICENSE file for details.
