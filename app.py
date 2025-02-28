import tabula
from tabula import read_pdf
from PyPDF2 import PdfReader
import PyPDF2
import pandas as pd
import tensorflow
import torch
import sklearn
import spacy
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import seaborn as sns 
from sklearn.model_selection import train_test_split
import os
from PyPDF2 import PdfReader, PdfWriter
import fitz 
import io
from PIL import Image as PILImage
import pdf2image
from pdf2image import convert_from_path
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import pytesseract
from pytesseract import Output
from ultralyticsplus import YOLO, render_result
import shutil
from pypdf import PdfReader
import fitz
import pdfplumber
from tabula import read_pdf
import tabulate
import tabula
from IPython.display import display_html
from PIL import Image as PILImage
from openpyxl import load_workbook
from io import BytesIO
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
from img2table.document import Image
from img2table.document import PDF
from openpyxl import load_workbook
from io import BytesIO
import cv2
import pytesseract
from img2table.document import PDF
from img2table.ocr import TesseractOCR
import tensorflow as tf
from tensorflow.keras.layers import Input,Concatenate
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import concatenate, Activation, Dropout, Flatten, Dense,AveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D,BatchNormalization, Input, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mae, sparse_categorical_crossentropy, binary_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, Callback,ReduceLROnPlateau,LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from PyPDF2 import PdfWriter, PdfReader
import os
import pytesseract
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
import PyPDF2
import re
from img2table.document import Image
import sys
import csv
import unittest
import json
import roman
from bs4 import BeautifulSoup

class YakootaExtracter:
    """This is the YakootaExtracter class
    This class take as input a pdf and give a json file as output with extracted informations
    Make sure to have this file in the same folder as the pdf file you want to process

    Created on 01/03/2023 16:05
    @autors: Asser Abdel Diarra Axel"""
        

    def __init__(self, model_path=r"C:\PGE 2\AI CLININC\AI OPTIMIZATION FOR DATA EXTRACTION\classifier.h5"):
        self.initial_path = os.getcwd()
        self.labeliser = tf.keras.models.load_model(model_path)

    # Split the pdf into pages to have much better result

    def spliter(self):
        """Spliter function
        Returns True if split succed else return False"""

        input_folder = self.initial_path
        output_folder = os.path.join(input_folder, "output")

        def split_pdfs(input_folder, output_folder): # Imported code from a draft notebook
            for pdf_file_name in os.listdir(input_folder): 
                    if pdf_file_name.endswith(".pdf"):
                        pdf_file_path = os.path.join(input_folder, pdf_file_name)

                        output_subfolder = os.path.join(output_folder, os.path.splitext(pdf_file_name)[0])
                        os.makedirs(output_subfolder, exist_ok=True)

                        with open(pdf_file_path, 'rb') as file:
                            pdf_reader = PdfReader(file)
                            total_pages = len(pdf_reader.pages)

                            for page_num in range(total_pages):
                                pdf_writer = PdfWriter()
                                pdf_writer.add_page(pdf_reader.pages[page_num])

                                output_pdf_path = os.path.join(output_subfolder, f"{os.path.basename(pdf_file_path)}_page_{page_num + 1}.pdf")

                                with open(output_pdf_path, 'wb') as output_file:
                                    pdf_writer.write(output_file)

        split_pdfs(input_folder, output_folder)
    

    
    # Transform each pdf pages into image
    def transformer(self):
        def converter(pdf_folder_path, poppler_path): # Imported function from a draft notebook
                for pdf_folder in os.listdir(pdf_folder_path):
                        chemin = os.path.join(pdf_folder_path, pdf_folder)
                        # Vérification si l'élément est un dossier
                        if os.path.isdir(chemin) : 
                            output_images_folder = os.path.join(pdf_folder_path+"\\"+pdf_folder, 'output_images')
                            if not (os.path.exists(output_images_folder)):
                                os.makedirs(output_images_folder)
                            for pdf_file in os.listdir(pdf_folder_path+"\\"+pdf_folder):
                                if pdf_file.endswith('.pdf'):
                                    pdf_path = os.path.join(os.path.join(pdf_folder_path,pdf_folder), pdf_file)

                                    images = convert_from_path(pdf_path, poppler_path=poppler_path)

                                    for i, image in enumerate(images):
                                        image_id = f"{pdf_file}_page{i + 1}.jpg"
                                        image.save(os.path.join(output_images_folder, image_id), format='JPEG')

        pdf_folder_path = os.path.join(self.initial_path, "output")
        poppler_path = r'C:\PGE 2\AI CLININC\AI OPTIMIZATION FOR DATA EXTRACTION\poppler\Library\bin'

        try :
            converter(pdf_folder_path, poppler_path)
        except :
            print("Unexpected error")
            print(sys.exc_info()[0])
            print(sys.exc_info()[1])
            
    def classifier(self):
        def imported_classifier(pdf_folder_path, output_csv, model): # Imported function from a draft notebook
            def prerpocesser(image_path):

                img_path = image_path 
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(48, 48))
                x = tf.keras.preprocessing.image.img_to_array(img)
                x = tf.image.rgb_to_grayscale(x)
                x = np.expand_dims(x, axis=0)
                return x
            
            with open(output_csv, 'w', newline='') as csv_file:
                fieldnames = ['Image_id', 'Pdf_file', 'Label']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
            
                for pdf_folder in os.listdir(pdf_folder_path):
                    chemin = os.path.join(pdf_folder_path, pdf_folder)
                    if os.path.isdir(chemin) :
                            for img_file in os.listdir(pdf_folder_path+"\\"+pdf_folder+"\\"+"output_images"):
                                if img_file.endswith(".jpg"):
                                    image = prerpocesser(os.path.join(pdf_folder_path+"\\"+pdf_folder+"\\"+"output_images", img_file))
                                    probability_label = np.argmax(model.predict(image))
                                    if probability_label == 0:
                                        label = "table"
                                    elif probability_label == 1:
                                        label = "text"
                                    elif probability_label == 2:
                                        label = "text_table"
                                if "_page1.jpg" in img_file :
                                    pdf_file = img_file.replace("_page1.jpg", "")
                                    writer.writerow({'Image_id': img_file, 'Pdf_file': pdf_file, 'Label': label})

        output_csv_path = f"{self.initial_path}\\val.csv"
        pdf_folder_path = f"{self.initial_path}\\output"

        try :
            imported_classifier(pdf_folder_path, output_csv_path, self.labeliser)
        except :
            print("Unexpected error")
            print(sys.exc_info()[0])
            print(sys.exc_info()[1])
    
    # Function for extraction
    def text_extrater(self, page_path):
        def extract_text(pdf_path=page_path):
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                text = ''
                for page_num in range(len(pdf_reader.pages)):
                    text += pdf_reader.pages[page_num].extract_text()
            return text
        return extract_text()

    def table_extracter(self, img_page_path):
        tesseract_ocr = TesseractOCR(n_threads=1, lang="eng")
        image = img_page_path
        img = Image(image)
        list_html = []
        # Extract tables
        extracted_tables = img.extract_tables(ocr=tesseract_ocr,
                                                    implicit_rows=False,
                                                    borderless_tables=False,
                                                    min_confidence=50)
        for page, table in enumerate(extracted_tables):
            html_string = table.html_repr(title=f"Page {page + 1} - Extracted table n°{page + 1}")
            soup = BeautifulSoup(html_string, 'html.parser')
            html_string = soup.prettify()
            table_bbox = table.bbox
            list_html.append(html_string)
        
        return list_html
    

    def text_table_extraction(self, img_page_path, output_img_drawed_path):
        tesseract_ocr = TesseractOCR(n_threads=1, lang="eng")
        img = Image(img_page_path)
        # Extract tables
        extracted_tables = img.extract_tables(ocr=tesseract_ocr,
                                            implicit_rows=False,
                                            borderless_tables=False,
                                            min_confidence=50)
        try :
            for page, table in enumerate(extracted_tables):
                html_string = table.html_repr(title=f"Page {page + 1} - Extracted table n°{page + 1}")
                soup = BeautifulSoup(html_string, 'html.parser')
                html_string = soup.prettify()
                table_bbox = table.bbox

            for page, table in enumerate(extracted_tables):
                table_bboxes = [table.bbox.to_tuple() for table in extracted_tables]

        except :
            print("             Failed to create bboxes")

        if extracted_tables :
            if table_bboxes :
                try :
            
                    def draw_bounding_boxes(img_path=img_page_path, bboxes=table_bboxes, color=(0, 255, 0), thickness=2, output_path=output_img_drawed_path):
                        img = cv2.imread(img_path)

                        for x1, y1, x2, y2 in bboxes:
                            if (x1 >= 0 and x2 <= img.shape[1] and
                                y1 >= 0 and y2 <= img.shape[0]):
                                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

                        if output_path:
                            cv2.imwrite(output_path, img)

                        return output_path
                    
                except Exception as e:
                    print("Unexpected error")
        
                def save_modified_image_with_inpainting(image_path, save_path, bboxes=table_bboxes, min_area=1000, color_tol=10):
                    """Saves the image with the green box (table) content replaced using inpainting.

                    Args:
                        image_path (str): Path to the image file.
                        bboxes (list of tuples, optional): List of bounding box coordinates of the green boxes [(x1, y1, x2, y2), ...].
                            If None, the function will automatically detect the boxes using color filtering.
                        min_area (int, optional): Minimum area of a text region to be considered. Defaults to 1000.
                        color_tol (int, optional): Tolerance value for color matching. Defaults to 10.
                        save_path (str, optional): Path to save the modified image. Defaults to "modified_image.jpg".
                    """

                    # Read the image and preprocess
                    img = cv2.imread(image_path)
                    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                    green_lower = np.array([40, 50, 50])
                    green_upper = np.array([80, 255, 255])

                    # Creating mask and apply it 

                    mask = cv2.inRange(hsv_img, green_lower, green_upper)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

                    if bboxes is not None:
                        for bbox in bboxes:
                            
                            x1, y1, x2, y2 = bbox
                            box_mask = np.zeros_like(mask)
                            cv2.fillPoly(box_mask, [np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])], 255)

                            # Inpaint area
                            dst = cv2.inpaint(img, box_mask, 3, cv2.INPAINT_NS)
                            img = dst 

                        cv2.imwrite(save_path, dst)

                    else:
                        if len(filtered_contours) == 0:
                            print("No green box found in the image.")
                            return

                        for box_cnt in filtered_contours:
                            x1, y1, w, h = cv2.boundingRect(box_cnt)
                            x2 = x1 + w
                            y2 = y1 + h

                            box_mask = np.zeros_like(mask)
                            cv2.fillPoly(box_mask, [np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])], 255)

                            # Inpaint area
                            dst = cv2.inpaint(img, box_mask, 3, cv2.INPAINT_NS) 

                        cv2.imwrite(save_path, dst)
                    
                    return save_path
                
                def extract_text_outside_box(img_path, bboxes=table_bboxes, min_area=1000, color_tol=10, text_conf=80, ignore_chars="`~!@#$%^&*()-_=+[]{};':|\",<>/?\\"):
                    """Extracts text outside green boxes (tables) from an image.

                    Args:
                        img_path (str): Path to the image file.
                        bboxes (list of tuples): List of bounding box coordinates of the green boxes [(x1, y1, x2, y2), ...].
                        min_area (int, optional): Minimum area of a text region to be considered. Defaults to 1000.
                        color_tol (int, optional): Tolerance value for color matching. Defaults to 10.
                        text_conf (int, optional): Tesseract confidence threshold for text detection. Defaults to 80.
                        ignore_chars (str, optional): Characters to ignore during text extraction. Defaults to "`~!@#$%^&*()-_=+[]{};':|\",<.>/?\\".

                    Returns:
                        list of str: Extracted text outside the green boxes.
                    """

                    # Read the image and preprocess it 
                    img = cv2.imread(img_path)

                    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    green_lower = np.array([40, 50, 50])
                    green_upper = np.array([80, 255, 255])

                    # Initialize list to store extracted text
                    extracted_text = ""

                    for bbox in bboxes:
                        # Creating and applying the mask for each bbox
                        mask = np.zeros(img.shape[:2], dtype=np.uint8)
                        cv2.rectangle(mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255), -1)
                        masked_img = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))

                        gray_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)

                        # Binazation
                        thresh_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                            cv2.THRESH_BINARY, 11, 2)
                        
                        text = pytesseract.image_to_string(thresh_img, config='--psm 6', lang='fra')
                        text = ''.join([char for char in text if char not in ignore_chars])
                        text = text.strip()

                        extracted_text += text
                        extracted_text = str(extracted_text)

                    return extracted_text

                drawed_image_path = draw_bounding_boxes()
                inpaint_drawed_image_path = save_modified_image_with_inpainting(image_path=drawed_image_path, save_path="inpaint_image.jpg")
                text_outised_boxes = extract_text_outside_box(img_path=inpaint_drawed_image_path)

                return [html_string, text_outised_boxes]
            
    def search_and_retrieve(self, df : pd.DataFrame, column, search_value, retrieve_columns):
        result = df[df[column] == search_value][retrieve_columns]
        result = result.squeeze()  # Convert DataFrame to Series
        result = result.apply(lambda x: x if isinstance(x, str) else x.iloc[0])  # Extract values from Series
        return result
    
    # looping to extract text and tables
    def looping_for_extracting_text_and_table(self, folder_path, csv_file):
            chemin = os.path.join(self.initial_path, folder_path)
            if os.path.isdir(chemin): 
                data = pd.read_csv(csv_file)
                html_string_list = []
    
                image_folder = os.path.join(folder_path, "output_images")
                print(data.sample(3))
                print("Successfully load data")
                last_element = os.path.basename(folder_path)

                with open(f"extracted_{last_element}.txt", "w+") as f:

                    for pdf_file in os.listdir(folder_path):
                        if pdf_file.endswith(".pdf"):
                            retrieved = self.search_and_retrieve(data, "Pdf_file", pdf_file, ['Label', 'Image_id'])
                            print("Retrived found")
                            label, img_path = str(retrieved["Label"]), str(retrieved["Image_id"])
                            print("LABEL :", label)
                            print("IMG_PATH : ", img_path)
                            print("Successfully taken out label and ImageID")
                            if label == "text":
                                print("     Extracting text...")
                                text = self.text_extrater(page_path=os.path.join(folder_path, pdf_file))
                                if text :
                                    print(f"Successfully extract text from {pdf_file}")
                                    f.write(text)
                                else :
                                    print(f"Error while Text extraction of {pdf_file}")
                            if label == "table":
                                page_image_path = img_path
                                print("     Extracting table...")
                                tables = self.table_extracter(img_page_path=os.path.join(image_folder, page_image_path))
                                if tables:
                                    print(f"Successfully extract tables from {pdf_file}")
                                    html_string_list.append(tables)
                                else :
                                    print(f"Error while Table extraction of {pdf_file}")
                            if label == "text_table":
                                print("     Extracting text and tables...")
                                text_table = self.text_table_extraction(img_page_path=os.path.join(image_folder, page_image_path), output_img_drawed_path="drawed.jpg")
                                if text_table:
                                    html_string = text_table[0]
                                    text_extracted = text_table[1]
                                    print(f"Successfully extracted text and tables from {pdf_file}")
                                    html_string_list.append(html_string)
                                    f.write(text_extracted)
                                else :
                                    print(f"Error while text table extraction of {pdf_file}")
                        else :
                            print('Error while looping inside pdf folder')

                with open(f"extracted_tables_{last_element}.txt", "w+") as file :
                    for html_string in html_string_list:
                        if isinstance(list, type(html_string)):
                            for string in html_string:
                                file.write(string)
                        elif isinstance(str, type(html_string)):
                            file.write(html_string)

                return html_string_list

    # Function for TOC extraction (normal numerals)

    def extract_toc_p1(self, pdf_path):
        def find_toc_in_text(text, page_num, lexique_count):
            toc = []
            pattern = r'^\d+\.\s+(.*?)(?:\s+(\d+)\s*)?$'  
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                title = match.group(1).strip()
                page_number = match.group(2)
                if page_number:
                    toc.append((title, int(page_number)))
                else:
                    toc.append((title, page_num + 1))
                    
                # In this case we will stop TOC extraction when detecting the "lexique" two times. 
                if 'lexique' in title.lower():
                    lexique_count += 1
                    if lexique_count >= 2:  
                        return toc, lexique_count
            return toc, lexique_count

        toc = []
        lexique_count = 0  
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                text = reader.pages[page_num].extract_text()
                toc_entries, lexique_count = find_toc_in_text(text, page_num, lexique_count)
                toc.extend(toc_entries)
                if lexique_count >= 2:  
                    break
    
        # Clean the str in the tuple and return 
        toc = toc[:-1]
        toc = [(title.replace('.', '').strip(), page_number) for title, page_number in toc]
        toc = [(title.replace('*', '').strip(), page_number) for title, page_number in toc]
        toc_normal = [(title.replace('*', '').strip(), page_number) for title, page_number in toc]
        toc_numeral = [(f'{i+1}' + '.' + title, page_num) for i, (title, page_num) in enumerate(toc)]
        return (toc_normal, toc_numeral)
    
    # Function for TOC extraction (roman numerals)
    def extract_toc_p2(self, pdf_path):
            def find_toc_in_text(text, page_num, lexique_count, sommaire_count, first_title):
                toc = []
                pattern = r'^[IVXLCDM]+\.\s+(.*?)(?:\s+\d+\s*)?$'  
                matches = re.finditer(pattern, text, re.MULTILINE)
                for match in matches:
                    title = match.group(1).strip()
                    if first_title is not None and first_title == title:
                        return toc, lexique_count, sommaire_count, True  
                    if 'lexique' in title.lower() or 'sommaire' in title.lower():
                        title = title.replace('.', '') 
                        if 'lexique' in title.lower():
                            lexique_count += 1
                        if 'sommaire' in title.lower():
                            sommaire_count += 1
                        if lexique_count >= 2 :
                            return toc, lexique_count, sommaire_count, False  
                    toc.append((title, page_num + 1))
                return toc, lexique_count, sommaire_count, False

            toc = []
            lexique_count = 0  
            sommaire_count = 0  
            first_title = None  
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    text = reader.pages[page_num].extract_text()
                    toc_entries, lexique_count, sommaire_count, stop_extraction = find_toc_in_text(text, page_num, lexique_count, sommaire_count, first_title)
                    if stop_extraction:
                        break
                    if toc_entries and first_title is None:
                        first_title = toc_entries[0][0]
                    toc.extend(toc_entries)
                    if lexique_count >= 2 or sommaire_count >= 2:  
                        break
            
            toc_nomal = [(title.replace('.', ''), page_number) for title, page_number in toc]
            toc_roman = [(roman.toRoman(i + 1) + ". " + title, page_number) for i, (title, page_number) in enumerate(toc)]

            return (toc_nomal, toc_roman)

    def detect_numbering_style(self, toc):
        roman_pattern = r'^[IVXLCDM]+[\.|:|-]?'
        normal_pattern = r'^\d+[\.|:|-]?'
        roman_count = 0
        normal_count = 0
        
        for tuple_title in toc:
            if re.match(roman_pattern, tuple_title[0]):
                roman_count += 1
            elif re.match(normal_pattern, tuple_title[0]):
                normal_count += 1
        
        if roman_count > normal_count:
            return "Roman"
        elif normal_count > roman_count:
            return "Normal"
        else:
            return "No"

    def define_toc_and_extraction(self, pdf_path):
        toc_1 = self.extract_toc_p1(pdf_path)[1]
        toc_2 = self.extract_toc_p2(pdf_path)[1]
        tuple_of_toc = (toc_1, toc_2)

        for toc in tuple_of_toc :
            type_toc = self.detect_numbering_style(toc)
            
            if type_toc == "Roman":
                return self.extract_toc_p2(pdf_path)[0]
            elif type_toc == "Normal":
                return self.extract_toc_p1(pdf_path)[0]

    def match_toc_with_data(self, extracted_text_file, html_string_list, toc_tuple, name):
        json_dict = {"text": []}
        
        with open(extracted_text_file, "r") as file:
            text = file.read()

        for i in range(0, len(toc_tuple)-1):
            title_1, _ = toc_tuple[i]
            title_2, _ = toc_tuple[i + 1]
            pattern = re.compile(re.escape(title_1) + r'(.*?)' + re.escape(title_2), re.DOTALL | re.IGNORECASE)
            match = pattern.search(text)
            if match:
                extracted_text = match.group(1).strip()
                json_dict[title_1] = extracted_text

        for i, html_string in enumerate(html_string_list):
            json_dict[f"table {i}"] = html_string
        
        with open(f'extracted_text_{name}.json', 'w') as f:
            json.dump(json_dict, f, indent=4)

    def run(self):
        # Split the PDFs
        self.spliter()
        # Convert to images 
        self.transformer()
        # Classify the images 
        self.classifier()
        # Looping
        html_dict = {}
        for _, pdf_folder_path in enumerate(os.listdir(os.path.join(self.initial_path, "output"))) :
            folder_path = pdf_folder_path
            html_string_list = self.looping_for_extracting_text_and_table(os.path.join(self.initial_path+"\\output", folder_path), csv_file="val.csv")
            html_dict[folder_path] = html_string_list

        for _, pdf_path in enumerate(os.listdir(self.initial_path)):
            if pdf_path.endswith(".pdf"):
                file_name = os.path.splitext(pdf_path)[0]
                pdf_toc = self.define_toc_and_extraction(pdf_path=pdf_path)
                folder_path = os.path.splitext(pdf_path)[0]
                self.match_toc_with_data(f"extracted_{folder_path}.txt", html_string_list=html_dict[file_name], toc_tuple=pdf_toc, name=file_name)
            

Extracter = YakootaExtracter()
Extracter.run()