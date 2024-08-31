from pdfminer.high_level import extract_text
import re
import os
import shutil
import pickle 

 
class PDFParser:
    def __init__(self, file_path: str,
                      toc_pages:list[int],
                      toc_extract_section_regex = r'(([0-9]|[A-Z]){1}\.\ *[A-Za-z :-]+\.+\ *[A-Z]*[0-9]+)',
                      new_section_regex = r'(([0-9]|[A-Z]){1}\.\ *[A-Za-z :-]+\ *)'):


        """
        Initilize the PDF parser

        Paramaters:
        file_path: Path to the PDF file to process
        toc_pages: List of pages that have the Table of Content 
        toc_extract_section_regex: Regex to extract headers form pages in TOC list
        new_section_regex: Regex to identify new section in the document body
        """
        self.file_path = file_path
        self.toc_pages = toc_pages
        self.toc_extract_section_regex = toc_extract_section_regex
        self.new_section_regex = new_section_regex
        self.section_headings = []

    def parse_section_heading(self):
        """
        This method will parse the section headings fom the TOC pages using the toc_extract_section_regex
        """
        print('Extracting headers')
        text = extract_text(self.file_path, page_numbers=self.toc_pages)
        for line in text.split('\n'):

            if self.is_heading(self.toc_extract_section_regex,line.strip()):
              self.section_headings.append( re.search(self.new_section_regex, line.strip()).group().strip() )


    def parse_pdf(self,
                  path:str = None):
        """
        This method will parse teh PDF

        Parameters:
        path: Path to save the parsed file
        """
        self.parse_section_heading()
        text = extract_text(self.file_path)
        sections = self.split_into_sections(text)
        if path is not None:
          print('Saving Results')
          with open(path, 'wb') as f:
            pickle.dump(sections, f)
          
        return sections
   
    def split_into_sections(self, 
                           text:str):
        """
        This method will split the passed document text into sections

        Paramaters:
        text: Text of teh document to be parsed
        """
        print('Splitting into sections')
        lines = text.split('\n')
        sections = {'Pre-section': ''} # Initialize with 'Pre-section' key
        current_section = 'Pre-section'
        section_count = 0

        for idx, line in enumerate(lines):

            if self.is_heading(self.new_section_regex ,line.strip()):

              if (self.section_headings[section_count].lower().strip() in line.lower().strip()) or (self.section_headings[section_count].lower().strip() in line.lower().strip() +" " + lines[idx + 1].lower().strip()):

                if self.section_headings[section_count].lower().strip() in line.lower().strip():
                  current_section = line.strip()
                else:
                  current_section = line.strip() +" " + lines[idx + 1].strip() #.split('.')[1].strip()
                print(f"Parsing {current_section}.")
                sections[current_section] = '' # Create new section when heading is found

                if section_count < len(self.section_headings) - 1:
                  section_count += 1
            else:

                if not line.strip() in ['Evidence Report – Paroxysmal Nocturnal Hemoglobinuria','©Institute for Clinical and Economic Review, 2024','Return to Table of Contents']:
                  sections[current_section] += line.strip() +'\n'# Append to the current section

        return sections

    def is_heading(self,
                  regex:str,
                  line:str):
        """
        This is a helper method to identify the section headers

        Paramaters:
        regex: Regex to identify the sections
        line: Line in which the header is to be identified 
        """                 
        if re.fullmatch(regex, line) is not None:
          return re.fullmatch(regex, line)
        else:
          return False