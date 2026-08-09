import fitz
import re 
import unicodedata
from RagPipeline.tools.text_cleaner import clean_text

class ExtractPdfContent:
    """_summary_
    * **Extract the content on pdf page-by-page with word len ,char len,page number **
    * **NOTE**:  it can't able to extract the images from the pdf so if the pdf images is containing important
        info then it going to miss that informations 
    * contains two methods :
        * **extract_texts** : extract the content from the given docs with its title (for title it is only use to give accessing the docs by its name rather then using index because sometimes you can't able to access needed docs with index)
        * **clean_text** : clean the text after extracting the content from given docs  
    * **need_page_wise**: if the user make this flag true then it will return the docs extracted content from each page and save the  metadata
                        but if not then it will give the whole docs_data  with its metadata as **pdf_title,total_page,page_text,total_char_count,total_word_count**,convinient for "docs > 1" document

    
    """
    def __init__(self) -> None:
        super().__init__()
    
    
    
    ##----------------------------------------------------------
    def extract_texts(self,pdf_url):
        """
        * **Extract each paper at a time**
        * NOTE: if sended one document then it just return document dicctonary else it will
                return list of document 
        """
        
        
        # Open the PDF document
        document = fitz.open(pdf_url)
        total_pages = len(document)
        text = " "
        total_character_count = 0
        total_word_count = 0
        total_page_number = 0
        
        for page_index in range(total_pages):
            page = document[page_index]
            # Extract text from the page
            text += page.get_text()
            # Basic metadata
            
            total_page_number += page_index + 1
            total_word_count += len(text.split())
            total_character_count += len(text)

        doc_info = {
            "page_number": total_page_number,
            "text": clean_text(text.replace("\n", " ")),
            "word_count": total_word_count,
            "character_count": total_character_count
        }
        
        document.close()
        return doc_info
    
    
