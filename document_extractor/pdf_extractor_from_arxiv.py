from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import urllib.parse


class ExtractPdf:
    """_summary_
    
    * Extract research papers  from the Arxiv within certain limits 
    * Extracts the research paper info like Title,Author name
    * NOTE **parameter size/total_researh_paper Parameter** should be only between : (25, 50, 100, or 200)
    
    """
    def __init__(self,query,searchType:str='all',total_reasearch_paper:int = 25) -> None:
        super().__init__()
        '''
        Help: for me 
            * size=50 :Controls the number of results shown per page
            * The dot (.) forces Selenium to look ONLY inside the current 'curr_link' (li tag)
        '''

        self.total_pdf = total_reasearch_paper
        # 1. Properly encode your query text for URLs (handles spaces/special chars)
        encoded_query = urllib.parse.quote_plus(query)
        
        self.url =  f"https://arxiv.org/search/?searchtype={searchType}&query={encoded_query}&abstracts=show&size={total_reasearch_paper}&order=-announced_date_first&size={self.total_pdf}" 

    def extract_authors(self,authors) -> list:
        '''**takes (variable) authors: list[WebElement]** and outputs the 
            list of authors name according to the pdf stored order 
        '''
        elements = authors
        
        authors_names = []
        for i,name in enumerate(elements):
            authors_names.append(name.text)
        
        return authors_names
    
    def extract_titles(self,titles) -> list:
        '''**takes (variable) authors: list[WebElement]** and outputs the 
            list of authors name according to the pdf stored order 
        '''
        elements = titles
        
        all_titles = []
        for i,name in enumerate(elements):
            all_titles.append(name.text)
        
        return all_titles
        
    def extract_pdf(self) -> tuple[list,list,list]:
        """_summary_
        * **Extract Pdf throw url ,extracts about pdf name ,author and title**
        Returns:
            tuple[list,list,list]: _description_
        """
        driver = webdriver.Chrome()
        driver.get(self.url)
        
        # div: "content"
        root_results = driver.find_elements(By.CLASS_NAME, "content") 
        
        #extracting all the children of "content"
        elements = root_results[0]

        # Finds all <li> tags that live inside the <ol class="breathe-horizontal">
        
        results = elements.find_elements(By.CSS_SELECTOR,"ol.breathe-horizontal li")
        
        try:
            
            authors = elements.find_elements(By.CSS_SELECTOR,"ol.breathe-horizontal li p.authors")
            titles = elements.find_elements(By.CSS_SELECTOR,"ol.breathe-horizontal li p.title")
            authors_names = self.extract_authors(authors)
            titles_names = self.extract_titles(titles)
        except NoSuchElementException as e:
            print(f"{e}")
        
        pdf_urls = []
        
        for idx,curr_link in enumerate(results):
        # Target the <a> tag that has the text 'pdf' inside it
        
        # The dot (.) forces Selenium to look ONLY inside the current 'curr_link' (li tag)
            pdf_element = curr_link.find_element(By.XPATH, ".//span/a[text()='pdf']")
        
        # Extract the URL from the href attribute
            pdf_url = pdf_element.get_attribute("href")
            pdf_urls.append(pdf_url)
            
        
        assert len(pdf_urls) == self.total_pdf , "Some pdf urls is extracted and  some not"
        assert len(authors_names) == self.total_pdf , "Some pdf authors_names is extracted and some not"
        assert len(titles_names) == self.total_pdf , "Some pdf titles is extracted and some not"
        
        return pdf_urls,authors_names,titles_names #NOTE : if you see red lines here then this is the problem of Pylance so kindly ignore it :) 

    
    
    

#NOTE For test uncomment this and run

# query = 'Reinforcement Learning'
# extractor = ExtractPdf(query=query,total_reasearch_paper=25)

# pdf_urls,authors_name,titles_name =extractor.extract_pdf()

# print(f"Urls Preview : {pdf_urls[0]}\n Urls Len :  {len(pdf_urls)}")
# print(f"Authors Preview : {authors_name[0]}\n Authors Len :  {len(authors_name)}")
# print(f"titles Preview : {titles_name[0]}\n titles Len :  {len(titles_name)}")


