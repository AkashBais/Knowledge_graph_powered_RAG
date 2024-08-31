from typing import Optional, List, Tuple
from googlesearch import search
from duckduckgo_search import DDGS
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
import nest_asyncio

from langchain.docstore.document import Document as LangchainDocument

def search_web(
    query:str, 
    engine:str="google",
    num_results:int=2, 
    lang:str="eng",
    advanced:bool=True,
    truncate_threshold:int=None,
) -> List[LangchainDocument]:

  '''
  query:str -> The query you want to search over the web
  engine:str="google" -> The web search engine to use
  num_results:int=10  -> The number of web search results to return
  lang:str="eng"  -> The language of the web search [Applicable for Google search]
  advanced:bool=True -> Do we want to return advance search results [Applicable for Google search]
  truncate_threshold:int=None, -> Number of characters to retain from each the search results. If None retain the result as is  
  '''
  print(f"\nSearching {engine} for {query}")
  urls = []
  url_metadata = []

  '''
  Below code block performs 2 functions
  a.) Runs the respective web search APIs [as per the engine argument] to get results regarding a relevant search query
  b.) Extracts the relevent metadata from those APIs regarding the returned search results
  '''
  if engine == "google":
    search_results = search(query, num_results=num_results,advanced=advanced,lang=lang)
    for idx,result in enumerate(search_results):
      urls.append(result.url)
      url_metadata.append(result.title)
  else:
    search_results = DDGS().text(query, max_results=5)
    for idx,result in enumerate(search_results):
      urls.append(result['href'])
      url_metadata.append([result['title'],result['body']])


  '''
  Scrape HTML pages from URLs using a headless instance of Chromium.[headless instance is a browser without a GUI]
  '''
  url_page_content = scrape_url(urls)
  url_page_content_processed = []
  '''
  Truncates the page content of the scraped URLs based on the threshold specified
  '''
  for url_page in url_page_content:
    if truncate_threshold is not None:
      url_page_content_processed.append(truncate(url_page,threshold = truncate_threshold))
    else:
      url_page_content_processed.append(url_page)

  '''
  Adding the retrived metadata to the LangChain Documents
  '''
  for document,metadata in zip(url_page_content_processed,url_metadata):
    
    if isinstance(metadata, List):
      document.metadata['title'] = metadata[0]
      document.metadata['body'] = metadata[1]
    else:
      document.metadata['title'] = metadata
    
  del urls
  del url_page_content
  del url_metadata
  print(f"\nData Fetch completed for {query}")
  return {query : url_page_content_processed}#, {query :  }



    
def scrape_url(
    urls:list,
)->List[LangchainDocument]:

  '''
  Scrape HTML pages from URLs using a headless instance of Chromium.[headless instance is a browser without a GUI]
  Note: We apply nest_asyncio as Langchain suggests we do so if you are using AsyncChromiumLoader in Jupyter notebooks
  '''
  nest_asyncio.apply()
  loader = AsyncChromiumLoader(urls)
  html = loader.load()

  '''
  Transform HTML content by extracting specific tags and removing unwanted ones.
  '''
  bs_html_text_transformer = BeautifulSoupTransformer()
  documents_transformed = bs_html_text_transformer.transform_documents(html,tags_to_extract=["p"], remove_unwanted_tags=["a"])
  
  return documents_transformed

def truncate(
    text:str,
    threshold:int=1000
    )-> str:
    '''
    This function will truncate the text till the threshold character length
    '''
    words = text.split()
    return "".join(words[:threshold])