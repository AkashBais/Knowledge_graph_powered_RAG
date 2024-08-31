import config
from typing import Union


from tqdm.notebook import tqdm
import pandas as pd
from typing import Optional, List, Tuple
from datasets import Dataset
import matplotlib.pyplot as plt

from transformers import AutoTokenizer,AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer

from transformers import pipeline

from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_documents(
    chunk_size: int,
    sections: dict,
    chunk_overlap: Optional[int] = None,
    tokenizer_name: Optional[str] = config.EMBEDDING_MODEL_NAME,
    metadata_dict: Optional[dict] = {},
    visvalize_token_distribution: Optional[bool] = True,
) -> List[LangchainDocument]:

  '''
  This method will create chunks(splits) from the document passed

  Parameters:
  chunk_size: int -> Maximum size of the individual chunks in intiger
  sections: dict -> Dictionary of str Documents to chunk
  chunk_overlap: Optional[int] = None -> Over lap between chunks, If None it is computed as int(chunk_size/10)
  tokenizer_name: Optional[str] = config.EMBEDDING_MODEL_NAME -> Model ID of the embedding model to use
  metadata_dict: Optiona[dict] = {} -> Addation metadata to be appended to chunk dict
  visvalize_token_distribution: Optional[bool] = True -> If true the below code helps us visvalize the size of created chunks
  in terms of tokens
  '''
  
  if chunk_overlap is None:
    chunk_overlap = int(chunk_size * 0.1)

  # print(chunk_overlap)

  '''
  Text splitter that uses HuggingFace tokenizer to count length.
  '''
  assert ("document_name" in metadata_dict.keys()) and (metadata_dict["document_name"] not in([None, "", " "])), "metadata_dict should have a valid document_name"

  # Initialize tokenizer
  text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
      AutoTokenizer.from_pretrained(tokenizer_name),
      chunk_size = chunk_size,
      chunk_overlap  = chunk_overlap,
      add_start_index = True,
      strip_whitespace = True,
      is_separator_regex = False,
  )
  tokenizer = AutoTokenizer.from_pretrained(config.NER_MODEL)
  model = AutoModelForTokenClassification.from_pretrained(config.NER_MODEL)
  ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
  '''
  Applying text splitter to the LandDocuments in the knowledge base
  '''
  chunks_with_metadata = []
  section_sources = {}
  for section,content in sections.items():
    temp_list = []
    for ele in content:
      chunk_seq_id = 0
      if isinstance(ele,str):
        chunks = text_splitter.split_text(ele)
        metadata_dict['chunk_type'] = 'Document'
        metadata_dict['source'] = metadata_dict["document_name"]
        temp_list.append(metadata_dict["document_name"])
      elif isinstance(ele,LangchainDocument):
        chunks = text_splitter.split_text(ele.page_content)
        metadata_dict['chunk_type'] = 'Search'
        metadata_dict['source'] = ele.metadata['source']
        temp_list.append(ele.metadata['source'])

      
      for chunk in chunks:
          ner_entities = get_ner_tag_list(ner_pipeline(chunk))
          temp_dict = {
              'text': chunk, 
              'chunk_seq_id': chunk_seq_id,
              'section': section,#.split('.')[1].strip(), 
              'chunkId' : f'{metadata_dict["document_name"]}--{section.replace(" ","_")}--{metadata_dict["source"]}--chunk{chunk_seq_id:04d}',    
              'documentId' : f'{metadata_dict["document_brand"]}--{metadata_dict["document_source"]}--{metadata_dict["document_name"]}--{metadata_dict["document_date"]}',   
              'node_entities' : ner_entities       
          }
          temp_dict.update(metadata_dict)

          chunks_with_metadata.append(temp_dict)
          chunk_seq_id += 1
        
    section_sources[section] = temp_list
  
  '''
  if visvalize_token_distribution pram is true the below code helps us visvalize the size of created chunks
  in terms of tokens. This is intended to inform:
  a.) Is the size of the chunks created in range of the  embedding model context size
  b.) How the chunk size distribution looks. Are the chinks predomintely small, is so individual chunks might lack relevant context
  '''   
  if visvalize_token_distribution:
    tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL_NAME)
    lengths = [len(tokenizer.encode(doc['text'])) for doc in chunks_with_metadata]
    print(f"Model's maximum sequence length: {SentenceTransformer(config.EMBEDDING_MODEL_NAME).max_seq_length}")
    print(f"Chunks's maximum sequence length: {max(lengths)}")
    fig = pd.Series(lengths).hist()
    plt.title("Distribution of document lengths in the knowledge base (in count of tokens)")
    plt.show()



  return chunks_with_metadata,section_sources



def get_ner_tag_list(ner_results:List[dict]
                    )-> List[str]:
  """
  This method processes the NER tags in the individual chunks 
  """
  ner_entities = set()
  for entity in ner_results:
    tag = entity['entity'].split('-')[1]


    if tag in ['MEDICATION']:
      ner_entities.add('medication')
    if tag in ['DISEASE_DISORDER']:
      ner_entities.add('disease')
    if tag in ['DOSAGE','FREQUENCY','VOLUME','TIME','WEIGHT']:
      ner_entities.add('administration')
    if tag in ['SIGN_SYMPTOM']:
      ner_entities.add('symptom') 
    if tag in ['LAB_VALUE','OUTCOME']:
      ner_entities.add('results') 
    if tag in ['AGE','AREA','SEX','DATE']:
      ner_entities.add('demographics')
 

  return list(ner_entities)
    
      


  