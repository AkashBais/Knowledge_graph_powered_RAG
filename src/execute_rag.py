# Langchain
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQAWithSourcesChain

import torch
from transformers import pipeline,AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from langchain.embeddings import SentenceTransformerEmbeddings

import concurrent.futures
from ctransformers import AutoModelForCausalLM as AutoModelForCausalLM_CT
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.prompts.prompt import PromptTemplate
from langchain.chains import GraphCypherQAChain

import config
import os
import re

from .create_data_chunks_utils import split_documents
from .parse_pdf import PDFParser
from .KG_utils import embed_chunk_text
from .KG_utils import add_chunk_node
from .KG_utils import clear_graph
from .KG_utils import build_document_structure
from .extract_data_utils import search_web

from src.KG_utils import get_all_chunks

from dotenv import load_dotenv

from langchain_google_vertexai import VertexAIEmbeddings
load_dotenv('/content/RAG/KG_for_RAG/keys.env')

class execute_rag():

  def __init__(self,
               pdf_path:str,
               toc_pages:list[int])-> None:

    """
    Initilize the class

    Parameters:
    pdf_path: Path to the PDF file to process
    toc_pages: List of pages that have the Table of Content
    """
    self.pdf_path = pdf_path
    self.toc_pages = toc_pages
    self._init_graph_for_query = False
    self._init_llm = False

  def pdf_parser_engine(self):
    """
    This method parses the PDF document 
    """
    self.pdf_parser = PDFParser(self.pdf_path, toc_pages = self.toc_pages)
    self.sections = self.pdf_parser.parse_pdf(path = os.path.join(config.PROCESSED_FILE_PATH,'parssed_pdf.pkl'))

  def web_search_engine(self,
                        search_engine:str = "google",
                        search_prefix:str = "Iptacopan",
                        header_exclude_substring:str = "Supplemental Information"):

    """
    This method expands the content of teh document based on web search 

    Parameters:
    search_engine: Search engine to use
    search_prefix: Prefix to be applied to the headers for the web search
    header_exclude_substring: Headers with these substrings will be excluded from the search
    """
    self.search_prefix = search_prefix
    self.header_exclude_substring = header_exclude_substring
    extended_sections = {}

    data_temp = [ (self.search_prefix+ " " + web_search_term.split(".")[1].strip(),search_engine) for web_search_term in self.pdf_parser.section_headings if (re.match(r'^[0-9A-Z]', web_search_term)) and (self.header_exclude_substring.lower() not in web_search_term.lower())]
    with concurrent.futures.ThreadPoolExecutor() as executor:
      self.search_result = list(executor.map(lambda params: search_web(*params), data_temp))

  def collate_results(self):
    """
    This method will combine the Document and Web search results

    """
    self.master_sections = {}
    for term in self.pdf_parser.section_headings:
      if self.header_exclude_substring not in term:
        # print(term)
        temp_list = [val for key,val in self.sections.items() if term.split('.')[1].strip() in key ]
        self.master_sections[term.split('.')[1].strip()] =  [temp_list + ele for dict_ in self.search_result for ele in dict_.values() if self.search_prefix+ " " +term.split(".")[1].strip() in dict_.keys()][0]

  def init_knowledge_graph(self,
                           metadata_dict:dict,
                           chunk_size:int = 512,
                           chunk_overlap:int = None,
                           visvalize_token_distribution:bool = False,
                           reinit_graph:bool = False,
                           vector_index_dimensions:int = 1024)-> bool:
      """
      This method will initilize the Knowledge Graph (KG)

      Parameters:
      metadata_dict: Dictionary of Metadata to be added to teh chunks
      chunk_size: Chunk size for the chunking exercise 
      chunk_overlap: Chunk overlap for the chunking exercise 
      visvalize_token_distribution: Flag to visvalize the token distribution of created chunks
      reinit_graph: Flag to re initilize the KG
      vector_index_dimensions: Dimention of teh vector index in KG
      """

      if chunk_overlap is None:
        chunk_overlap = int(chunk_size * 0.1)

      print('Creating data chunks')
      chunked_document,section_sources = split_documents(chunk_size = chunk_size,
                                                         chunk_overlap = chunk_overlap,
                                                         sections = self.master_sections,
                                                         metadata_dict=metadata_dict,
                                                         visvalize_token_distribution = visvalize_token_distribution)
      if reinit_graph:
        clear_graph()
        print(get_all_chunks())
      print('Adding chunks to graph')
      indices, sample_chunk = add_chunk_node(
                                            chunked_document = chunked_document,
                                            vector_index_dimensions = vector_index_dimensions
                                            )
      
      document_ID = metadata_dict['document_brand'] + '--' + metadata_dict['document_source'] + '--' + metadata_dict['document_name']+ '--' + metadata_dict['document_date']
      status = build_document_structure(section_list = [ele.split('.')[1].strip() for ele in self.pdf_parser.section_headings if re.match(r'^[0-9]', ele)],
                               source_dict = section_sources,
                               document_ID = document_ID )
      return status
      
  def embed_nodes(self,
                  data_dump_batch_size:int = 100,
                  embed_model_type:str = None)-> bool:

    """
    This method will embed the text in all the chunk nodes inside the knowledge graph (KG)

    Parameters:
    data_dump_batch_size: Number of chunk nodes to be embedded in order to update KG [To avoide frequent KG calls]
    embed_model_type: What type of embedding model to use.Currently supports "vertexai" and "HuggingFace" [Pass None for HuggingFace]
    """ 
    status = embed_chunk_text(data_dump_batch_size = data_dump_batch_size,
                              embed_model_type = embed_model_type)

    return status

  def init_llm(self,
               model= None,
               tokenizer= None,
               gemini:bool = False,
               pre_quantized:bool=False,
               save_model_locally:bool = False,
               local_model_path:str = None)->None:
    """
    This method will initilize the LLM

    Parameters:
    model: External model object to be passed
    tokenizer: External tokanizer object to be passed
    gemini: If True will initilize the Gimini model as your LLM
    pre_quantized: If True will initilize a GGUF version of LLaMa 2 as your LLM
    save_model_locally: If True will save the LLM locally
    local_model_path: Local path to load the LLM from
    """
    
    self.gemini = gemini
    self.pre_quantized = pre_quantized    

    if model is None or tokenizer is None:
      
      if not self.gemini:
        
        if not pre_quantized:
          print(f"Initilizing reader LLM as {config.READER_MODEL_NAME}")
          print(f"Initilizing tokenizer as {config.READER_MODEL_NAME}")
          self.tokenizer = AutoTokenizer.from_pretrained(config.READER_MODEL_NAME,truncation=True,model_max_length = 4096)

          if local_model_path is None:
            if torch.cuda.is_available():
              bnb_config = BitsAndBytesConfig(
                  load_in_4bit=True,
                  bnb_4bit_use_double_quant=True,
                  bnb_4bit_quant_type="nf4",
                  bnb_4bit_compute_dtype=torch.bfloat16 
              )


            else:
              bnb_config = None

            self.model = AutoModelForCausalLM.from_pretrained(config.READER_MODEL_NAME,
                                                          quantization_config=bnb_config,
                                                          trust_remote_code=True,
                                                          device_map="auto")


            if save_model_locally:
              print("Saving model")
              self.path = os.path.join(config.MODEL_FOLDER_PATH,config.READER_MODEL_NAME)
              os.makedirs(self.path, exist_ok=True) 
              model.save_pretrained(self.path, from_pt=True)


          else:
            self.model = AutoModelForCausalLM.from_pretrained(local_model_path,
                                                    use_safetensors=True,
                                                    local_files_only=True,
                                                    device_map="auto",
                                                    trust_remote_code=True,)
        else:
            
          print(f"Initilizing reader LLM as 'TheBloke/Llama-2-7B-GGUF'")
          print(f"Initilizing tokenizer as 'hf-internal-testing/llama-tokenizer'")   
          self.model = AutoModelForCausalLM_CT.from_pretrained("TheBloke/Llama-2-7B-GGUF",
                                                           hf=True,
                                                           context_length=3000)
          from transformers import LlamaTokenizerFast
          self.tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer",
                                                              truncation=True,
                                                              model_max_length = 3000,
                                                              legacy = False)
             
        self.READER_LLM = pipeline(
        model=self.model,
        tokenizer = self.tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.5,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens = 1024,

        )
        self.READER_LLM = HuggingFacePipeline(pipeline=self.READER_LLM)


      else:
        print(f"Initilizing reader LLM as {config.READER_LLM_GIMINI}")
        self.READER_LLM = ChatGoogleGenerativeAI(model=config.GEMINI_MODEL, 
                                                 google_api_key=os.getenv('GOOGLE_API_KEY'),
                                                 max_output_tokens = 1024,
                                                 temperature = 0.2,
                                                 verbose=False,
                                                 )

      
    else:
      print(f"Initilizing reader LLM and tokenizer as the passed objects")
      self.READER_LLM = model
      self.tokenizer = tokenizer
      
    print("Reader LLM initilization complet")
    self._init_llm = True

  def init_graph_for_query(self,
                          query_type: str = None,
                          embed_model_type:str = None
                            ):
    """
    This method will initilize the Knowledge Graph (KG) for query

    Parameters:
    query_type: Type of quering method to use. Select from 
                1. prompt: Will genrate Cypher queries using 1/few shot learning 
                2. self_genrate: Will use the LLM to genrate Cypher query based on KG Schema and question
                3. None: Will use the KG to expand the context for chunks retrived using Vector similarity search 
    embed_model_type: External tokanizer object to be passed
    """    
    
    if self._init_llm is not True:
      self._init_llm()

    self.query_type = query_type

    print("Initializing graph for query")
    
    VECTOR_INDEX_NAME = "chunk_text"
    VECTOR_SOURCE_PROPERTY = "text"
    VECTOR_EMBEDDING_PROPERTY = "text_embedding"
    VECTOR_NODE_LABEL = "Chunk"

    NEO4J_URI = os.getenv('NEO4J_URI')
    NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
    NEO4J_DATABASE = os.getenv('NEO4J_DATABASE') or 'neo4j'

    if (query_type is not None) and (query_type.lower() == 'prompt'):

      CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
      Instructions:
      Use only the provided relationship types and properties in the schema.
      Do not use any other relationship types or properties that are not provided.
      Schema:
      {schema}
      Note: Do not include any explanations, apologies, notes, or headers in your responses.
      Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
      Do not include any text in responce except the generated Cypher statement.

      Examples: Here are a few examples of generated Cypher statements for particular questions:
 
      #Which sections talk about medication?
      MATCH (ner:ner_entity)-[:PRESENT_IN]->(chunk:Chunk)
      WHERE ner.name = "medication" 
      RETURN DISTINCT chunk.section AS section_name

      #Which sections talk about medication in the document?
      MATCH (ner:ner_entity)-[:PRESENT_IN]->(chunk:Chunk)
      WHERE ner.name = "medication" AND chunk.chunk_type = "Document"
      RETURN DISTINCT chunk.section AS section_name

      #Which entities are discuessed in section Background?
      MATCH (ner:ner_entity)-[:PRESENT_IN]->(chunk:Chunk)
      WHERE chunk.section = "Background"
      RETURN DISTINCT ner.name AS entity_name

      #Which entities are discuessed in section Background in the document?
      MATCH (ner:ner_entity)-[:PRESENT_IN]->(chunk:Chunk)
      WHERE chunk.section = "Background" AND chunk.chunk_type = "Document"
      RETURN DISTINCT ner.name AS entity_name
        
      The question is:
      {question}""" 

      CYPHER_GENERATION_PROMPT = PromptTemplate(
          input_variables=['schema', "question"], 
          template=CYPHER_GENERATION_TEMPLATE
      )
      CHAIN_QA_TEMPLATE = """You are an assistant that helps to form nice and human understandable answers.
      The information part contains the provided information that you must use to construct an answer.
      The provided information is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
      Make the answer sound as a response to the question. Do not mention that you based the result on the given information.
      Here is an example:

      Question: Which managers own Neo4j stocks?
      Context:[manager:CTL LLC, manager:JANE STREET GROUP LLC]
      Helpful Answer: CTL LLC, JANE STREET GROUP LLC owns Neo4j stocks.

      Follow this example when generating answers.
      If the provided information is empty, say that you don't know the answer.
      Information:
      {context}

      Question: {question}
      Helpful Answer:
      """
      CHAIN_QA_PROMPT = PromptTemplate(
          input_variables=['context', 'question'], 
          template=CHAIN_QA_TEMPLATE
      )

    elif (query_type is not None) and (query_type.lower() == 'self_genrate'):
      CYPHER_GENERATION_PROMPT = None


    
    else:
      retrieval_query_extra_text = """
      MATCH window=
          (:Chunk)-[:NEXT*0..2]->(node)-[:NEXT*0..2]->(:Chunk)
      WITH node, score, window as longestWindow 
        ORDER BY length(window) DESC LIMIT 1
      WITH nodes(longestWindow) as chunkList, node, score
        UNWIND chunkList as chunkRows
      WITH collect(chunkRows.text) as textList,node, score
      RETURN apoc.text.join(textList, " \n ") as text,
          score,
          node {.source,.section,.document_date,.document_brand} AS metadata


      """

      if embed_model_type == 'vertexai':
        import google.auth
        credentials, project_id = google.auth.load_credentials_from_file('/content/RAG/KG_for_RAG/test_vortex_keys.json')
        model = VertexAIEmbeddings(
                                  model_name="text-embedding-004",
                                  credentials = credentials,
                                  project = project_id
                                  )
      elif embed_model_type == None:
        model = SentenceTransformerEmbeddings(model_name = config.EMBEDDING_MODEL_NAME)

      KG_vector_store = Neo4jVector.from_existing_index(
          embedding=model,
          url=NEO4J_URI,
          username=NEO4J_USERNAME,
          password=NEO4J_PASSWORD,
          database="neo4j",
          index_name=VECTOR_INDEX_NAME,
          text_node_property=VECTOR_SOURCE_PROPERTY,
          retrieval_query=retrieval_query_extra_text, 
      )

      # Create a retriever from the vector store
      retriever_extra_text = KG_vector_store.as_retriever(
            # search_type="mmr",
            # search_kwargs={'k': 6, 'fetch_k': 50} #,'lambda_mult': 0.25
      )

    if (query_type is not None) and (query_type.lower() in ['self_genrate','prompt']):
      
      self.QA_CHAIN = GraphCypherQAChain.from_llm(
          graph=self.return_graph(),
          llm=self.READER_LLM,
          verbose=True,
          cypher_prompt=CYPHER_GENERATION_PROMPT,
          # qa_prompt=CHAIN_QA_PROMPT,
      )
    else:
      # Create a chatbot Question & Answer chain from the retriever
      self.QA_CHAIN = RetrievalQAWithSourcesChain.from_chain_type(
          self.READER_LLM, 
          chain_type="stuff", 
          retriever=retriever_extra_text,
          return_source_documents=True,
          verbose=False,
      )

    self._init_graph_for_query = True

  def query(self,query):
    """
    This method will query the Knowledge Graph (KG)

    Parameters:
    query: query to be exicuted
    """ 
    if self._init_llm is False:
      self.init_llm()
    if self._init_graph_for_query is False:
      self.init_graph_for_query()


    if (self.query_type is not None) and (self.query_type.lower() in ['self_genrate','prompt']):
      answer = self.QA_CHAIN.invoke(
          {"query": query,},
          return_only_outputs=True,
      )
    elif self.query_type is None:
      answer = self.QA_CHAIN.invoke(
          {"question": query},
          return_only_outputs=True,
      )
    # 
    return answer

  def return_graph(self):
    """
    This method returns an instance of the Knowledge Graph(KG)
    """
    return Neo4jGraph(
                          url=os.getenv('NEO4J_URI'),
                          username=os.getenv('NEO4J_USERNAME'),
                          password=os.getenv('NEO4J_PASSWORD'),
                          database=os.getenv('NEO4J_DATABASE'),
                          )


