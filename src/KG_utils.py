from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv
import os
import sys

sys.path.append('/content/RAG/KG_for_RAG/')
import config

import neo4j
from sentence_transformers import SentenceTransformer

from langchain_google_vertexai import VertexAIEmbeddings

load_dotenv('/content/RAG/KG_for_RAG/keys.env', override=True)
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE') or 'neo4j'


def add_chunk_node(chunked_document: list[dict],
                   vector_index_dimensions: int = 1024,
                   add_unique_chunk_constraint:bool = True,
                   add_ner_entities:bool = True,
                   add_vector_index:bool = True,
                   )-> list:
                   
  """
  This method adds chunk nodes to the knowledge graph (KG)

  Parameters:
  chunked_document: List of chuncked documents that are to be added to the KG
  vector_index_dimensions: Dimention of the vector index in KG. Depends on the embedding model you choose to use
                           Same as the dimention of the embedding vectors produced by the embedding model
  add_unique_chunk_constraint: Flag to add a uniqueness constrant to the Graph Nodes
  add_ner_entities: Flag that will create nodes for various ner entities
  add_vector_index: Flag that will add a vector index to the KG
  """


  KnowledgeGraph =  Neo4jGraph(
                            url=NEO4J_URI,
                            username=NEO4J_USERNAME,
                            password=NEO4J_PASSWORD,
                            database=NEO4J_DATABASE,
                            )
  
  # config.MERGE_CHUNK_NODE_QUERY
  print("Adding chunks with metadata to the graph")
  for chunk in chunked_document:
    merged_chunks = KnowledgeGraph.query('''
    MERGE (MergedChunks:Chunk {chunkID: $chunkParam.chunkId})
      ON CREATE SET 
        MergedChunks.text = $chunkParam.text,
        MergedChunks.chunk_seq_id = $chunkParam.chunk_seq_id,
        MergedChunks.section = $chunkParam.section,
        MergedChunks.document_source = $chunkParam.document_source,
        MergedChunks.document_date = $chunkParam.document_date,
        MergedChunks.document_name = $chunkParam.document_name,
        MergedChunks.document_brand = $chunkParam.document_brand,
        MergedChunks.documentId = $chunkParam.documentId,
        MergedChunks.chunk_type = $chunkParam.chunk_type,
        MergedChunks.source = $chunkParam.source,
        MergedChunks.node_entities = $chunkParam.node_entities
    Return MergedChunks
    ''', params = {'chunkParam': chunk})

  if add_unique_chunk_constraint:
    config.ADD_UNIQUE_CHUNK_CONSTRANT_QUERY
    KnowledgeGraph.query("""
    CREATE CONSTRAINT unique_chunk IF NOT EXISTS
    FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE
    """)
  if add_ner_entities:
    param_dict = [
      {'name': 'medication','included_tags': ['MEDICATION'],'search_key_word':['medication']},
      {'name': 'disease','included_tags': ['DISEASE_DISORDER'],'search_key_word':['disease']},
      {'name': 'administration','included_tags': ['DOSAGE','FREQUENCY','VOLUME','TIME','WEIGHT'],'search_key_word':['administration']},
      {'name': 'symptom','included_tags': ['SIGN_SYMPTOM'],'search_key_word':['symptom']},
      {'name': 'results','included_tags':  ['LAB_VALUE','OUTCOME'],'search_key_word':['results']},
      {'name': 'demographics','included_tags':  ['AGE','AREA','SEX','DATE'],'search_key_word':['demographics']},
    ]
    for params in param_dict:
      KnowledgeGraph.query("""
          MERGE (node:ner_entity {name: $name}) 
            ON CREATE SET
              node.included_tags = $included_tags,
              node.search_key_word = $search_key_word
          """,params = params)
  if add_vector_index:
    config.ADD_VECTOR_INDEX_QUERY
    KnowledgeGraph.query("""
    CREATE VECTOR INDEX `chunk_text` IF NOT EXISTS
    FOR (c:Chunk) ON (c.text_embedding)
    OPTIONS {indexConfig : {
    `vector.dimensions`: $vector_dimensions,
    `vector.similarity_function`: 'cosine'
    }}
    """, params = {'vector_dimensions': vector_index_dimensions}) 

  print(f"Total count a chunks added to the graph {get_all_chunks()}")
  return KnowledgeGraph.query("SHOW INDEXES"),merged_chunks[0]

def build_document_structure(
                              section_list:list[str],
                              source_dict:dict,
                              document_ID:str,
                            ) -> bool:

    """
    This method recreates the document structure inside the knowledge graph (KG)

    Parameters:
    section_list: List of diffrent sections in the document
    source_dict: List of diffrent section sources. EG: Are the document section or Web sections
    document_ID: All nodes with the same document ID will be used to create the document
    """ 


    URI = NEO4J_URI
    AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)
    driver = neo4j.GraphDatabase.driver(URI, auth=AUTH)
    driver.verify_connectivity()

    with driver.session(database=NEO4J_DATABASE) as session:
    
      print(f'Building structure for {document_ID}')

     # Connect chunks to each other via a NEXT relationship
      for section in section_list:
        for source in source_dict[section.strip()]:
          
          chunks_linked = session.run("""
          MATCH (sampleChunk:Chunk {documentId : $document_ID , section : $section , source: $source})
          WITH sampleChunk
          ORDER BY sampleChunk.chunk_seq_id ASC
          WITH collect(sampleChunk) as sampleChunk_list
          CALL apoc.nodes.link(
            sampleChunk_list,
            "NEXT",
            {avoidDuplicates: true}
          )
          RETURN size(sampleChunk_list)
          """, document_ID = document_ID, section = section.strip(),source = source)


      #Creating a document node from the chunks
      doc_info_list = session.run("""
      MATCH (sampleChunk:Chunk {documentId : $document_ID})
      WITH sampleChunk LIMIT 1
      RETURN sampleChunk {.document_source , .document_date , .document_name , .document_brand, .documentId} 
      as documentInfo 
      """, document_ID = document_ID)
    
      for doc_info in doc_info_list:
        doc_info = dict(doc_info)
        session.run("""
          MERGE (document:Document {documentId : $doc_info.documentId}) 
            ON CREATE
              SET document.document_source = $doc_info.document_source
              SET document.document_date = $doc_info.document_date
              SET document.document_name = $doc_info.document_name
              SET document.document_brand = $doc_info.document_brand
          """, doc_info = doc_info['documentInfo'])
             
          
      #connecting chunks to the respective documents
      new_IS_PART_OF_created = session.run("""
      MATCH (chunk:Chunk {documentId : $document_ID}) , (document:Document {documentId : $document_ID})
      WITH chunk, document
      MERGE (chunk)-[new_partOfRelation:IS_PART_OF {source: chunk.source, chunk_type:chunk.chunk_type}]->(document)
      RETURN count(new_partOfRelation) as new_relations_created
      """, document_ID = document_ID)

      print(f"Created {new_IS_PART_OF_created.single()} IS_PART_OF relations")

      #connecting chunks to the respective ner entities 
      new_PRESENT_IN_created = session.run("""
      MATCH (n1:Chunk {documentId : $document_ID}), (n2:ner_entity)
      WHERE n2.name IN n1.node_entities
      CREATE (n2)-[new_present_in:PRESENT_IN]->(n1)
      RETURN count(new_present_in) as new_relations_created
      """, document_ID = document_ID)
      print(f"Created {new_PRESENT_IN_created.single()} PRESENT_IN relations")


      #connecting sirst chunk of a section to the respective documents
      new_SECTION_START_created = session.run("""
      MATCH (chunk:Chunk {documentId : $document_ID}) , (document:Document {documentId : $document_ID})
      WHERE chunk.chunk_seq_id = 0
      WITH chunk, document
      MERGE (chunk)-[new_sectionRelation:SECTION_START {section : chunk.section , source: chunk.source, chunk_type:chunk.chunk_type}]->(document)
      RETURN count(new_sectionRelation) as new_relations_created
      """, document_ID = document_ID)

      print(f"Created {new_SECTION_START_created.single()} SECTION_START relations")

      return  True
    
    

def embed_chunk_text(
                  data_dump_batch_size:int = 10,
                  embed_model_type:str = None
                  )-> bool :
    """
    This method will embed the text in all the chunk nodes inside the knowledge graph (KG)

    Parameters:
    data_dump_batch_size: Number of chunk nodes to be embedded in order to update KG [To avoide frequent KG calls]
    embed_model_type: What type of embedding model to use.Currently supports "vertexai" and "HuggingFace" [Pass None for HuggingFace]
    """ 

    URI = NEO4J_URI
    AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)
    driver = neo4j.GraphDatabase.driver(URI, auth=AUTH)
    driver.verify_connectivity()

    # Initialize Sentence Transformer
    
    if embed_model_type == 'vertexai':
      import google.auth
      credentials, project_id = google.auth.load_credentials_from_file('/content/RAG/KG_for_RAG/test_vortex_keys.json')
      model = VertexAIEmbeddings(
                                model_name="text-embedding-004",
                                credentials = credentials,
                                project = project_id
                                )
    elif embed_model_type == None:
      model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    chunk_with_embeddings = []
    batch_count = 0

    def run_batch(driver,chunk_with_embeddings,batch_count):
      """
      This is an internal support method that updates the KG in batches

      """

      driver.execute_query("""
      UNWIND $chunk_list as chunk
      MATCH (c:Chunk {chunkID : chunk.ID,text : chunk.text})
      CALL db.create.setNodeVectorProperty(c,'text_embedding',chunk.text_embedding)
      """, chunk_list=chunk_with_embeddings, database_= NEO4J_DATABASE)

      print(f'Successfull processed batch {batch_count}')

    

    with driver.session(database=NEO4J_DATABASE) as session:
      # Fetching the Chunk nodes
      result = session.run("""
        MATCH (c:Chunk) RETURN c.text as text , c.chunkID as ID
      """)


      for chunk in result:
        
        text = chunk.get('text')
        ID = chunk.get('ID')

        print(f"Embedding chunk with ID: {ID}")
        
        if text is not None:
          if embed_model_type == 'vertexai':
            text_embedding = model.embed_query(text)
          elif embed_model_type == None:
            text_embedding = model.encode(text)
            
          chunk_with_embeddings.append(
              {
                  'ID' : ID,
                  'text': text,
                  'text_embedding':text_embedding
              }
          )


        if (len(chunk_with_embeddings) == data_dump_batch_size):
          run_batch(driver,chunk_with_embeddings,batch_count)
          chunk_with_embeddings = []
          batch_count += 1

      # This runs the remaning chunks in the last batch that is smaller than batch size, if any 
      if (len(chunk_with_embeddings) > 0):
        run_batch(driver,chunk_with_embeddings,batch_count)

      records,_,_ = driver.execute_query("""
      MATCH (c:Chunk WHERE c.text_embedding IS NOT NULL)
      RETURN count(*) AS chunk_count_with_embedding, size(c.text_embedding) as embedding_size
      """, database_= NEO4J_DATABASE)

      print(f"""
      Embeddings generated and attached to nodes.
      Chunk nodes with embeddings: {records[0].get('chunk_count_with_embedding')}.
      Embedding size: {records[0].get('embedding_size')}.
          """)

      return True

def clear_graph():
  """
  This method will clear teh existing graph

  """
  KnowledgeGraph =  Neo4jGraph(
                          url=NEO4J_URI,
                          username=NEO4J_USERNAME,
                          password=NEO4J_PASSWORD,
                          database=NEO4J_DATABASE,
                          )
                          
  return KnowledgeGraph.query("""
                      MATCH (n)
                      OPTIONAL MATCH (n)-[r]-()
                      //WITH n,r LIMIT 50000
                      DELETE n,r
                      RETURN count(n) as deletedNodesCount
                      """)
  
def get_all_chunks():
  """
  This method will return the total chunk count in KG

  """
  KnowledgeGraph =  Neo4jGraph(
                          url=NEO4J_URI,
                          username=NEO4J_USERNAME,
                          password=NEO4J_PASSWORD,
                          database=NEO4J_DATABASE,
                          )
                          
  return KnowledgeGraph.query("""
                      MATCH (n)
                      RETURN count(n) as totalNodesCount
                      """) 
