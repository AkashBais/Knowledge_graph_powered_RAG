DATA_FOLDER_PATH = "./RAG/Test_RAG_ChatBot/data"
MODEL_FOLDER_PATH = "./RAG/Test_RAG_ChatBot/models"
DOT_ENV_PATH = "/content/RAG/Test_RAG_ChatBot/keys.env"
PROCESSED_FILE_PATH = '/content/RAG/KG_for_RAG/data/processed'

DATA_FILE_NAME = ["GAN.pdf"]
SEPARATORS = ["\n\n", "\n", ".", ""]
NER_MODEL = "blaze999/Medical-NER"
EMBEDDING_MODEL_NAME = "thenlper/gte-large"
READER_MODEL_NAME ="microsoft/Phi-3.5-mini-instruct" 

GEMINI_MODEL =   "gemini-1.5-pro"
READER_LLM_GIMINI = 'Gemini 1.5 Pro'

PROMPT_WEB_SEARCH_FORMAT = """
Please provide a comprehensive summary about the search term {} to get a better and broader understanding of the term.
The summary should include the following points
1 Background: Provide a general overview and context about {}.
2 Brief History: Discuss the origin and evolution of {} over time.
3 Market Landscape: Analyze the current market scenario, trends, and dynamics related to {}.
4 Competitors: Identify and describe the main competitors in the {} space.
5.Future scope: Analyze the plausible future market scenario, trends, and dynamics related to {}.
6.Implications: Analyze the implications of {} in tearms of the current market landscape"""   

PROMPT_GENRATE_WEB_QUERY_FORMAT ="""
Given the following summary, strictly identify the most relevent 3 search terms that are diverse and are best to search over the internet to explore the topic broadly:

summary:{}

The search terms should cover different aspects of the topic to ensure a comprehensive understanding.
Strictly return the search terms and abslutely nothing else in the responce.Don't return any metadata or discription or explanation.
Sample genration for search term 'Novartis' looks like this 'Novartis market landscape'\n'Novartis competitors'\n'Novartis history and mergers'. 
\n is the seprators between genrated search terms
"""

PROMPT_GENRATE_VECTOR_DB_QUERY_FORMAT ="""
Given the following query, identify the top 3 addational diverse queries that can also be searched in the vector database to explore the topic broadly:

query:{}

The addational queries should cover different aspects of the topic to ensure a comprehensive understanding.
Strictly return the search terms and abslutely nothing else in the responce.No need to return any metadata or discription or explanation.
Sample genration for queries 'Describe Novartis' looks like this 'Novartis market landscape'\n'Novartis competitors'\n'Novartis history and mergers'. 
\n is the seprators between search terms
"""

PROMPT_IN_CHAT_FORMAT = [
    {
        "role": "system",
        "content": """Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.
If the answer cannot be deduced from the context, do not give an answer.""",
    },
    {
        "role": "user",
        "content": """ 
Use the following Context to answer the question.
Context:{context}
---------
Here is the question you need to answer.
Question: {question}
""",
    },
]


GIMINI_RAG_PROMPT = """You are a helpful and informative bot that answers questions using text from the reference passage included below. \
Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
Respond only to the question asked, response should be concise and relevant to the question.\
Strike a friendly and converstional tone. \
Provide the number of the source document when relevant.
If the answer cannot be deduced from the context, do not give an answer..
QUESTION: '{query}'
CONTEXT: '{context}'

ANSWER:
"""


 
MERGE_CHUNK_NODE_QUERY = '''
    MERGE (MergedChunks:Chunk {chunkID: $chunkParam.chunkId})
      ON CREATE SET 
        MergedChunks.text = $chunkParam.text,
        MergedChunks.chunk_seq_id = $chunkParam.chunk_seq_id,
        MergedChunks.section = $chunkParam.section,
        MergedChunks.document_source = $chunkParam.document_source,
        MergedChunks.document_date = $chunkParam.document_date,
        MergedChunks.document_name = $chunkParam.document_name,
        MergedChunks.document_brand = $chunkParam.document_brand,
        MergedChunks.documentId = $chunkParam.documentId
    Return MergedChunks
    '''

ADD_UNIQUE_CHUNK_CONSTRANT_QUERY = """
CREATE CONSTRAINT unique_chunk IF NOT EXISTS
FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE
"""

ADD_VECTOR_INDEX_QUERY = """
CREATE VECTOR INDEX `chunk_text` IF NOT EXISTS
FOR (c:Chunk) ON (c.text_embedding)
OPTIONS {indexConfig : {
 `vector.dimensions`: $vector_dimensions,
 `vector.similarity_function`: 'cosine'
}}
"""