from pinecone import Pinecone
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

# import nltk
# nltk.download('punkt_tab')

pinecone_api_key="e6d17400-b5c0-4dca-8015-5f4c59ffaf80"

index_name="hybrid-database"
pc=Pinecone(api_key=pinecone_api_key)
index=pc.Index(index_name)


embeddings=HuggingFaceEmbeddings(model_name="ggrn/e5-small-v2")

bm25_encoder=BM25Encoder().default()

retriever=PineconeHybridSearchRetriever(embeddings=embeddings,sparse_encoder=bm25_encoder,index=index,top_k=10)

llm=ChatGroq(
    # model="llama-3.2-90b-text-preview",
    # model="llama-3.1-70b-versatile",
    model="gemma2-9b-it",
    api_key="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",
    temperature=0)


prompt_query = PromptTemplate.from_template(
    """
    ### PROVIDED PREVIOUS CONVERSATION IN JSON FORMAT (QUERY:ANSWER):
    {conversation}
    
    ### USER QUERY:
    {QUERY}
    
    ### INSTRUCTIONS:
    Given the user's previous conversation in JSON format, do the following:
    - If the question is not clear or not specific, use the conversation JSON to rewrite it into a more specific and focused query. Refer to the most recent conversation (located at the top of the JSON).
    - If the query is already specific, return the same query as the response.

    ### RESPONSE FORMAT:
    - Provide only the revised query or the original query, as applicable.
    - Do not add any additional information not explicitly mentioned in the provided content.
    """
)


prompt_ans = PromptTemplate.from_template(
    """
    ### PROVIDED CONTENT:
    {content_section}

    ### INSTRUCTIONS:
    You are an intelligent assistant, and your task is to provide an accurate response to the query below **using only the content provided**.
    Do not add any information not explicitly mentioned in the provided content.
    If the answer cannot be found in the content, respond with "The answer is not available in the provided content."
    Do not use preamptions!!!

    ### QUERY:
    {query}

    ### RESPONSE:
    """
)

def gen_ans(contents,query):
    ans_chain=prompt_ans | llm
    ans=ans_chain.invoke(input={"content_section":contents,"query":query})
    # print(ans.content)
    return ans.content

def new_query(query,conversation):
   query_chain=prompt_query | llm
   question=query_chain.invoke(input={"QUERY":query,"conversation":conversation})
   return question.content

def query(question,conversation):
  new_input=new_query(question,conversation)
  res=retriever.invoke(new_input)
  contents=[]
  matches=res
  for element in matches:
    content=element.page_content
    contents.append(content)  
  ans=gen_ans(contents,new_input)
  return ans

def upload(text_chunks):
   retriever.add_texts(texts=text_chunks)