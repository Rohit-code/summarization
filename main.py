import os
from apikey import apikey
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain import OpenAI, PromptTemplate
import glob
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.indexes import VectorstoreIndexCreator

os.environ["OPENAI_API_KEY"]= apikey

llm = OpenAI(temperature=0.3)
def summarize_pdfs_from_folder(pdfs_folder):
    summaries = []
    for pdf_file in glob.glob(pdfs_folder + "/*.pdf"):
        loader= PyPDFLoader(pdf_file)
        docs=loader.load_and_split()
        chain=load_summarize_chain(llm,chain_type="map_reduce")
        summary=chain.run(docs)
        print("summary for: ", pdf_file)
        print(summary)
        print("\n")
        summaries.append(summary)
        
    return summaries
    
def custom_summary(pdf_folder, custom_prompt):
    summaries = []
    for pdf_file in glob.glob(pdf_folder + "/*.pdf"):
        loader = PyPDFLoader(pdf_file)
        docs = loader.load_and_split()
        prompt_template = custom_prompt + """

        {text}

        SUMMARY:"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        chain = load_summarize_chain(llm, chain_type="map_reduce", 
                                    map_prompt=PROMPT, combine_prompt=PROMPT)
        summary_output = chain({"input_documents": docs},return_only_outputs=True)["output_text"]
        summaries.append(summary_output)
        
    return summaries
summaries = summarize_pdfs_from_folder("./pdfs")
print(summaries)


with open("summaries.txt","w") as f:
    for summary in summaries:
        f.write(summary + "\n"*3)

loader= PyPDFDirectoryLoader("./pdfs/")

docs= loader.load()
index= VectorstoreIndexCreator().from_loaders([loader])

query="what is the core idea behind the coOP(context optimization) paper?"
print(index.query(query))