from langchain_community.document_loaders import GoogleDriveLoader

import os

doc_id = input("Google Document id: ")
loader = GoogleDriveLoader(document_ids=[doc_id],
                          credentials_path="./client_secret_gdocs_conterr.json")
docs = loader.load()
print('loaded file')

from langchain_openai import OpenAI

from langchain.chains.question_answering import load_qa_chain
llm = OpenAI(openai_api_key=os.environ['OPENAI_API_KEY'])
chain = load_qa_chain(llm)
print('loaded chain. now you can start your queries')
print('''
    example queries:
      list the assertions in this text
      list the contradictions within this text
      list the error scenarios that may occur when a software solution implements these requirements
''')

while(True):
    query = input('Ask away: ')
    response = chain.invoke({'input_documents': docs, 'question': query})
    print(response['output_text'])
