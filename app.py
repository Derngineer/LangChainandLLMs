#lets create our prompt engineering system in here
import os
os.environ['OPENAI_API_KEY'] = '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'
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate




#Get key values for recommendations
soil_data =['7',(30,40,30), 0.5, 32] #sample values corresponding to pH, soil texture, organic carbon content, nitrogen and so fourth

#THE BELOW FUNCTION GETS RECOMMENDATIONS
def get_soilRecommendations(soil_data):
    soil_data = soil_data

    soil_values =f' A helpful answer includes incomporation the following values of the farm,  {soil_data[0]} pH value on standard pH scale, {soil_data[1]} loam sand and clay as percentages respectively,{soil_data[2]} organic carbon content as ppm\
                        ,{soil_data[3]}  nitrogen content ppm' # will appendd this to buff-up the template for prompt
    print('working _1')
    template = """ Use the following pieces of context to answer the question at the end. Make it as detailed\
    as possible \
    You are an agriculturalist with vast knowledge on soil data\
        You understand pH, texture, Nitrogen, Carbon, Potassium, Phosphorus and so much\
            you also undderstand how these variables affect output for farmers\
                You have to give professional well structured recommendations to the farmer\
                    First you explain the meaning of the available data given implications\
                        then double down on actions that may be taken to optimize the use of the soil\
                            for better output, sustainability and geometrical growth,\
                            You may include  valid crop choice reccomendations at the end.\
                            The format of your response should be JSON format\
                            the responses are to be long, as detailed as possible\
                            if they're are important actions in you json file for that particular value include action section\
                            if there data values are critical level either too low or too low also include a WARNING section\
                            your json output should be well formatted consistently , with so much detail\
                            Make use of the context provided with the additional embedded documents while creating you response\
                            Each variable/observation should have a related recommendation close to it\
                            return your answer in a JSON format\
                            as  follows in tripple delimeted text\
                                    {context}\
                                        Question: {question}\
                                        """
    template += soil_values #concatenation to join the template

    input_variables = ['context', 'question']
    chain_prompt = PromptTemplate(input_variables = input_variables, template = template)
    llm = ChatOpenAI(model_name = 'gpt-3.5-turbo', temperature=0, verbose= True)

    #Feeding custom data here, vectorise it too
    loader = PyPDFLoader('/Users/mac/myproject_4/env/nyanzvi/managerAI/sources/soilsource.pdf')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap =0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(texts, embeddings)

    retriever = db.as_retriever(search_type ='similarity', search_kwargs = {'k':2})
    qa_chain = RetrievalQA.from_chain_type(llm, retriever = retriever, chain_type_kwargs= {'prompt':chain_prompt})
    question = 'What do you recommend?'


    result = qa_chain.run(question)
    return result

print(get_soilRecommendations(soil_data))
