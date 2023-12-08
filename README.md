# Using LangChain and LLMs to provide prompt context for better results from (GPT models)👌
Feed values, pdfs , to create agricultural recommendations using open ai chat model

#OBJECTIVE
To pass farmer based data and custom recommendations that are context bound to the farmer's soil
Supply with 300 + pages pdf on soil information and best practices to make recommendations a lot more legitimate
Return data in web friendly format, for custom display, return recommendations in JSON format

#METHOD
Using langchain technology to access OpenAI'
Use langchain to create prompts
Use langchain to load, vectorise pdfs 
Feed pdfs as context
extend prompts with values that corresponds to soil figures
get recommendation as question
