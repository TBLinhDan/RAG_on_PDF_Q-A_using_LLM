**I. Sử dụng thư viện LangChain xây dựng một chương trình RAG với các bước xử lý cơ bản.:**  
Thư viện LangChain là một thư viện được thiết kế chuyên biệt cho việc xây dựng Ứng dụng về các mô hình ngôn ngữ lớn, Large Language Models (LLMs).  
Trong LLMs, Retrieval Augmented Generation (RAG) là một kỹ thuật, tích hợp nội dung truy vấn được từ một nguồn tài liệu nào đó để trả lời cho một câu hỏi đầu vào.

**1. Cài đặt các gói thư viện cần thiết:**
```
!pip install -q transformers==4.41.2
!pip install -q bitsandbytes==0.43.1
!pip install -q accelerate==0.31.0
!pip install -q langchain==0.2.5
!pip install -q langchainhub==0.1.20
!pip install -q langchain-chroma==0.1.1
!pip install -q langchain-community==0.2.5
!pip install -q langchain_huggingface==0.0.3
!pip install -q python-dotenv==1.0.1
!pip install -q pypdf==4.2.0
!pip install -q numpy==1.24.4
```

**2. Xây dựng Vector Database:**  

Với dữ liệu nguồn là một file pdf, thực hiện đưa các nội dung trong file này vào cơ sở dữ liệu, với các bước:

***load file.pdf --> text Splitter --> Vectorrization --> Vector database***

* Import các thư viện cần thiết:
```
import torch

from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline

from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.chains import ConversationalRetrievalChain

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
```
* sử dụng class PyPDFLoader để đọc file pdf đầu vào:

```
Loader = PyPDFLoader
FILE_PATH = "./AIO-2024-All-Materials.pdf"
loader = Loader(FILE_PATH)
documents = loader.load()
```

* Khởi tạo bộ tách văn bản (text splitter), tách file.pdf ra thành các đoạn văn bản nhỏ, mỗi đoạn văn bản nhỏ được coi như là một tài liệu trong cơ sở dữ liệu.

```
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap =100)
docs = text_splitter.split_documents(documents)

print ("Number of sub-documents:", len(docs)) # số lượng tài liệu 
print (docs[0])		                       # nội dung tài liệu đầu tiên
```

*  Khởi tạo instance vectorization, chuyển đổi các văn bản thành các vector, giúp dễ dàng và chính xác trong thục hiện truy vấn.

```
embedding = HuggingFaceEmbeddings() # chuyển đổi văn bản thành vector
```

* Khởi tạo vector database từ text_splitter các đoạn văn bản nhỏ đã được Vectorrization qua embedding

```
vector_db = Chroma.from_documents(documents=docs, embedding=embedding)
retriever = vector_db.as_retriever()
```

* Thử thực hiện truy vấn với một đoạn văn bản bất kì.  
(Kết quả sẽ trả về cho ta danh sách các tài liệu có liên quan đến câu hỏi đầu vào)

```
result = retriever.invoke("What is YOLO?")
print("Number of relevant documents:", len(result))
```

**3. Khởi tạo mô hình ngôn ngữ lớn, sử dụng mô hình Vicuna**  
Vicuna là mô hình LLM nguồn mở, có hiệu suất rất ổn, và phản hồi tốt với tiếng Việt

* Khai báo một số cài đặt cần thiết cho mô hình:
```
nf4_config = BitsAndBytesConfig(
	                load_in_4bit=True,
	                bnb_4bit_quant_type="nf4",
	                bnb_4bit_use_double_quant=True,
	                bnb_4bit_compute_dtype=torch.bfloat16
	                )
```

* Khởi tạo mô hình và tokenizer:
```
model = AutoModelForCausalLM.from_pretrained(
	                    MODEL_NAME,
	                    quantization_config=nf4_config,
	                    low_cpu_mem_usage = True
	                    )
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
```

* Tích hợp tokenizer và model thành một pipeline để tiện sử dụng:
```
model_pipeline = pipeline(
	                "text-generation",
	                model=model,
	                tokenizer=tokenizer,
	                max_new_tokens=512,
	                pad_token_id=tokenizer.eos_token_id,
	                device_map="auto"
	                )
llm = HuggingFacePipeline(pipeline = model_pipeline,)

```

**4. Kết hợp vector database, retriever và mô hình Vicuna để hoàn thành chương trình RAG, có khả năng hỏi đáp các nội dung trong một file pdf.**

```
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
	{"context": retriever | format_docs, "question": RunnablePassthrough
	()}
	| prompt
	| llm
	| StrOutputParser()
)
USER_QUESTION = "YOLOv10 là gì?"
output = rag_chain.invoke(USER_QUESTION)
answer = output.split('Answer:')[1].strip()
print(answer)
```

**II. Tích hợp chương trình RAG (sử dụng thư viện LangChain có thể trả về output đúng như dự định) với một giao diện chat (sử dụng thư viện Chainlit) để có được một ứng dụng chat hoàn chỉnh.**

**1. Tải các gói thư viện:**
```
! pip install -q transformers==4.41.2
! pip install -q bitsandbytes==0.43.1
! pip install -q accelerate==0.31.0
! pip install -q langchain==0.2.5
! pip install -q langchainhub==0.1.20
! pip install -q langchain-chroma==0.1.1
! pip install -q langchain-community==0.2.5
! pip install -q langchain-openai==0.1.9
! pip install -q langchain_huggingface==0.0.3
! pip install -q chainlit==1.1.304
! pip install -q python-dotenv==1.0.1
! pip install -q pypdf==4.2.0
! npm install -g localtunnel
! pip install -q numpy==1.24.4
```

**2. Import các gói thư viện cần thiết:**
```
import chainlit as cl
import torch

from chainlit.types import AskFileResponse

from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface.llms import HuggingFacePipeline

from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import ConversationalRetrievalChain

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
```

**3. Khởi tạo bộ tách văn bản (text splitter), tách file.pdf ra thành các đoạn văn bản nhỏ, mỗi đoạn văn bản nhỏ được coi như là một tài liệu trong cơ sở dữ liệu.**
```
text_splitter = RecursiveCharacterTextSplitte(chunk_size=1000, chunk_overlap =100)
```

* Khởi tạo instance vectorization, chuyển đổi các văn bản thành các vector, giúp dễ dàng và chính xác trong thục hiện truy vấn.

```
embedding = HuggingFaceEmbeddings() # chuyển đổi văn bản thành vector
```

**4. Xây dựng hàm xử lý file input đầu vào (đọc và tách văn bản):**
```
def process_file(file: AskFileResponse):
    if file.type == "text/plain":
	    Loader = TextLoader
    elif file.type == "application/pdf":
	    Loader = PyPDFLoader

    loader = Loader(file.path)
    documents = loader.load()
    docs = text_splitter.split_documents(documents)
    for i, doc in enumerate(docs):
	    doc.metadata["source"] = f"source_{i}"
    return docs
```

**5. Xây dựng hàm khởi tạo Chroma database (khởi tạo vector database):**  
(Tại line 2, gọi hàm process_file() để xử lý file input và trả về các tài liệu nhỏ (docs). Sau đó, khởi tạo Chroma vector database bằng cách gọi Chroma.from_documents() và truyền vào docs cũng như embedding đã khởi tạo trước đó).

```
def get_vector_db(file: AskFileResponse):
   docs = process_file(file)
   cl.user_session.set("docs", docs)
   vector_db = Chroma.from_documents(documents=docs, embedding=embedding)
   return vector_db
```

**6. Khởi tạo mô hình ngôn ngữ lớn:**
```
def get_huggingface_llm(model_name: str = "lmsys/vicuna-7b-v1.5",
			max_new_token: int = 512):
    nf4_config = BitsAndBytesConfig(
	    load_in_4bit=True,
	    bnb_4bit_quant_type="nf4",
	    bnb_4bit_use_double_quant=True,
	    bnb_4bit_compute_dtype=torch.bfloat16
	    )
    model = AutoModelForCausalLM.from_pretrained(
	    model_name,
	    quantization_config=nf4_config,
	    low_cpu_mem_usage=True
	    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_pipeline = pipeline(
	    "text-generation",
	    model=model,
	    tokenizer=tokenizer,
	    max_new_tokens=max_new_token,
	    pad_token_id=tokenizer.eos_token_id,
	    device_map="auto"
	    )

    llm = HuggingFacePipeline(pipeline = model_pipeline,)
    return llm

LLM = get_huggingface_llm()
```

**7. Khởi tạo welcome message:**
```
welcome_message = """ Welcome to the PDF QA! To get started:
                    1. Upload a PDF or text file
                    2. Ask a question about the file
                  """
```

**8. Khởi tạo hàm on_chat_start:**
```
!pip install chainlit
import chainlit as cl

@cl.on_chat_start
async def on_chat_start():
    files = None
    while files is None:
	    files = await cl.AskFileMessage(
	        content=welcome_message,
    	    accept=["text/plain","application/pdf"],
	        max_size_mb=20,
	        timeout=180,
	    ).send()
    file = files[0]

    msg = cl.Message(content=f"Processing'{ file.name}'...", disable_feedback=True)
    await msg.send()

    vector_db = await cl.make_async(get_vector_db)(file)

    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
     	output_key="answer",
     	chat_memory=message_history,
     	return_messages=True,
    )
    retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={'k': 3})
    chain = ConversationalRetrievalChain.from_llm(
	    llm =LLM,
	    chain_type="stuff",
	    retriever=retriever,
	    memory=memory,
	    return_source_documents=True
            )

    msg.content = f"'{ file.name}' processed.You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)
```

**9. Khởi tạo hàm on_message:**
```
@cl.on_message
    async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]
    text_elements = []

    if source_documents:
	    for source_idx, source_doc in enumerate(source_documents):
	        source_name = f"source_{source_idx}"
	        text_elements.append(
		        cl.Text(content=source_doc.page_content,
			    name=source_name)
	            )
	    source_names = [text_el.name for text_el in text_elements]

	    if source_names:
	        answer += f"\ nSources: {', '.join(source_names)}"
	    else :
	        answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()
```

**10. Chạy chainlit app:**
```
!chainlit run app.py --host 0.0.0.0 --port 8000 &>/content/logs.txt &
```

**11. Expose localhost thành public host bằng localtunnel:**
```
import urllib
print("Password/Enpoint IP for localtunnel is:",urllib.request.urlopen('
        https://ipv4.icanhazip.com').read().decode('utf8').strip("\n"))

!lt --port 8000 -- subdomain aivn-simple-rag
