import bs4
import os
from dotenv import load_dotenv
load_dotenv()
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

os.environ['OPENAI_API_KEY'] = os.getenv("MY_API_KEY")
# 1 단계 : 나무위키 내용을 로드하고, 청크로 나누고, 인덱싱
url = "https://namu.wiki/w/%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B8(%EC%9D%B8%EA%B3%B5%EC%8B%A0%EA%B2%BD%EB%A7%9D)"
loader = WebBaseLoader(
    web_paths=(url,),
    requests_kwargs={
        "headers": {
            "User-Agent": "Mozilla/5.0"
        }
    }
)
docs = loader.load()
docs
# 2 단계 : 문서 분할(Split Documents)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)
splits
# 3 단계 : 벡터스토어를 생성합니다.
vectorstore = FAISS.from_documents(splits, embedding=OpenAIEmbeddings())
# 4 단계 : 검색(Search) - 나무위키에 포함되어 있는 정보를 검색하고 생성
retriever = vectorstore.as_retriever()
# 5 단계 : 프롬프트를 생성합니다.
prompt = hub.pull("rlm/rag-prompt").partial(instructions="다음 내용을 요약하세요:")
# 6 단계 : 모델(LLM) 을 생성합니다.
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
def format_docs(docs):
# 검색한 문서 결과를 하나의 문단으로 합침
    return "\n\n".join(doc.page_content for doc in docs)
# 7 단계 : 체인 생성(Create Chain)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
# 8 단계 : 체인 실행(Run Chain)# 문서에 대한 질의를 입력하고, 답변을 출력
question = "Transformer에 대해 설명해줘"
summary = rag_chain.invoke(question)
# 결과 출력
print("===" * 30)
print(f"[URL] {url}")
print(f"[Summary]\n{summary}")
