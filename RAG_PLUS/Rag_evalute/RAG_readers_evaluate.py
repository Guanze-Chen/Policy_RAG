# pip install transformers langchain faiss-gpu dashscope zhipuai datasets sentence-transformers
import os
import time
import json
import datasets
import pandas as pd
from typing import Optional, List, Tuple
from zhipuai import ZhipuAI
from langchain.docstore.document import Document as LangchainDocument
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models.llms import LLM
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import datasets
from http import HTTPStatus
import dashscope
from dashscope import Generation



def split_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name: str,
) -> List[LangchainDocument]:
    """
    Split documents into chunks of size `chunk_size` characters and return a list of documents.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", "。", " ", ""],
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique



def load_embeddings(
    langchain_docs: List[LangchainDocument],
    chunk_size: int,
    embedding_model_name: Optional[str] = "moka-ai/m3e-base",
) -> FAISS:
    """
    Creates a FAISS index from the given embedding model and documents. Loads the index directly if it already exists.

    Args:
        langchain_docs: list of documents
        chunk_size: size of the chunks to split the documents into
        embedding_model_name: name of the embedding model to use

    Returns:
        FAISS index
    """
    # load embedding_model
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},  # set True to compute cosine similarity
    )

    # Check if embeddings already exist on disk
    index_name = f"index_chunk_{chunk_size}_embeddings_{embedding_model_name.replace('/', '~')}"
    index_folder_path = ".\\data\\indexes\\{}\\".format(index_name)
    if os.path.isdir(index_folder_path):
        return FAISS.load_local(
            index_folder_path,
            embedding_model,
            distance_strategy=DistanceStrategy.COSINE,
            allow_dangerous_deserialization=True 
        )

    else:
        print("Index not found, generating it...")
        docs_processed = split_documents(
            chunk_size,
            langchain_docs,
            embedding_model_name,
        )
        knowledge_index = FAISS.from_documents(
            docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
        )
        knowledge_index.save_local(index_folder_path)
        return knowledge_index




def answer_with_rag(
    question: str,
    llm,
    model_name: str,
    knowledge_index: VectorStore,
    reranker_name: str='BAAI/bge-reranker-base',
    num_retrieved_docs: int = 5,
    num_docs_final: int = 3
) -> Tuple[str, List[LangchainDocument]]:
    """Answer a question using RAG with the given knowledge index."""
    # Gather documents with retriever
    relevant_docs_with_metadata = knowledge_index.similarity_search(query=question, k=num_retrieved_docs)
    relevant_docs = [doc.page_content for doc in relevant_docs_with_metadata]  # keep only the text
    relevant_link = [doc.metadata['url'] for doc in relevant_docs_with_metadata]
    # Optionally rerank results
    if reranker_name:
        tokenizer = AutoTokenizer.from_pretrained(reranker_name)
        reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_name)
        pairs = []
        for doc in relevant_docs:
            pairs.append([question, doc])
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        with torch.no_grad():
            inputs = {key: inputs[key] for key in inputs.keys()}
            scores = reranker_model(**inputs, return_dict=True).logits.view(-1, ).float()
            top_k_indices = scores.topk(num_docs_final).indices
            relevant_docs = [relevant_docs[idx] for idx in top_k_indices]

    relevant_docs = relevant_docs[:num_docs_final]

    # Build the final prompt
    context = "\nExtracted documents:\n"
    context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])

    final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context, link=relevant_link[0])

    # Redact an answer
    answer = llm(final_prompt, model_name)
    answer = f'Answer: {answer}\n related-link:{relevant_link[0]}'

    return answer, relevant_docs


def call_qwen(prompt, model_name):
    messages = [
        {'role': 'user', 'content': prompt}]
    response = dashscope.Generation.call(
        model_name,
        messages=messages,
        # set the random seed, optional, default to 1234 if not set
        result_format='message',  # set the result to be "message" format.
    )
    if response.status_code == HTTPStatus.OK:
        return response['output']['choices'][0]['message']['content']
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))

def call_with_stream(prompt, model_name):
    messages = [
        {'role': 'user', 'content': prompt}]
    responses = Generation.call(
        model=model_name,
        messages=messages,
        result_format='message',  # set the result to be "message" format.
        stream=True,
        incremental_output=True  # get streaming output incrementally
    )
    full_content = ''  # with incrementally we need to merge output.
    for response in responses:
        if response.status_code == HTTPStatus.OK:
            content = response.output.choices[0]['message']['content']
            full_content += content
        else:
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))
    return full_content

def call_Baichuan2(prompt, model_name):
    rsp = dashscope.Generation.call(model='baichuan2-7b-chat-v1',
                                    prompt=prompt)
    if rsp.status_code == HTTPStatus.OK:
        return rsp['output']['text']
    else:
        print('Failed, status_code: %s, code: %s, message: %s' %
              (rsp.status_code, rsp.code, rsp.message))



def run_rag_tests(
    eval_dataset: datasets.Dataset,
    llm: LLM,
    model_name: str,
    knowledge_index: VectorStore,
    output_file: str,
    reranker_name: str='BAAI/bge-reranker-base',
    verbose: Optional[bool] = True,
    test_settings: Optional[str] = None,  # To document the test settings used
):
    """Runs RAG tests on the given dataset and saves the results to the given output file."""
    try:  # load previous generations if they exist
        with open(output_file, "r", encoding='utf-8') as f:
            outputs = json.load(f)
    except:
        outputs = []

    for example in tqdm(eval_dataset):
        question = example["question"]
        if question in [output["question"] for output in outputs]:
            continue
        time.sleep(2)
        answer, relevant_docs = answer_with_rag(question, llm, model_name, knowledge_index, reranker_name=reranker_name)
        if verbose:
            print("=======================================================")
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f'True answer: {example["answer"]}')
        result = {
            "question": question,
            "true_answer": example["answer"],
            "source_doc": example["source_url"],
            "generated_answer": answer,
            "retrieved_docs": [doc for doc in relevant_docs],
        }
        if test_settings:
            result["test_settings"] = test_settings
        outputs.append(result)

        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(outputs, f, ensure_ascii=False)


if __name__ == "__main__":
    generated_questions = pd.read_csv("evaluate_qa_pair.csv")
    generated_questions = generated_questions.loc[
        (generated_questions["groundedness_score"] >= 4)
        & (generated_questions["relevance_score"] >= 4)
        & (generated_questions["standalone_score"] >= 4)
    ]

    client = ZhipuAI(api_key="3febca8c763728a65309298b9e20b9d6.6KzxyxJ1Cv4Wn0L1")

    eval_dataset = datasets.Dataset.from_pandas(generated_questions, split='train', preserve_index=False)


    ### try to construct RAG System

    with open('work_policy_data.json', 'r', encoding='utf-8') as json_file:
        raw_data = json.load(json_file)
        data = raw_data['data']

    RAW_KNOWLEDGE_BASE = [
        LangchainDocument(page_content=doc["content"], metadata={"source": doc['metadata']["source"], "url": doc['metadata']["url"]}) for doc in tqdm(data)
    ]

    RAG_PROMPT_TEMPLATE = """
    请你使用上下文中包含的信息，对问题给出全面的答案。
    仅回答所提出的问题，回答应简洁且与问题相关。
    如果无法从上下文中推断出答案，请不要给出答案，输出无法回答。

    下面是提供上下文：
    上下文:{context}

    ---
    这是你需要回答的问题。

    问题: {question}
    """

    if not os.path.exists("./RAG2_output"):
        os.mkdir("./RAG2_output")
    
    embeddings = 'infgrad/stella-large-zh-v2'
    chunk_size = 300
    dashscope.api_key_file_path="./api_key.txt"
    # llm_model_list = ['baichuan2-7b-chat-v1', "yi-6b-chat"]
    llm_model_list = ["qwen-1.8b-chat", "qwen1.5-7b-chat", 'internlm-7b-chat', "yi-6b-chat", 'baichuan2-7b-chat-v1']
    qwen_models = ["qwen-1.8b-chat", "qwen1.5-7b-chat"]
    stream_models = ['internlm-7b-chat', "yi-6b-chat"]
    baichuan_model = ['baichuan2-7b-chat-v1']

    for model_name in llm_model_list:
        if model_name in qwen_models:
            llm = call_qwen
        
        elif model_name in stream_models:
            llm = call_with_stream
        elif model_name in baichuan_model:
            llm = call_Baichuan2
        else:
            print("Error, the model is not in the list")

        settings_name = f"{model_name}_chunk_{chunk_size}_embeddings_{embeddings.replace('/', '~')}"
        output_file_name = f"./RAG2_output/rag_{settings_name}.json"
        print(f"Running evaluation for {settings_name}:")

        print("Loading knowledge base embeddings...")

        knowledge_index = load_embeddings(
            RAW_KNOWLEDGE_BASE,
            chunk_size=chunk_size,
            embedding_model_name=embeddings
            )
        print("Running RAG...")

        
        run_rag_tests(
            eval_dataset=eval_dataset,
            llm=llm,
            model_name=model_name,
            knowledge_index=knowledge_index,
            output_file=output_file_name,
            reranker_name='BAAI/bge-reranker-base',
            verbose=False,
            test_settings=settings_name,
        )






