import re

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser


def clean_transcript(transcript_text: str) -> str:
    """Clean timestamps and unnecessary whitespace."""

    transcript = re.sub(
        r"(?m)^\s*\d{1,2}:\d{2}(?::\d{2})?\s*$",
        "",
        transcript_text,
    )

    transcript = " ".join(transcript.split())

    return transcript.strip()


def format_docs(retrieved_docs):
    """Combine retrieved transcript chunks into context."""

    return "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )


def build_rag_chain(transcript_text: str, api_key: str):
    """Create the transcript-based RAG question-answering chain."""

    # 1. Clean the transcript
    transcript = clean_transcript(transcript_text)

    if not transcript:
        raise ValueError("The uploaded transcript is empty.")

    # 2. Split the transcript into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=150,
    )

    chunks = splitter.create_documents([transcript])

    # 3. Generate embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
    )

    # 4. Store the embeddings in FAISS
    vector_store = FAISS.from_documents(
        chunks,
        embeddings,
    )

    # 5. Create the retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    # 6. Initialize the language model
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key,
    )

    # 7. Create the prompt template
    prompt = PromptTemplate(
        template="""
You are a helpful assistant answering questions
about a YouTube video transcript.

Answer ONLY using the provided transcript context.
Do not invent information.

If the context is insufficient, say:
"I couldn't find enough information about that in the transcript."

Transcript context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"],
    )

    # 8. Create the parallel chain
    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )

    # 9. Connect retrieval, prompt, LLM, and parser
    main_chain = (
        parallel_chain
        | prompt
        | llm
        | StrOutputParser()
    )

    return main_chain, len(chunks)