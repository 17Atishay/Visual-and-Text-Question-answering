import streamlit as st
from dotenv import load_dotenv
import os
load_dotenv()
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import Ollama
import pickle
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer
import fitz  # PyMuPDF
from PIL import Image
import io
from ollama_llava_image_qa import query_llava_ollama
from gemini_utils import query_gemini

# sidebar contents
with st.sidebar:
    st.title("Visual and Text Question answering")
    # st.markdown(''' 
    # ## About
    # This App is an LLM-powered Chatbot built using Ollama (local models).
    # ''')
    st.write("\n" * 2)
    # --- Chat History in Sidebar ---
    with st.expander("Chat History", expanded=False):
        if 'chat_history' in st.session_state and st.session_state['chat_history']:
            for i, (q, a) in enumerate(st.session_state['chat_history']):
                st.markdown(f"**Q{i+1}:** {q}")
                st.markdown(f"**A{i+1}:** {a}")
                st.markdown("---")
        else:
            st.info("No chat history yet.")
    st.write("\n" * 2)
    st.write("Made by Atishay Jain..")

def main():
    # st.header("Chat with pdf-bot (Text & Image)")
    st.header("What can i Help With?")

    pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

    if pdf:
        # --- TEXT EXTRACTION (unchanged) ---
        import tempfile
        tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp_pdf.write(pdf.read())
        tmp_pdf.flush()
        pdf_path = tmp_pdf.name

        # Extract text
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()

        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            length_function=lambda x: len(tokenizer.encode(x))
        )
        chunks = text_splitter.split_text(text=text)
        formatted_chunks = [Document(page_content=t) for t in chunks]
        # st.write("Text chunks:", formatted_chunks)  # Hide text chunks from UI

        if not formatted_chunks:
            st.error("No extractable text found in this PDF. Please upload a PDF with selectable text, or use the image Q&A feature for images.")
            return

        # --- Summarization Feature ---
        st.markdown("---")
        st.subheader("Summarize PDF")
        if st.button("Summarize PDF"):
            llm = Ollama(model="llama2")
            summary_prompt = PromptTemplate(
                input_variables=["context"],
                template="""
                Summarize the following document in a concise, clear, and informative way for a general audience.\n\nDocument:\n{context}\n\nSummary:"""
            )
            chain = load_qa_chain(llm, chain_type="stuff", prompt=summary_prompt)
            try:
                summary = chain.run(input_documents=formatted_chunks)
                st.success("Summary:")
                st.write(summary)
            except Exception as e:
                st.error(f"Summarization failed: {e}")
        st.markdown("---")

        store_name = pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vector_db = pickle.load(f)
            st.write("Embeddings loaded from file")
        else:
            embeddings = HuggingFaceEmbeddings()
            vector_db = FAISS.from_documents(formatted_chunks, embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vector_db, f)
            st.write("Embeddings computation completed.")

        # --- IMAGE EXTRACTION ---
        st.subheader("Extracted Images:")
        image_bytes_list = []
        image_display_list = []
        image_page_map = []
        for page_num, page in enumerate(doc):
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                image_bytes_list.append(img_bytes)
                image_display_list.append(Image.open(io.BytesIO(img_bytes)))
                image_page_map.append((page_num, img_index))
        if image_display_list:
            st.image(image_display_list, caption=[f"Page {p+1} Image {i+1}" for p,i in image_page_map], width=200)
            selected_idx = st.multiselect(
                "Select image(s) to ask about:",
                options=list(range(len(image_display_list))),
                format_func=lambda i: f"Page {image_page_map[i][0]+1} Image {image_page_map[i][1]+1}"
            )
        else:
            st.info("No images found in PDF.")
            selected_idx = []

        # --- Contextual Chat History ---
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

        # --- Auto-image checkbox above chat input ---
        auto_image_match = st.checkbox("Auto-detect relevant images for question", value=True)
        use_gemini = st.checkbox("Include latest web facts (Google Gemini)")
        # --- Chat input always at the bottom ---
        query = st.chat_input("Ask questions about your pdf data (text or images):")

        gemini_api_key = None
        if use_gemini:
            # Only use .env (os.environ) for Gemini API key
            gemini_api_key = os.environ.get("GEMINI_API_KEY", None)

        if query:
            # --- TEXT QA ---
            docs = vector_db.similarity_search(query, k=3)
            llm = Ollama(model="llama2")
            qa_prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="""
                Given the following document extract, answer the question clearly.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer in a well-structured manner with relevant details.
                """
            )
            chain = load_qa_chain(llm, chain_type="stuff", prompt=qa_prompt)
            response_text = chain.run(input_documents=docs, question=query)
            st.markdown("**Text-based Answer:**")
            st.write(response_text)
            # Add to chat history
            st.session_state['chat_history'].append((query, response_text))

            # --- Gemini PDF-context-aware Answer ---
            if use_gemini:
                if not gemini_api_key:
                    st.warning("Gemini API Key not found. Please set GEMINI_API_KEY in your .env file.")
                else:
                    # Build a prompt that gives Gemini both the PDF context and the question
                    pdf_context = "\n\n".join([doc.page_content for doc in docs])
                    gemini_prompt = f"""
You are an expert assistant. Here is a document extract from a PDF:
{pdf_context}

The user asks: {query}

Answer using the PDF context above. If you need to add any recent facts from the web, clearly indicate those as well.
"""
                    with st.spinner("Getting a PDF-aware answer from Gemini..."):
                        gemini_answer = query_gemini(gemini_prompt, gemini_api_key)
                        if gemini_answer:
                            st.markdown("**Gemini Answer (PDF-aware + Web):**")
                            st.write(gemini_answer)

            # --- IMAGE QA ---
            relevant_indices = set(selected_idx)
            if auto_image_match and not selected_idx:
                # Simple heuristic: if question contains 'image', 'figure', 'diagram', or 'photo', use all images
                if any(word in query.lower() for word in ["image", "figure", "diagram", "photo"]):
                    relevant_indices = set(range(len(image_bytes_list)))
            if relevant_indices:
                st.markdown("**Image-based Answer(s) (Llava):**")
                for idx in relevant_indices:
                    img_bytes = image_bytes_list[idx]
                    try:
                        answer = query_llava_ollama(img_bytes, query)
                        st.image(image_display_list[idx], caption=f"Llava Answer for Page {image_page_map[idx][0]+1} Image {image_page_map[idx][1]+1}")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"Error querying Llava: {e}")
            elif len(image_bytes_list) > 0:
                st.info("No images selected or matched for image-based QA.")

if __name__ == '__main__':
    main()
