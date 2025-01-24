import streamlit as st
import logging
import time
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from retrieval import hybrid_retriever , qaretrieval
# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    #format='%(asctime)s - %(levelname)s - %(message)s',
    format = '%(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
from model_config import load_gpt4o_mini_model , load_gemini15, load_together_model , load_groq_model
from prompt import QA_TEMPLATE, REWRITE_TEMPLATE
COLLECTION_NAME = "raptor"
BASE_COLLECTION_NAME = "base"
QA_COLLECTION_NAME = "faq_data"
THRESHOLD = 0.85
# Khởi tạo các thành phần global
load_dotenv()
st.set_page_config(page_title="University Assistant", page_icon="🤖")
st.title("University Assistant")
@st.cache_resource
def initialize_db():
    logger.info("Khởi tạo database và FAQ...")
    db = hybrid_retriever(COLLECTION_NAME)
    qa_db = qaretrieval(QA_COLLECTION_NAME)
    return db, qa_db

@st.cache_resource
def initialize_model(selected_model):
    logger.info(f"Khởi tạo model {selected_model}...")
    
    if selected_model == "Together_ai.meta-llama/Llama-3.3-70B-Instruct-Turbo":
        llm = load_together_model()
    elif selected_model == "Groq_llama-3.3-70b-versatile":
        llm = load_groq_model()
    # elif selected_model == "GPT4o Mini":
    #     llm = load_gpt4o_mini_model()
    elif selected_model == "Gemini1.5Flash":
        llm = load_gemini15()
    else:
        raise ValueError("Model không hợp lệ!")
    return llm
db, qa_db = initialize_db()

# Thêm giao diện chọn model
selected_model = st.sidebar.selectbox(
    "Chọn model",
    [ "Together_ai.meta-llama/Llama-3.3-70B-Instruct-Turbo", "Groq_llama-3.3-70b-versatile","Gemini1.5Flash"]
    )

# Khởi tạo model dựa trên lựa chọn
llm = initialize_model(selected_model)



def rewrite_question(user_query, formatted_history, llm):  # Thêm tham số `llm`
    logger.info("=== Bắt đầu viết lại câu hỏi ===")
    #logger.info(f"Câu hỏi gốc: {user_query}")
    
    if formatted_history:
        logger.info("Đang sử dụng lịch sử chat để viết lại câu hỏi")
        #logger.info(f"Lịch sử chat đã format: {formatted_history}")
    else:
        logger.info("Không có lịch sử chat trước đó")
    
    rewrite_prompt = ChatPromptTemplate.from_template(REWRITE_TEMPLATE)
    chain = rewrite_prompt | llm | StrOutputParser()  # Sử dụng `llm` truyền vào thay vì toàn cục
    
    logger.info("Đang gửi yêu cầu viết lại câu hỏi tới model...")
    rewritten_query = chain.invoke({
        "chat_history": formatted_history,
        "question": user_query
    })
    logger.info(f"Câu hỏi gốc: {user_query}")
    logger.info(f"Câu hỏi đã được viết lại: {rewritten_query.strip()}")
    logger.info("=== Kết thúc viết lại câu hỏi ===")
    return rewritten_query.strip()


def format_chat_history(chat_history, max_pairs=5):
    formatted_history = []
    qa_pairs = []

    for i in range(len(chat_history)-1):
        if isinstance(chat_history[i], HumanMessage) and isinstance(chat_history[i+1], AIMessage):
            qa_pairs.append({
                "question": chat_history[i].content,
                "answer": chat_history[i+1].content
            })

    recent_pairs = qa_pairs[-max_pairs:] if qa_pairs else []
    
    for pair in recent_pairs:
        formatted_history.append(f"Human: {pair['question']}\nAssistant: {pair['answer']}")

    return "\n\n".join(formatted_history)

def check_faq(user_query, qa_db):  # Loại bỏ giá trị mặc định, `qa_db` là tham số bắt buộc
    logger.info("Truy vấn DB cho FAQ...")
    # Gọi `semantic_qa_search` để tìm kiếm trong collection
    result = qa_db.invoke(user_query)  # Sử dụng `qa_db` truyền vào thay vì toàn cục
    
    if len(result) > 0:
        logger.info(f"Câu hỏi phù hợp tìm thấy trong QADB. Score = {result[0].metadata.get('_score')}")
        return result[0].page_content  # Trả về câu trả lời tốt nhất

    logger.info("Không tìm thấy câu trả lời phù hợp trong QADB.")
    return None


def stream_text(text):
    """Generator function để stream từng từ của văn bản"""
    time.sleep(1)
    # Chuỗi mới được thêm vào cuối văn bản
    new_text =text +" Nếu bạn có câu hỏi nào khác hãy cho tôi biết nhé! Mời bạn tham khảo thêm các thông tin khác tại đây: "
    link = "[Sổ tay sinh viên](https://ctsv.hust.edu.vn/#/so-tay-sv)"
    # Streaming từng từ trong văn bản
    for word in new_text.split():
        yield word + " "
        # Nếu muốn điều chỉnh tốc độ stream, có thể thêm time.sleep()
        time.sleep(0.05)
    yield link
    # Sau khi hoàn thành streaming, hiển thị link
    #st.markdown(link)

def get_response(user_query, chat_history, db, qa_db, llm):  # Thêm `qa_db` và `llm` làm tham số
    logger.info("=== Bắt đầu quá trình tạo câu trả lời ===")

    # Kiểm tra trong FAQ
    answer = check_faq(user_query, qa_db)  # Truyền `qa_db` vào
    if answer:
        return stream_text(answer)
    # Không tìm thấy, gọi đến RAG
    logger.info("Không tìm thấy câu hỏi phù hợp trong FAQ, chuyển đến hệ thống RAG.")
    # Viết lại câu hỏi
    logger.info("=== Bắt đầu format chat history ===")
    formatted_history = format_chat_history(chat_history, max_pairs=5)
    rewritten_query = rewrite_question(user_query, formatted_history, llm)  # Truyền `llm` vào
    if "<spam>" in rewritten_query:
        default_response = "Xin lỗi, tôi chỉ là trợ lý ảo của Đại học và tôi không thể trả lời câu hỏi này. "
        logger.info("Phát hiện câu hỏi không liên quan, trả lời mặc định.")
        return stream_text(default_response)
    # Truy vấn thông tin liên quan  
    logger.info("=== Bắt đầu truy xuất tài liệu ===")
    relevant_docs = db.invoke(rewritten_query)
    logger.info(f"Số lượng tài liệu tìm được: {len(relevant_docs)}")

    logger.info("=== Kết thúc truy xuất tài liệu ===")
    doc_context = "\n\n".join([doc.page_content for doc in relevant_docs])
    logger.info(f"Độ dài context: {len(doc_context)} ký tự")
    logger.info("=== Kết thúc format dữ liệu ===")

    QAprompt = ChatPromptTemplate.from_template(QA_TEMPLATE)
    logger.info("=== Bắt đầu tạo câu trả lời ===")
    logger.info(f"=== full QA_prompt {QAprompt}===")
    
    chain = QAprompt | llm | StrOutputParser()  # Sử dụng `llm` truyền vào thay vì toàn cục
    logger.info("Đang gửi yêu cầu tới model...")
    response = chain.stream({
        "chat_history": formatted_history,
        "documents": doc_context,
        "question": rewritten_query
    })
    logger.info("=== Kết thúc tạo câu trả lời ===")
    logger.info(f"Câu trả lời: {response}")
    return response


###app###
def process_user_input(user_query):
    if not user_query:
        return None
    if len(user_query) > 512:
        st.error("Độ dài câu hỏi quá dài, vui lòng nhập câu hỏi ngắn hơn!")
        return None
    logger.info(f"\n=== Bắt đầu xử lý câu hỏi mới:{user_query} ===\n")
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
    
    with st.chat_message("AI"):
        response = st.write_stream(get_response(user_query, st.session_state.chat_history, db, qa_db, llm))

        logger.info(f"Câu trả lời cho người dùng: {response}===\n")

    st.session_state.chat_history.append(AIMessage(content=response))
    return response
def display_chat_history():
    if "chat_history" not in st.session_state:
        logger.info("=== Khởi tạo phiên chat mới ===\n")
        st.session_state.chat_history = []
        st.session_state.chat_history.append(AIMessage(
            content="Chào bạn, Tôi là trợ lý ảo của Đại học . Tôi có thể giúp gì cho bạn?"
        ))
    
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

# Main interaction
display_chat_history()
user_query = st.chat_input("Nhập tin nhắn của bạn...")
if user_query:
    process_user_input(user_query)