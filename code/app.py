import logging
import streamlit as st
import time
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from model_config import load_gpt4o_mini_model, VietnameseEmbeddings
from pipeline import step5_retrieval ,step6_qa_db
# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
COLLECTION_NAME = "raptor"
QA_COLLECTION_NAME = "base_qa"
THRESHOLD = 0.85
# Khởi tạo các thành phần global
load_dotenv()
st.set_page_config(page_title="University Assistant", page_icon="🤖")
st.title("University Assistant")
@st.cache_resource
def initialize_model_and_db():
    logger.info("Khởi tạo model, database và FAQ...")
    llm = load_gpt4o_mini_model()
    corpus, milvus_db= step5_retrieval(collection_name=COLLECTION_NAME)
    milvus_qa_db = step6_qa_db(collection_name = QA_COLLECTION_NAME)
    return llm, corpus, milvus_db, milvus_qa_db

llm, corpus, milvus_db, milvus_qa_db = initialize_model_and_db()

def rewrite_question(user_query, formatted_history):
    logger.info("=== Bắt đầu viết lại câu hỏi ===")
    #logger.info(f"Câu hỏi gốc: {user_query}")
    
    if formatted_history:
        logger.info("Đang sử dụng lịch sử chat để viết lại câu hỏi")
        #logger.info(f"Lịch sử chat đã format: {formatted_history}")
    else:
        logger.info("Không có lịch sử chat trước đó")
    
    rewrite_template = """
    Bạn đang trong một cuộc hội thoại FAQ. 
    Nhiệm vụ của bạn là:
    1. Tinh chỉnh lại ##TRUY VẤN CỦA NGƯỜI DÙNG một cách rõ ràng, độc lập, không phụ thuộc vào ngữ cảnh ##LỊCH SỬ TRÒ CHUYỆN: bằng cách tập trung vào chi tiết cuộc hội thoại gần nhất.
    2. Phân loại ##TRUY VẤN CỦA NGƯỜI DÙNG có thuộc thẻ <spam> hay không, nếu có thì trả về <spam> + ##TRUY VẤN CỦA NGƯỜI DÙNG.

    QUY TẮC TINH CHỈNH:
    - Làm cho truy vấn cụ thể và rõ ràng hơn, đảm bảo phù hợp với ý định của người dùng và hiểu biết từ ##LỊCH SỬ TRÒ CHUYỆN:.
    - Tập trung vào nội dung chính, giữ nguyên ý nghĩa câu hỏi.
    - Viết ngắn gọn, dễ hiểu, sửa lỗi chính tả, ngữ pháp.

    QUY TẮC PHÂN LOẠI:
    Trả về <spam> + ##TRUY VẤN CỦA NGƯỜI DÙNG nếu câu hỏi chứa:
    - Ngôn từ xúc phạm, thù địch
    - Nội dung quảng cáo, spam
    - Từ ngữ tục tĩu, khiêu dâm
    - Link độc hại, lừa đảo
    - Nội dung chính trị, tôn giáo nhạy cảm
    - Câu hỏi không liên quan, không có ý nghĩa
    - Câu hỏi xã giao đơn giản (ví dụ: chào, tạm biệt, bạn khỏe không)
    - Câu hỏi về thời tiết chung chung (ví dụ: hôm nay trời thế nào)
    - Giữ nguyên ##TRUY VẤN CỦA NGƯỜI DÙNG nếu đã rõ ràng và không thuộc các trường hợp trên.
    Ví dụ: <spam> bạn thật ngốc
    
    ##LỊCH SỬ TRÒ CHUYỆN:
    {chat_history}

    ##TRUY VẤN CỦA NGƯỜI DÙNG: 
    User: {question}

    ##TRUY VẤN ĐÃ CHỈNH SỬA:
    User:
"""
    rewrite_prompt = ChatPromptTemplate.from_template(rewrite_template)
    chain = rewrite_prompt | llm | StrOutputParser()
    
    logger.info("Đang gửi yêu cầu viết lại câu hỏi tới model...")
    rewritten_query = chain.invoke({
        "chat_history": formatted_history,
        "question": user_query
    })
    logger.info(f"câu hỏi gốc: {user_query}")
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
def check_faq(user_query, milvus_qa_db = milvus_qa_db, threshold=THRESHOLD):
    """
    Kiểm tra câu hỏi trong MilvusQADB thay vì file JSON.
    - `user_query`: Câu hỏi người dùng.
    - `milvus_qa_db`: Đối tượng MilvusQADB đã khởi tạo.
    - `threshold`: Ngưỡng độ tương đồng (cosine similarity).

    Trả về:
    - Câu trả lời từ Milvus nếu tìm thấy.
    - None nếu không tìm thấy câu trả lời phù hợp.
    """
    logger.info("Truy vấn MilvusQADB cho FAQ...")
    # Gọi `semantic_qa_search` để tìm kiếm trong collection
    result = milvus_qa_db.semantic_qa_search(query=user_query, similarity_threshold=threshold, top_k=1)

    if result:
        logger.info("Câu hỏi phù hợp tìm thấy trong MilvusQADB.")
        return result.get("answer")  # Trả về câu trả lời tốt nhất
    
    logger.info("Không tìm thấy câu trả lời phù hợp trong MilvusQADB.")
    return None
def stream_text(text):
    """Generator function để stream từng từ của văn bản"""
    time.sleep(1)
    for word in text.split():
        yield word + " "
        # Nếu muốn điều chỉnh tốc độ stream, có thể thêm time.sleep()
        time.sleep(0.05)

def get_response(user_query, chat_history,milvus_db = milvus_db ):
    logger.info("=== Bắt đầu quá trình tạo câu trả lời ===")

    # Kiểm tra trong FAQ
    answer = check_faq(user_query)
    if answer:
        return stream_text(answer)
    #Không tìm thấy, gọi đến RAG
    logger.info("Không tìm thấy câu hỏi phù hợp trong FAQ, chuyển đến hệ thống RAG.")
    # Viết lại câu hỏi
        #Format dữ liệu
    logger.info("=== Bắt đầu format chat history ===")
    formatted_history = format_chat_history(chat_history, max_pairs=5)
    rewritten_query = rewrite_question(user_query, formatted_history)
    if rewritten_query.startswith("<spam>"):
        default_response = "Xin lỗi, tôi chỉ là trợ lý ảo của Đại học và tôi không thể trả lời câu hỏi này. Nếu bạn có câu hỏi nào khác hãy cho tôi biết nhé!"
        logger.info("Phát hiện câu hỏi không liên quan, trả lời mặc định.")
        return stream_text(default_response)
    # Truy vấn thông tin liên quan
    logger.info("=== Bắt đầu truy xuất tài liệu ===")
    relevant_docs = milvus_db.perform_retrieval(query=rewritten_query, top_k=5)
    logger.info(f"Số lượng tài liệu tìm được: {len(relevant_docs)}")
    logger.info(f"relevant docs: {relevant_docs}")
    logger.info("=== Kết thúc truy xuất tài liệu ===")


    doc_context = "\n\n".join([doc.page_content + doc.metadata.get("source") for doc in relevant_docs])
    logger.info(f"Độ dài context: {len(doc_context)} ký tự")
    logger.info("=== Kết thúc format dữ liệu ===")
    
    QAtemplate = """Bạn là một trợ lý ảo của Đại học , thông minh và thân thiện. 
    
    Dựa trên ##LỊCH SỬ TRÒ CHUYỆN và ##TÀi LIỆU THAM KHẢO được cung cấp, hãy trả lời ##CÂU HỎI CỦA NGƯỜI DÙNG.
    Nếu câu trả lời yêu cầu thông tin cá nhân hoặc thông tin từ cuộc trò chuyện trước đó, hãy sử dụng thông tin từ lịch sử trò chuyện.
    Nếu câu trả lời yêu cầu kiến thức chuyên môn, hãy tham khảo các tài liệu được cung cấp.
    Nếu không tìm thấy thông tin trong cả hai nguồn trên, hãy trả lời 'Xin lỗi, tôi không có đủ thông tin để trả lời câu hỏi này, nếu bạn muốn biết thêm thông tin gì khác hãy cho tôi biết nhé!'.

    ##LỊCH SỬ TRÒ CHUYỆN:
    {chat_history}

    ##TÀi LIỆU THAM KHẢO:
    {documents}

    ##CÂU HỎI CỦA NGƯỜI DÙNG: {question}
    
    Trả lời bằng tiếng Việt và giữ giọng điệu thân thiện, từ chối trả lời các câu hỏi mang tính nhạy cảm.
    ##CÂU TRẢ LỜI:
    """

    QAprompt = ChatPromptTemplate.from_template(QAtemplate)
    logger.info("=== Bắt đầu tạo câu trả lời ===")
    logger.info(f"=== full QA_prompt {QAprompt}===")
    
    chain = QAprompt | llm | StrOutputParser()
    logger.info("Đang gửi yêu cầu tới model...")
    response = chain.stream({
        "chat_history": formatted_history,
        "documents": doc_context,
        "question": rewritten_query
    })
    logger.info("=== Kết thúc tạo câu trả lời ===")
    logger.info(f"Câu trả lời: {response}")
    return response
####code mới #######
def process_user_input(user_query):
    if not user_query:
        return None
    if len(user_query) > 512:
        st.error("Độ dài câu hỏi quá dài, vui lòng nhập câu hỏi ngắn hơn!")
        return None
    logger.info(f"\n=== Bắt đầu xử lý câu hỏi mới ===")
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
    
    with st.chat_message("AI"):
        response = st.write_stream(get_response(user_query, st.session_state.chat_history))
        logger.info(f"Câu trả lời cho người dùng: {response}")
        logger.info("Đã gửi câu trả lời cho người dùng")

    st.session_state.chat_history.append(AIMessage(content=response))
    logger.info("Đã cập nhật lịch sử chat")
    logger.info("=== Kết thúc xử lý câu hỏi ===\n")
    return response

def display_chat_history():
    if "chat_history" not in st.session_state:
        logger.info("=== Khởi tạo phiên chat mới ===")
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