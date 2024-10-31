# Bước 0: Đọc file Q&A và tạo embedding cho từng câu hỏi
def load_sample_qa(file_path):
    with open(file_path, 'r') as f:
        sample_qa = {line.split(':')[0].strip(): line.split(':')[1].strip() for line in f}
    sample_embeddings = {q: encode_text_to_vector(q) for q in sample_qa.keys()}
    return sample_qa, sample_embeddings

# Bước 3: Định tuyến câu hỏi mới đến module hoặc file Q&A nếu score cao
def route_new_query(new_query, sample_qa, sample_embeddings):
    new_embedding = encode_text_to_vector(new_query)

    # Tính điểm cosine với các câu hỏi trong sample.txt
    sample_scores = {q: cosine_similarity(new_embedding, emb) for q, emb in sample_embeddings.items()}
    max_sample_score, max_sample_question = max((score, q) for q, score in sample_scores.items())

    # Tính điểm giữa các module khác
    avg_policy_score = mean([cosine_similarity(new_embedding, emb) for emb in policy_query_embeddings])
    avg_support_score = mean([cosine_similarity(new_embedding, emb) for emb in support_query_embeddings])

    # Xác định module hoặc file
    if max_sample_score > avg_policy_score and max_sample_score > avg_support_score:
        return "sample", sample_qa[max_sample_question]
    elif avg_support_score > avg_policy_score:
        return "support", None
    else:
        return "policy", None

# Bước 4: Xử lý câu hỏi hoặc trỏ đến câu trả lời từ sample.txt
def handle_query(query, module, sample_answer=None):
    if module == "sample":
        return sample_answer
    elif module == "support":
        return generate_response_using_LLM(query)
    else:
        return generate_response_using_RAG(query)
