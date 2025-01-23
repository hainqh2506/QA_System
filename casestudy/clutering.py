import random
from typing import Dict, List, Optional, Tuple, Set , Literal, Callable
import numpy as np
import pandas as pd
import umap
from sklearn.mixture import GaussianMixture
import tiktoken
# Set a random seed for reproducibility
RANDOM_SEED = 224
random.seed(RANDOM_SEED)
def global_cluster_embeddings(
        embeddings:np.ndarray,
        dim: int,
        n_neighbors: Optional[int] = None,
        metric:str = "cosine",
) -> np.ndarray:
    """Thực hiện giảm số chiều toàn cục trên các embeddings bằng cách sử dụng UMAP.

    Tham số:
    - embeddings: Các embeddings đầu vào dưới dạng mảng numpy.
    - dim: Số chiều mục tiêu cho không gian sau khi giảm số chiều.
    - n_neighbors: Tùy chọn; số lượng hàng xóm cần xem xét cho mỗi điểm. 
                   Nếu không được cung cấp, sẽ mặc định là căn bậc hai của số lượng embeddings.
    - metric: Thước đo khoảng cách được sử dụng cho UMAP, mặc định là "cosine".

    Trả về: Mảng numpy chứa các embeddings đã được giảm số chiều theo số chiều được chỉ định.
    """
    
    # Nếu số lượng hàng xóm không được chỉ định, tính giá trị mặc định
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)  # Mặc định là căn bậc hai của số embeddings
    
    # Áp dụng UMAP để giảm số chiều của embeddings với metric được chọn
    return umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric 
    ).fit_transform(embeddings)


def local_cluster_embeddings(
        embeddings: np.ndarray, dim:int, num_neighbors:int=10, metric:str="cosine"
) -> np.ndarray:
    """Thực hiện giảm số chiều cục bộ trên các embeddings bằng cách sử dụng UMAP.

    Tham số:
    - embeddings: Các embeddings đầu vào dưới dạng mảng numpy.
    - dim: Số chiều mục tiêu cho không gian sau khi giảm số chiều.
    - num_neighbors: Số lượng hàng xóm cần xem xét cho mỗi điểm, mặc định là 10.
    - metric: Thước đo khoảng cách được sử dụng cho UMAP, mặc định là "cosine".

    Trả về: Mảng numpy chứa các embeddings đã được giảm số chiều theo số chiều được chỉ định.
    """

    # Áp dụng UMAP để giảm số chiều của embeddings với số hàng xóm và metric được chọn
    return umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)

def get_optimal_clusters(
        embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    """Tìm số lượng cụm tối ưu cho các embeddings bằng cách sử dụng Gaussian Mixture Model (GMM) và tiêu chí thông tin Bayes (BIC).

    Tham số:
    - embeddings: Các embeddings đầu vào dưới dạng mảng numpy.
    - max_clusters: Số lượng cụm tối đa cần xem xét. Mặc định là 50.
    - random_state: Hạt giống ngẫu nhiên để đảm bảo tái lập kết quả. Sử dụng RANDOM_SEED để cố định.

    Trả về: Số lượng cụm tối ưu dựa trên giá trị BIC nhỏ nhất.
    """

    # Giới hạn số lượng cụm tối đa bằng số lượng embeddings nếu số cụm lớn hơn số embeddings
    max_clusters = min(max_clusters, len(embeddings))
    
    # Tạo dãy số từ 1 đến max_clusters để kiểm tra số lượng cụm
    n_clusters = np.arange(1, max_clusters)
    
    # Khởi tạo danh sách để lưu giá trị BIC
    bics = []
    
    # Duyệt qua từng số lượng cụm
    for n in n_clusters:
        # Khởi tạo Gaussian Mixture Model với số lượng cụm n và random_state cố định
        gm = GaussianMixture(n_components=n, random_state=random_state)
        
        # Huấn luyện mô hình với các embeddings
        gm.fit(embeddings)
        
        # Tính toán giá trị BIC và thêm vào danh sách bics
        bics.append(gm.bic(embeddings))
    
    # Trả về số lượng cụm tối ưu tương ứng với giá trị BIC nhỏ nhất
    return n_clusters[np.argmin(bics)]

def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state:int=0) -> Tuple[List[np.ndarray], int]:
    """Thực hiện gom cụm các embeddings bằng Gaussian Mixture Model (GMM) với ngưỡng xác suất.

    Tham số:
    - embeddings: Mảng numpy chứa các embeddings đầu vào.
    - threshold: Ngưỡng xác suất để quyết định thuộc tính cụm của mỗi embedding.
    - random_state: Hạt giống ngẫu nhiên để cố định kết quả, mặc định là 0.

    Trả về:
    - labels: Danh sách các mảng numpy chứa chỉ số cụm cho mỗi embedding, tương ứng với các xác suất vượt quá ngưỡng.
    - n_clusters: Số lượng cụm tối ưu dựa trên GMM.
    """
    
    # Tìm số lượng cụm tối ưu
    n_clusters = get_optimal_clusters(embeddings)
    
    # Khởi tạo mô hình Gaussian Mixture Model với số lượng cụm tối ưu
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    
    # Huấn luyện mô hình với các embeddings
    gm.fit(embeddings)
    
    # Tính xác suất của mỗi embedding thuộc về các cụm
    probs = gm.predict_proba(embeddings)
    
    # Gán nhãn cụm cho mỗi embedding dựa trên ngưỡng xác suất
    labels = [np.where(prob > threshold)[0] for prob in probs]
    
    return labels, n_clusters


def perform_clustering(embeddings: np.ndarray, dim: int, threshold: float) -> List[np.ndarray]:
    """Thực hiện quy trình gom cụm bằng cách kết hợp giảm số chiều toàn cục và cục bộ, sau đó áp dụng GMM để phân cụm.

    Tham số:
    - embeddings: Mảng numpy chứa các embeddings đầu vào.
    - dim: Số chiều mục tiêu để giảm số chiều của các embeddings.
    - threshold: Ngưỡng xác suất để quyết định thuộc tính cụm của mỗi embedding.

    Trả về:
    - all_local_clusters: Danh sách các mảng numpy chứa chỉ số cụm cục bộ sau khi phân cụm.
    """
    # If the number of embeddings is less than or equal to the dimension, return a list of zeros
    # This means all nodes are in the same cluster.
    # Otherwise, we will get an error when trying to cluster.
    # Tránh phân cụm khi số lượng embeddings ít hơn hoặc bằng số chiều + 1
    if len(embeddings) <= dim + 1:
        return [np.array([0]) for _ in range(len(embeddings))]

    # Giảm số chiều toàn cục cho các embeddings
    reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)

    # Thực hiện phân cụm toàn cục bằng GMM
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global, threshold
    )

    # Khởi tạo danh sách rỗng để lưu cụm cục bộ và tổng số cụm
    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    # Lặp qua từng cụm toàn cục để thực hiện phân cụm cục bộ
    for i in range(n_global_clusters):
        # Lấy các embeddings thuộc về cụm toàn cục hiện tại
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]

        # Nếu không có embeddings trong cụm toàn cục hiện tại, bỏ qua
        if len(global_cluster_embeddings_) == 0:
            continue

        # Nếu cụm nhỏ, gán cụm trực tiếp
        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            # Giảm số chiều cục bộ và phân cụm bằng GMM
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local, threshold
            )

        # Gán ID cụm cục bộ, điều chỉnh theo tổng số cụm đã xử lý
        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])
            ]
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        # Cập nhật tổng số cụm
        total_clusters += n_local_clusters

    return all_local_clusters

def get_clusters(
    texts: List[str],
    tokenizer: tiktoken.Encoding ,
    max_length_in_cluster: int = 5000,
    reduction_dimension: int = 10,
    threshold: float = 0.1,
    prev_total_length=None,
    embedding_function: Callable = None  # Hàm embedding cần được truyền vào
) -> pd.DataFrame:
    """
    Thực hiện gom cụm các văn bản dựa trên embeddings và số token, với giới hạn số lượng token trong mỗi cụm.
    
    Trả về: DataFrame chứa văn bản, embedding và nhãn cụm.
    """
    if tokenizer is None:
            tokenizer = tiktoken.get_encoding("cl100k_base")
    if embedding_function is None:
        raise ValueError("embedding_function không được cung cấp")

    # Tính toán embedding cho các văn bản
    text_embedding = embedding_function(texts)
    clusters = perform_clustering(text_embedding, reduction_dimension, threshold)
    
    # Chuyển đổi các mảng NumPy thành số nguyên hoặc danh sách
    cluster_labels = [int(c[0]) for c in clusters]  # Lấy giá trị đầu tiên từ mỗi mảng numpy
    
    df_store = pd.DataFrame({
        "text": texts,
        "embd": list(text_embedding),
        "cluster": cluster_labels
    })
    
    # Xử lý các cụm vượt quá giới hạn token
    df_expanded = pd.DataFrame()
    for cluster in df_store['cluster'].unique():
        df_cluster = df_store[df_store['cluster'] == cluster]
        total_length = df_cluster['text'].apply(lambda x: len(tokenizer.encode(x))).sum()
        
        if total_length > max_length_in_cluster and (prev_total_length is None or total_length < prev_total_length):
            # Đệ quy để phân cụm lại, truyền embedding_function vào
            sub_df = get_clusters(
                df_cluster['text'].tolist(),
                max_length_in_cluster=max_length_in_cluster,
                tokenizer=tokenizer,
                reduction_dimension=reduction_dimension,
                threshold=threshold,
                prev_total_length=total_length,
                embedding_function=embedding_function  # Truyền hàm embedding vào
            )
            df_expanded = pd.concat([df_expanded, sub_df], ignore_index=True)
        else:
            df_expanded = pd.concat([df_expanded, df_cluster], ignore_index=True)
    
    return df_expanded
