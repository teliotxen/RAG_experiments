import os
import numpy as np
import re
import unicodedata
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz
import uuid
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from tqdm import tqdm
from torch import sigmoid
from collections import defaultdict


def load_documents_from_folder(folder_path):
    documents = []
    sources = []
    headers = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

                # 텍스트를 줄 단위로 분리
                lines = content.split('\n')

                current_header = None
                for line in lines:
                    line = line.strip()
                    if not line:  # 빈 줄 건너뛰기
                        continue

                    # #으로 시작하는 헤더 텍스트 확인
                    if line.startswith('-----## '):
                        current_header = line
                        continue

                    # 문서 정보 저장
                    documents.append(line)
                    sources.append(filename)
                    headers.append(current_header)
    return documents, sources, headers


# 6️⃣ 검색 함수 정의
def search_documents(embedding_model, query, index, headers, sources, documents,  top_k=3):
    query_vector = np.array([embedding_model.encode(query)])

    # Faiss를 사용해 가장 유사한 문서 검색
    distances, indices = index.search(query_vector, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        result = {
            "파일명": sources[idx],
            "헤더": headers[idx] if headers[idx] else "없음",
            "내용": documents[idx],
            "유사도 점수": 1 / (1 + distances[0][i])  # 거리를 유사도 점수로 변환 (0~1 사이)
        }
        results.append(result)

    return results

def spliter(_txt, name):
    _title = ""
    _source = []
    for line in _txt.split("\n"):
        if "-----##" in line:
            _title = line.split("-----##")[1]
        else:
            _source.append({"text": line, "source": f"{name}-{_title}"})
    return _source


def extract_text_from_pdf(pdf_path):
    # 사용자 지정 CropBox 여백 (단위: pt)
    CROPBOX_MARGIN = {
        "top": 8.05,      # 2cm
        "bottom": 56.7,   # 2cm
        "left": 56.7,     # 1.5cm + 0.5cm(제본 여백)
        "right": 42.525   # 1.5cm
    }
    """PDF 파일에서 CropBox를 적용하여 텍스트를 추출"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            texts = []
            metadata_list = []
            for page_num, page in enumerate(pdf.pages, start=1):
                # 페이지 크기 가져오기
                x0, y0, x1, y1 = page.bbox

                # CropBox 적용하여 크롭된 영역 설정
                cropped_bbox = (
                    x0 + CROPBOX_MARGIN["left"],
                    y0 + CROPBOX_MARGIN["bottom"],
                    x1 - CROPBOX_MARGIN["right"],
                    y1 - CROPBOX_MARGIN["top"]
                )

                # CropBox 적용
                cropped_page = page.within_bbox(cropped_bbox)
                text = cropped_page.extract_text() or ""
                text = preprocess_text(text)

                if text:
                    metadata = {
                        "source": os.path.basename(pdf_path),
                        "full_path": pdf_path,
                        "file_type": "pdf",
                        "page_number": page_num  # 페이지 번호 추가
                    }
                    texts.append(text)
                    metadata_list.append(metadata)
        return texts, metadata_list
    except Exception as e:
        print(f"⚠️ {pdf_path} 변환 중 오류 발생: {str(e)}")
        return [], []


def preprocess_text(text):
    """텍스트 전처리 함수"""
    # 유니코드 정규화
    text = unicodedata.normalize('NFC', text)
    # 보이지 않는 특수 문자 제거 (좌-to-우 마크 등)
    text = re.sub(r'[​‌‍⁠﻿]', '', text)
    # 한글 뒤에 오는 % 기호 제거
    text = re.sub(r'([\uac00-\ud7a3])%', r'\1', text)
    # 연속된 공백을 하나로
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def clean_percent_symbols(text):
    # 한글 문자 뒤에 붙은 '%' 제거
    cleaned_text = re.sub(r'([\uac00-\ud7a3])%', r'\1', text)
    return cleaned_text


def create_chunks_from_texts(texts, metadata_list, chunk_size=500, chunk_overlap=50):
    """텍스트를 청크로 분할하며 PDF 메타데이터 추가"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        length_function=len
    )
    documents = []
    for text, metadata in zip(texts, metadata_list):
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_id"] = i  # 청크 ID 추가
            documents.append(Document(page_content=chunk, metadata=chunk_metadata))
    return documents


def process_documents_to_vector_db(pdf_dir, output_dir,
                                   embedding_model_name="intfloat/multilingual-e5-small",
                                   chunk_size=500, chunk_overlap=50):
    """PDF 문서를 변환, 전처리 후 벡터 DB로 저장"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    if not pdf_files:
        print("⚠️ 처리할 PDF 파일이 없습니다.")
        return

    print(f"🔍 총 {len(pdf_files)}개의 PDF 파일을 처리합니다...")
    texts = []
    metadata_list = []

    for file_name in tqdm(pdf_files, desc="파일 변환 중"):
        file_path = os.path.join(pdf_dir, file_name)
        pdf_texts, pdf_metadata = extract_text_from_pdf(file_path)
        texts.extend(pdf_texts)
        metadata_list.extend(pdf_metadata)

    print(f"📄 텍스트를 청크 크기 {chunk_size}(겹침 {chunk_overlap})로 분할합니다...")
    documents = create_chunks_from_texts(texts, metadata_list, chunk_size, chunk_overlap)

    if not documents:
        print("⚠️ 처리할 문서가 없습니다.")
        return

    print(f"✅ {len(documents)}개의 청크가 생성되었습니다.")

    print(f"🧠 임베딩 모델 '{embedding_model_name}'을 로드합니다...")
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

    print("🔢 벡터 DB를 생성합니다...")
    vector_db = FAISS.from_documents(documents, embedding_model)

    vector_db.save_local(output_dir)
    print(f"✅ '{output_dir}'에 벡터 데이터베이스 저장 완료")

    return vector_db
  # PyMuPDF


def extract_text_with_positions(pdf_path):
    doc = fitz.open(pdf_path)
    texts = []
    metadata = []

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    line_text = " ".join([span["text"] for span in line["spans"]])
                    bbox = line["bbox"]
                    if line_text.strip():
                        texts.append(line_text)
                        metadata.append({
                            "source": pdf_path,
                            "page": page_num,
                            "bbox": bbox,
                        })

    return texts, metadata


def create_chunks_with_metadata(texts, metadata, chunk_size, chunk_overlap):
    chunks = []
    for i, text in enumerate(texts):
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk_text = text[start:end]
            chunk_meta = metadata[i].copy()
            chunk_meta["offset"] = start
            chunk_meta["length"] = end - start
            chunks.append(Document(page_content=chunk_text, metadata=chunk_meta))
            start += chunk_size - chunk_overlap
    return chunks


def process_pdfs_to_faiss_with_positions(pdf_dir, output_dir,
                                         embedding_model_name="intfloat/multilingual-e5-small",
                                         chunk_size=500, chunk_overlap=50):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    if not pdf_files:
        print("⚠️ PDF 없음")
        return

    texts = []
    metadata_list = []

    for file in tqdm(pdf_files, desc="PDF 추출 중"):
        file_path = os.path.join(pdf_dir, file)
        file_texts, file_metadata = extract_text_with_positions(file_path)
        texts.extend(file_texts)
        metadata_list.extend(file_metadata)

    documents = create_chunks_with_metadata(texts, metadata_list, chunk_size, chunk_overlap)
    print(documents)
    print(f"✅ 총 {len(documents)}개 청크 생성")

    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vector_db = FAISS.from_documents(documents, embedding_model)
    vector_db.save_local(output_dir)

    print(f"✅ 저장 완료: {output_dir}")
    return vector_db


def highlight_matches_in_pdf(pdf_path, matched_chunks, output_pdf_path):
    doc = fitz.open(pdf_path)
    for chunk in matched_chunks:
        meta = chunk.metadata
        page = doc[meta["page"]]
        bbox = meta.get("bbox")
        if bbox:
            page.add_highlight_annot(bbox)
    doc.save(output_pdf_path)
    print(f"✅ 결과 저장됨: {output_pdf_path}")


def export_evidence_pdfs(result, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # source_path + page_num 기준으로 bbox 모으기
    grouped_evidence = defaultdict(list)

    for item in result['answer']['input_documents']:
        doc_obj = item
        source_path = doc_obj.metadata['source']
        page_num = int(doc_obj.metadata['page'])
        bbox_data = doc_obj.metadata['bbox']
        grouped_evidence[(source_path, page_num)].append(bbox_data)

    for (source_path, page_num), bboxes in grouped_evidence.items():
        try:
            if not os.path.isfile(source_path):
                print(f"[Warning] 파일 없음: {source_path}")
                continue

            doc = fitz.open(source_path)

            if page_num < 0 or page_num >= len(doc):
                print(f"[Warning] 잘못된 페이지 번호: {page_num} (총 페이지: {len(doc)})")
                doc.close()
                continue

            page = doc[page_num]

            for bbox_data in bboxes:
                if isinstance(bbox_data, (list, tuple)):
                    bbox = fitz.Rect(*bbox_data)
                elif isinstance(bbox_data, fitz.Rect):
                    bbox = bbox_data
                else:
                    raise ValueError(f"[Error] bbox 형식이 잘못됨: {bbox_data}")

                # 형광 표시
                page.draw_rect(
                    bbox,
                    color=None,
                    fill=(1, 1, 0),
                    fill_opacity=0.5,
                    overlay=True
                )

            # 새로운 PDF에 해당 페이지만 삽입
            new_doc = fitz.open()
            new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)

            filename = f"{os.path.splitext(os.path.basename(source_path))[0]}_page{page_num}_{uuid.uuid4().hex[:8]}.pdf"
            save_path = os.path.join(output_dir, filename)
            new_doc.save(save_path)

            print(f"[✓] 저장 완료: {save_path}")

            new_doc.close()
            doc.close()

        except Exception as e:
            print(f"[Error] 처리 중 오류 발생: {e}")


def get_qa_score(question, context, qa_tokenizer, qa_model):
    inputs = qa_tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = qa_model(**inputs)
        # ko-reranker 모델은 logits이 [1] 형태로 나오므로 squeeze
        # relevance_score = outputs.logits.squeeze().item()
        relevance_score = sigmoid(outputs.logits).squeeze().item()
    return relevance_score


from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import os
from tqdm import tqdm


def process_pdfs_to_chroma_with_positions(
    pdf_dir,
    output_dir,
    embedding_model_name="intfloat/multilingual-e5-small",
    chunk_size=500,
    chunk_overlap=50
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    if not pdf_files:
        print("⚠️ PDF 없음")
        return

    texts = []
    metadata_list = []

    for file in tqdm(pdf_files, desc="📄 PDF 추출 중"):
        file_path = os.path.join(pdf_dir, file)
        file_texts, file_metadata = extract_text_with_positions(file_path)
        texts.extend(file_texts)
        metadata_list.extend(file_metadata)

    documents = create_chunks_with_metadata(texts, metadata_list, chunk_size, chunk_overlap)
    print(f"✅ 총 {len(documents)}개 청크 생성")

    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # Chroma에 저장
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=output_dir
    )

    vector_db.persist()
    print(f"✅ Chroma 저장 완료: {output_dir}")

    return vector_db


def export_evidence_pdfs_beta(result, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # source_path + page_num 기준으로 bbox 모으기
    grouped_evidence = defaultdict(list)

    for item in result:
        doc_obj = item
        source_path = doc_obj.metadata['source']
        page_num = int(doc_obj.metadata['page'])
        bbox_data = doc_obj.metadata['bbox']
        grouped_evidence[(source_path, page_num)].append(bbox_data)

    for (source_path, page_num), bboxes in grouped_evidence.items():
        try:
            if not os.path.isfile(source_path):
                print(f"[Warning] 파일 없음: {source_path}")
                continue

            doc = fitz.open(source_path)

            if page_num < 0 or page_num >= len(doc):
                print(f"[Warning] 잘못된 페이지 번호: {page_num} (총 페이지: {len(doc)})")
                doc.close()
                continue

            page = doc[page_num]

            for bbox_data in bboxes:
                if isinstance(bbox_data, (list, tuple)):
                    bbox = fitz.Rect(*bbox_data)
                elif isinstance(bbox_data, fitz.Rect):
                    bbox = bbox_data
                else:
                    raise ValueError(f"[Error] bbox 형식이 잘못됨: {bbox_data}")

                # 형광 표시
                page.draw_rect(
                    bbox,
                    color=None,
                    fill=(1, 1, 0),
                    fill_opacity=0.5,
                    overlay=True
                )

            # 새로운 PDF에 해당 페이지만 삽입
            new_doc = fitz.open()
            new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)

            filename = f"{os.path.splitext(os.path.basename(source_path))[0]}_page{page_num}_{uuid.uuid4().hex[:8]}.pdf"
            save_path = os.path.join(output_dir, filename)
            new_doc.save(save_path)

            # print(f"[✓] 저장 완료: {save_path}")

            new_doc.close()
            doc.close()

        except Exception as e:
            print(f"[Error] 처리 중 오류 발생: {e}")
