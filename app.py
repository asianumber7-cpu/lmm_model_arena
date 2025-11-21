import streamlit as st
import torch
from PIL import Image
from transformers import (
    AutoProcessor, AutoModel, 
    CLIPProcessor, CLIPModel, 
    AutoTokenizer, SiglipProcessor, SiglipModel
)
import json
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ---------------------------------------------------------------------------------------------------
# 0. [핵심] 모델 확장 설정 (비교하고 싶은거 추가하면 됨) 돌려보니 4개이상이면 컴터아파함 돌릴것들만 주석풀어서비교ㄱ 
# ---------------------------------------------------------------------------------------------------
MODELS_CONFIG = [
    {
        "name": "KoCLIP (Ours)", 
        "id": "koclip/koclip-base-pt", 
        "type": "koclip",
        "desc": "한국어 특화 모델 (선정 모델)"
    },
    {
        "name": "OpenAI CLIP (Base)", 
        "id": "openai/clip-vit-base-patch32", 
        "type": "clip_std",
        "desc": "글로벌 스탠다드 (영어 기반)"
    },
    {
        "name": "Google SigLIP (SoTA)", 
        "id": "google/siglip-base-patch16-224", 
        "type": "siglip",
        "desc": "구글의 최신 모델 (성능 매우 높음)"
    },
    {
        "name": "AltCLIP (Multilingual)", 
        "id": "BAAI/AltCLIP", 
        "type": "clip_std", 
        "desc": "다국어 지원 모델 (비교군)"
    },
    # {
    #     "name": "Fashion-CLIP", 
    #     "id": "patrickjohncyh/fashion-clip", 
    #     "type": "clip_std",
    #     "desc": " 패션 데이터로 학습된 모델 (쇼핑몰 최적)"
    # },
    # {
    #     "name": "AltCLIP (Multilingual)", 
    #     "id": "BAAI/AltCLIP", 
    #     "type": "clip_std",
    #     "desc": "다국어 지원 (한국어 가능, 무거움)"
    # },
    # {
    #     "name": "MetaCLIP (Facebook)", 
    #     "id": "facebook/metaclip-b32-400m", 
    #     "type": "clip_std",
    #     "desc": "메타(페이스북)의 고성능 CLIP"
    # },
    # {
    #     "name": "LAION-2B (Open Source)", 
    #     "id": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K", 
    #     "type": "clip_std",
    #     "desc": "오픈소스 데이터 20억개로 학습"
    # },
    # {
    #     "name": "DFN-CLIP (Apple)", 
    #     "id": "apple/DFN5B-CLIP-ViT-H-14-378", 
    #     "type": "clip_std",
    #     "desc": "애플의 고품질 데이터 학습 (초대형 모델)"
    # }
]

# ------------------------------------------------
# 1. 환경 설정
# ------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
st.set_page_config(page_title="LMM Model Arena", layout="wide")

# 한글 폰트 설정 (Windows 깨짐 방지)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

st.title("🏟️ LMM 모델 성능 비교 아레나")
st.markdown(f"""
**실험 목적:** 유명한 글로벌 LMM 모델들과 **KoCLIP**을 동일한 조건에서 경쟁시켜, 
한국어 쇼핑몰 검색 환경에서의 **적합성(Accuracy)**과 **효율성(Speed)**을 증명합니다.
* **실행 환경:** {device.upper()}
""")

# ------------------------------------------------
# 2. 동적 모델 로더
# ------------------------------------------------
@st.cache_resource
def load_all_models():
    loaded_models = {}
    
    for config in MODELS_CONFIG:
        model_name = config['name']
        model_id = config['id']
        m_type = config['type']
        
        print(f"🚀 로딩 시작: {model_name}") 
        
        try:
            if m_type == 'koclip':
                model = AutoModel.from_pretrained(model_id).to(device)
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                loaded_models[model_name] = {
                    "model": model, "tokenizer": tokenizer, "processor": processor, "type": m_type
                }
                
            elif m_type == 'siglip':
                model = SiglipModel.from_pretrained(model_id).to(device)
                processor = SiglipProcessor.from_pretrained(model_id)
                loaded_models[model_name] = {
                    "model": model, "processor": processor, "type": m_type
                }
                
            else:
                # 일반적인 CLIP 계열
                model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
                processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
                loaded_models[model_name] = {
                    "model": model, "processor": processor, "type": "auto"
                }
            
            print(f"✅ {model_name} 로드 성공")

        except Exception as e:
            
            print(f"❌ {model_name} 로드 실패: {e}")
            continue
            
    return loaded_models

# ------------------------------------------------
# 3. 통합 추론 엔진
# ------------------------------------------------
def get_similarity_score(model_pack, image, text):
    model = model_pack['model']
    m_type = model_pack['type']
    
    try:
        with torch.no_grad():
            # --- A. KoCLIP ---
            if m_type == 'koclip':
                processor = model_pack['processor']
                tokenizer = model_pack['tokenizer']
                
                img_inputs = processor(images=image, return_tensors="pt").to(device)
                txt_inputs = tokenizer([text], padding=True, return_tensors="pt").to(device)
                
                img_feat = model.get_image_features(**img_inputs)
                txt_feat = model.get_text_features(**txt_inputs)
                
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
                return (img_feat @ txt_feat.T).item()

            # --- B. Google SigLIP ---
            elif m_type == 'siglip':
                processor = model_pack['processor']
                inputs = processor(text=[text], images=image, return_tensors="pt", padding="max_length").to(device)
                outputs = model(**inputs)
                # SigLIP은 값이 큼 -> 0~1 사이로 대략적 스케일링 (비교용)
                logits = outputs.logits_per_image.item()
                return max(0, logits) / 10.0 

            # --- C. Standard CLIP ---
            else:
                processor = model_pack['processor']
                inputs = processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
                outputs = model(**inputs)
                return outputs.logits_per_image.item() / 100.0
                
    except Exception:
        return 0.0

# ------------------------------------------------
# 4. 메인 UI
# ------------------------------------------------
with st.sidebar:
    st.header("⚙️ 설정")
    
    uploaded_file = st.file_uploader("데이터셋 (JSON)", type=['json'])
    
    default_path = os.path.join("data", "images")
    image_folder = st.text_input("이미지 경로", value=default_path)
    
    st.divider()
    st.write("📋 **비교 모델 목록**")
    for conf in MODELS_CONFIG:
        st.caption(f"- {conf['name']}")

if uploaded_file and image_folder:
    data = json.load(uploaded_file)
    
    if st.button("🔥 아레나 배틀 시작 (Run Benchmark)"):
        
        loaded_models = load_all_models()
        
        # 로드된 모델이 하나도 없으면 에러 출력하고 멈춤
        if not loaded_models:
            st.error("❌ 로드된 모델이 하나도 없습니다. 라이브러리를 설치하거나 인터넷 연결을 확인하세요.")
            st.stop()
            
        st.success(f"총 {len(loaded_models)}개의 모델이 참전했습니다! 실험을 시작합니다.")
            
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        start_time = time.time()
        
        # --- [디버깅] 로그창 생성 ---
        log_container = st.expander("🔍 디버깅 로그 (문제가 생기면 여기를 클릭하세요)", expanded=True)
        
        for i, item in enumerate(data):
            # [수정 포인트] JSON 파일에 있는 'image_filename' 키를 제일 먼저 찾도록 변경!
            img_name = item.get('image_filename') or item.get('image_file') or item.get('filename')
            caption = item.get('caption') or item.get('description') or item.get('text')
            
            # [디버그] 이제 파일명을 제대로 찾는지 확인
            if i < 3:
                log_container.write(f"[{i}번 데이터] 찾은 파일명: {img_name}")

            if not img_name:
                # 여전히 못 찾으면 에러 출력
                if i < 5: log_container.error(f"[{i}번] 여전히 파일명을 못 찾음. 키 목록: {list(item.keys())}")
                continue
            
            if not caption: caption = "unknown"
            
            # 2. 경로 확인 및 이미지 로드
            img_path = os.path.join(image_folder, img_name)
            
            # 파일이 진짜 있는지 확인
            if not os.path.exists(img_path):
                if i < 5: 
                    log_container.error(f"❌ 파일이 폴더에 없음: {img_path}")
                    log_container.info(f"폴더 경로 '{image_folder}' 안에 '{img_name}' 파일이 들어있는지 확인하세요.")
                continue
            
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                if i < 5: log_container.error(f"이미지 깨짐 ({img_name}): {e}")
                continue
            
            row_data = {"Index": i, "Caption": caption}
            
            # 로드에 성공한 모델들만 돌림
            for m_name, m_pack in loaded_models.items():
                score = get_similarity_score(m_pack, image, caption)
                row_data[m_name] = score
            
            results.append(row_data)
            
            progress = (i + 1) / len(data)
            progress_bar.progress(progress)
            if i % 5 == 0:
                status_text.text(f"Processing {i+1}/{len(data)}...")
        
        total_time = time.time() - start_time
        
        # 결과가 없으면 에러 출력하고 멈춤
        if not results:
            st.error("🚨 데이터 처리에 실패했습니다. 위의 '디버깅 로그'를 확인해보세요.")
            st.warning("가장 흔한 원인: 이미지 폴더 경로가 틀렸거나, JSON 파일 안의 파일명과 실제 파일명이 다릅니다.")
            st.stop()

        df = pd.DataFrame(results)
        
        # ------------------------------------------------
        # 5. 결과 시각화
        # ------------------------------------------------
        st.divider()
        st.subheader("🏆 최종 스코어보드")
        
        # 로드된 모델 컬럼만 선택해서 평균 계산
        valid_model_cols = [name for name in loaded_models.keys() if name in df.columns]
        
        if not valid_model_cols:
            st.error("결과를 계산할 모델 데이터가 없습니다.")
            st.stop()

        means = df[valid_model_cols].mean().sort_values(ascending=False)
        
        # 그래프
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['#FF4B4B' if 'KoCLIP' in name else '#A9A9A9' for name in means.index]
        
        sns.barplot(x=means.index, y=means.values, palette=colors, ax=ax)
        ax.set_title("모델별 평균 의미 이해도 (Semantic Accuracy)", fontsize=16, fontweight='bold')
        ax.set_ylabel("유사도 점수 (높을수록 좋음)")
        
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.4f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
            
        st.pyplot(fig)
        
        # 승자 결정 로직
        if not means.empty:
            winner = means.idxmax()
            st.success(f"🎉 **최종 승자:** {winner}")
            
            st.info(f"""
            **[결과 분석]**
            * **{winner}** 모델이 현재 데이터셋에서 가장 높은 정확도를 보였습니다.
            * 한국어 쇼핑 데이터 특성상 한국어 학습 모델이 유리함을 확인할 수 있습니다.
            """)
        
        st.download_button("결과 CSV 다운로드", df.to_csv().encode('utf-8'), "lmm_arena_results.csv")