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
# 0. [핵심] 모델 설정 (AltCLIP, KoCLIP 등 리모트 코드 필요한 모델 설정 강화)
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
        "name": "AltCLIP (Multilingual)", 
        "id": "BAAI/AltCLIP", 
        "type": "clip_std", 
        "desc": "다국어 지원 모델 (비교군)"
    },
    # {
    #     "name": "Google SigLIP (SoTA)", 
    #     "id": "google/siglip-base-patch16-224", 
    #     "type": "siglip",
    #     "desc": "구글의 최신 모델 (성능 매우 높음)"
    # },
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
**실행 상태:** `{device.upper()}` 환경에서 실행 중
* **KoCLIP / AltCLIP 로드 팁:** 보안 경고가 뜨면 `transformers` 버전을 4.46.3으로 낮춰주세요.
""")

# ------------------------------------------------
# 2. 동적 모델 로더 (안정성 강화 버전)
# ------------------------------------------------
@st.cache_resource
def load_all_models():
    loaded_models = {}
    
    for config in MODELS_CONFIG:
        model_name = config['name']
        model_id = config['id']
        m_type = config['type']
        
        print(f"🚀 로딩 시작: {model_name}...") 
        
        try:
            # --- A. KoCLIP 로드 ---
            if m_type == 'koclip':
                # KoCLIP은 koclip/koclip-base-pt 경로에서 바로 로드
                model = AutoModel.from_pretrained(
                    model_id, 
                    trust_remote_code=True # 필수: 외부 코드 허용
                ).to(device)
                
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    trust_remote_code=True
                )
                # KoCLIP은 이미지 처리를 위해 OpenAI CLIP의 전처리기(Processor)를 빌려 씀
                processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                
                loaded_models[model_name] = {
                    "model": model, "tokenizer": tokenizer, "processor": processor, "type": m_type
                }
                
            # --- B. SigLIP 로드 ---
            elif m_type == 'siglip':
                model = SiglipModel.from_pretrained(model_id).to(device)
                processor = SiglipProcessor.from_pretrained(model_id)
                loaded_models[model_name] = {
                    "model": model, "processor": processor, "type": m_type
                }
                
            # --- C. CLIP / AltCLIP (Standard) ---
            else:
                # AltCLIP 등은 trust_remote_code=True가 있어야 안전하게 로드됨
                model = AutoModel.from_pretrained(
                    model_id, 
                    trust_remote_code=True 
                ).to(device)
                
                try:
                    processor = AutoProcessor.from_pretrained(
                        model_id, 
                        trust_remote_code=True
                    )
                except:
                    # 만약 AutoProcessor가 실패하면 CLIPProcessor로 시도
                    processor = CLIPProcessor.from_pretrained(model_id)

                loaded_models[model_name] = {
                    "model": model, "processor": processor, "type": "auto"
                }
            
            print(f"✅ {model_name} 로드 성공")

        except Exception as e:
            print(f"❌ {model_name} 로드 실패: {e}")
            # Streamlit 화면에도 에러 띄워주기 (디버깅용)
            st.error(f"⚠️ **{model_name}** 로드 실패! \n에러 내용: {e}")
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
                
                # 이미지 처리
                img_inputs = processor(images=image, return_tensors="pt").to(device)
                # 텍스트 처리
                txt_inputs = tokenizer([text], padding=True, return_tensors="pt").to(device)
                
                img_feat = model.get_image_features(**img_inputs)
                txt_feat = model.get_text_features(**txt_inputs)
                
                # 정규화 및 코사인 유사도 계산
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
                return (img_feat @ txt_feat.T).item()

            # --- B. Google SigLIP ---
            elif m_type == 'siglip':
                processor = model_pack['processor']
                inputs = processor(text=[text], images=image, return_tensors="pt", padding="max_length").to(device)
                outputs = model(**inputs)
                
                logits = outputs.logits_per_image.item()
                # SigLIP 스케일링 (Logits가 큼)
                return max(0, logits) / 10.0 

            # --- C. Standard CLIP / AltCLIP ---
            else:
                processor = model_pack['processor']
                # AltCLIP은 텍스트 길이가 길 수 있으므로 truncation 옵션 추가
                inputs = processor(
                    text=[text], 
                    images=image, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=77 
                ).to(device)
                
                outputs = model(**inputs)
                
                # CLIP 계열은 보통 Logit Scale이 100이므로 100으로 나눠서 0~1 사이로 맞춤
                return outputs.logits_per_image.item() / 100.0
                
    except Exception as e:
        print(f"Inference Error ({m_type}): {e}")
        return 0.0

# ------------------------------------------------
# 4. 메인 UI 로직
# ------------------------------------------------
with st.sidebar:
    st.header("⚙️ 설정")
    uploaded_file = st.file_uploader("데이터셋 (JSON)", type=['json'])
    default_path = os.path.join("data", "images")
    image_folder = st.text_input("이미지 경로", value=default_path)
    
    st.divider()
    st.write("📋 **참전 모델 목록**")
    for conf in MODELS_CONFIG:
        st.caption(f"- {conf['name']}")

if uploaded_file and image_folder:
    data = json.load(uploaded_file)
    
    if st.button("🔥 아레나 배틀 시작 (Run Benchmark)"):
        
        loaded_models = load_all_models()
        
        if not loaded_models:
            st.error("❌ 로드된 모델이 없습니다. 터미널에서 'pip install transformers==4.46.3'을 실행해보세요.")
            st.stop()
            
        st.success(f"총 {len(loaded_models)}개의 모델이 로드되었습니다.")
            
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # --- 디버깅용 로그창 ---
        log_expander = st.expander("🔍 진행 로그 확인", expanded=True)
        
        for i, item in enumerate(data):
            img_name = item.get('image_filename') or item.get('image_file') or item.get('filename')
            caption = item.get('caption') or item.get('description') or item.get('text')
            
            if not img_name: continue
            if not caption: caption = "unknown"
            
            img_path = os.path.join(image_folder, img_name)
            
            if not os.path.exists(img_path):
                if i < 5: log_expander.warning(f"이미지 못 찾음: {img_name}")
                continue
            
            try:
                image = Image.open(img_path).convert("RGB")
            except:
                continue
            
            row_data = {"Index": i}
            
            for m_name, m_pack in loaded_models.items():
                score = get_similarity_score(m_pack, image, caption)
                row_data[m_name] = score
            
            results.append(row_data)
            
            progress_bar.progress((i + 1) / len(data))
            if i % 10 == 0:
                status_text.text(f"처리 중... {i+1}/{len(data)}")
        
        # --- 결과 처리 ---
        if results:
            df = pd.DataFrame(results)
            st.divider()
            st.subheader("🏆 최종 스코어보드")
            
            # 숫자 데이터만 골라서 평균 내기
            numeric_cols = [col for col in df.columns if col not in ['Index']]
            means = df[numeric_cols].mean().sort_values(ascending=False)
            
            # 그래프 그리기
            fig, ax = plt.subplots(figsize=(10, 5))
            # KoCLIP 강조색
            colors = ['#FF4B4B' if 'KoCLIP' in idx else '#A9A9A9' for idx in means.index]
            sns.barplot(x=means.index, y=means.values, palette=colors, ax=ax)
            
            ax.set_title("Image-Text Alignment Score (Cosine Similarity)", fontsize=14, fontweight='bold')
            ax.set_ylabel("평균 유사도 (0~1)")
            ax.set_ylim(0, 0.6) # Y축 고정 (비교 편하게)
            
            # 막대 위에 점수 표시
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.4f}', 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            st.pyplot(fig)
            
            # 분석 멘트
            winner = means.idxmax()
            st.success(f"🥇 **최종 승자:** {winner}")
            st.info(f"""
            **[결과 해석]**
            * **{winner}** 모델이 평균 유사도 **{means.max():.4f}**를 기록했습니다.
            * 이는 텍스트 설명과 이미지 간의 의미적 연결(Alignment)이 가장 강력함을 의미합니다.
            * 0점(0.0000)이 나온 모델이 있다면 로드 실패이므로 로그를 확인하세요.
            """)
        else:
            st.error("결과가 없습니다. 이미지 경로를 확인해주세요.")