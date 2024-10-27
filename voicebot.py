import streamlit as st
from audiorecorder import audiorecorder
import os, tenacity
import numpy as np
from numpy import dot
from numpy.linalg import norm
import ast
import pandas as pd
import requests
import json
from datetime import datetime
from gtts import gTTS
import base64
import openai
from openai.embeddings_utils import get_embedding

# 모든 캐시 데이터 삭제
st.cache_data.clear()  # st.cache를 사용한 경우 st.cache.clear()를 사용하세요

##### 기본 설정 및 API 초기화 #####
openai.api_key = st.secrets["OPENAI_API_KEY"]
api_key = st.secrets["OPENAI_API_KEY"]
MODEL_WHISPER = "whisper-1"
MODEL_GPT = "gpt-4o-mini-2024-07-18"

# 데이터 파일 경로
folder_path = './data'
file_name = 'embedding.csv'
file_path = os.path.join(folder_path, file_name)

# 문서 임베딩 파일 불러오기
if os.path.isfile(file_path):
    df = pd.read_csv(file_path)
    df['embedding'] = df['embedding'].apply(ast.literal_eval)
else:
    folder_path = './data' # data 폴더 경로
    txt_files = [file for file in os.listdir(folder_path) if file.endswith('.txt')]  # txt 파일 목록

    data = []
    for file in txt_files:
        txt_file_path = os.path.join(folder_path, file)
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            text = f.read() # 파일 내용 읽기
            data.append(text)

    df = pd.DataFrame(data, columns=['text'])

    # 데이터프레임의 text 열에 대해서 embedding을 추출
    df['embedding'] = df.apply(lambda row: get_embedding(
        row.text,
        engine="text-embedding-ada-002"
    ), axis=1)
    df.to_csv(file_path, index=False, encoding='utf-8-sig')

##### 기능 함수 정의 #####

# STT: 오디오 파일에서 텍스트 추출
def stt(audio):
    filename = 'input.mp3'
    audio.export(filename, format="mp3")
    with open(filename, "rb") as audio_file:
        transcript = openai.Audio.transcribe(MODEL_WHISPER, audio_file)
    os.remove(filename)
    return transcript.text

# GPT에게 질문
def ask_gpt(messages):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": MODEL_GPT,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 1000  # 답변을 1000자 이내로 제한
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    content = json.loads(response.content)
    return content['choices'][0]['message']['content'].strip()

# TTS: 텍스트를 음성으로 변환
def tts(response):
    filename = "output.mp3"
    tts = gTTS(text=response, lang="ko")
    tts.save(filename)

    with open(filename, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        audio_html = f"""
            <audio autoplay="True">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(audio_html, unsafe_allow_html=True)
    
    os.remove(filename)

# 코사인 유사도 계산 함수
def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))

# 입력 질문과 가장 유사한 문서 찾기
def return_answer_candidate(df, query):
    query_embedding = get_embedding(
        query,
        engine="text-embedding-ada-002"
    )
    df["similarity"] = df.embedding.apply(lambda x: cos_sim(np.array(x), np.array(query_embedding)))
    top_three_doc = df.sort_values("similarity", ascending=False).head(3)
    return top_three_doc

# GPT를 위한 프롬프트 생성
def create_prompt(df, query):
    # 상위 3개의 유사한 문서를 찾아 result에 저장
    result = return_answer_candidate(df, query)
    
    # 시스템 메시지에 유사한 상위 3개의 문서 내용을 포함
    system_role = f"""You are an artificial intelligence language model named "정채기" that specializes in summarizing \
    and answering documents about Seoul's youth policy, developed by developers 사용자1 and 사용자2.
    You need to take a given document and return a very detailed summary of the document in the query language.
    Here are the document: 
            doc 1 :""" + str(result.iloc[0]['text']) + """
            doc 2 :""" + str(result.iloc[1]['text']) + """
            doc 3 :""" + str(result.iloc[2]['text']) + """
    You must return in Korean. Return a accurate answer based on the document.
    """
    
    # 사용자의 질문을 포함한 메시지
    user_content = f"""User question: "{str(query)}". """

    # GPT에게 전달할 messages 리스트 생성
    messages = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": user_content}
    ]
    
    return messages

# 채팅 기록 시각화
def display_chat():
    for sender, time, message in st.session_state["chat"]:
        if sender == "user":
            st.write(f'<div style="display:flex;align-items:center;"><div style="background-color:#007AFF;color:white;border-radius:12px;padding:8px 12px;margin-right:8px;">{message}</div><div style="font-size:0.8rem;color:gray;">{time}</div></div>', unsafe_allow_html=True)
        else:
            st.write(f'<div style="display:flex;align-items:center;justify-content:flex-end;"><div style="background-color:lightgray;border-radius:12px;padding:8px 12px;margin-left:8px;">{message}</div><div style="font-size:0.8rem;color:gray;">{time}</div></div>', unsafe_allow_html=True)

# 세션 상태 초기화 함수
def init_session_state():
    if "chat" not in st.session_state:
        st.session_state["chat"] = []
    if "messages" not in st.session_state:
        st.session_state["messages"] = [] # [{"role": "system", "content": "You are a thoughtful assistant. Respond to all input in less than 1000 and answer in korea"}]
    if "check_reset" not in st.session_state:
        st.session_state["check_reset"] = False

##### 메인 함수 #####

def main():

    st.image('./images/ddc.jpg')

    init_session_state()

    col1, col2 = st.columns(2)

    # 초기화 버튼이 눌렸을 경우 우선적으로 세션 상태 초기화
    if st.button(label="초기화"):
        st.session_state["chat"] = []
        st.session_state["messages"] = [] # [{"role": "system", "content": "You are a thoughtful assistant. Respond to all input in less than 1000 and answer in korea"}]
        st.session_state["check_reset"] = True  # 대답하지 않고 바로 초기화
    
    # 음성 입력 및 질문 처리
    with col1:
        st.subheader("질문하기")
        audio = audiorecorder("클릭하여 녹음하기", "녹음중...")
        if (audio.duration_seconds > 0) and not st.session_state["check_reset"]:
            st.audio(audio.export().read())
            question = stt(audio)
            now = datetime.now().strftime("%H:%M")
            st.session_state["chat"].append(("user", now, question))

            # 사용자 질문을 기반으로 프롬프트 생성 및 messages에 추가
            st.session_state["messages"] = create_prompt(df, question)
            # st.session_state["messages"].append({"role": "user", "content": question})

    # 질문에 대한 답변 처리 및 시각화
    with col2:
        st.subheader("질문/답변")
        if (audio.duration_seconds > 0) and not st.session_state["check_reset"]:
            response = ask_gpt(st.session_state["messages"])
            now = datetime.now().strftime("%H:%M")
            st.session_state["chat"].append(("bot", now, response))
            st.session_state["messages"].append({"role": "system", "content": response})
            display_chat()
            tts(response)
        elif st.session_state["check_reset"]:
            st.session_state["check_reset"] = False  # 초기화 후 다시 처리 가능하도록 플래그 해제

if __name__ == "__main__":
    main()
