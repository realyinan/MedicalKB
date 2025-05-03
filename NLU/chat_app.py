import streamlit as st
from modules import *
from utils.json_utils import *


def main():
    st.title("欢迎访问小康医疗智能问答系统")

    # 初始化会话状态，如果没有则创建
    if 'history' not in st.session_state:
        st.session_state.history = []

    # 显示对话历史
    for chat in st.session_state.history:
        if chat['role'] == 'user':
            with st.chat_message("user"):
                st.markdown(chat['content'])
        else:
            with st.chat_message("assistant"):
                st.markdown(chat['content'])
    user_input = st.chat_input("Chat with 小康: ")
    # user_input接收用户的输入
    if user_input:
        # 在页面上显示用户的输入
        with st.chat_message("user"):
            st.markdown(user_input)

        # 判断用户输入是否为闲聊意图
        user_intent = classifier(user_input)
        # print(f'user_intent-->{user_intent}')
        if user_intent in ["greet", "goodbye", "deny", "isbot"]:
            # 闲聊意图时，直接随机返回设计好的答案，
            response = gossip_robot(user_intent)
        elif user_intent == "accept":
            # 澄清意图时，进行回复
            reply = load_user_dialogue_context()
            response = reply.get("choice_answer")
        else:
            # 为诊断意图时，返回图谱答案
            reply = medical_robot(user_input)
            # print(f'reply11111-->{reply}')
            if reply["slot_values"]:
                dump_user_dialogue_context(reply)
            response = reply.get("replay_answer")
        # 将用户的输入加入历史
        st.session_state.history.append({"role": "user", "content": user_input})
        # 在页面上显示模型生成的回复
        with st.chat_message("assistant"):
            st.markdown(response)

        # 将模型的输出加入到历史信息中
        st.session_state.history.append({"role": "assistant", "content": response})

        # 只保留十轮对话，这个可根据自己的情况设定
        if len(st.session_state.history) > 20:
            st.session_state.history = st.session_state.history[-20:]

if __name__ == "__main__":
    main()
