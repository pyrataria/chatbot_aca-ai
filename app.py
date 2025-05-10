import streamlit as st


# Configurações da página
st.set_page_config(
    page_title='Chatbot Aça.AI',
    page_icon='🥤',
    layout='wide'
)

from sklearn.metrics.pairwise import cosine_similarity
from data import dados
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
import nltk
import pt_core_news_sm


nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Carrega o CSS externo
with open(".streamlit/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Inicializa o histórico na sessão
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar
with st.sidebar:
    st.title('Aça.AI')
    st.write('Chatbot para perguntas sobre açaí')
    st.markdown('---')
    st.markdown(
        '<small>Powered by Streamlit, NLTK, spaCy, scikit-learn</small>',
        unsafe_allow_html=True
    )

# Título principal
st.markdown(
    "<h1 style='text-align: center;'>Chatbot Aça.AI</h1>",
    unsafe_allow_html=True
)

# Formulário de input do usuário
col_left, col_form, col_right = st.columns([1, 3, 1])
with col_form:
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([7.5, 1])
        with col1:
            pergunta = st.text_input(
                label="Faça sua pergunta",
                placeholder="Ex: Quais os benefícios do açaí?",
                label_visibility="collapsed",
                key="chat_input"
            )
        with col2:
            enviar = st.form_submit_button("Enviar")

_nlp = pt_core_news_sm.load()

def preprocessar_texto(texto: str) -> list[str]:
    nlp = _nlp
    texto_norm = texto.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(texto_norm, language='portuguese')
    stop_words = set(stopwords.words('portuguese'))
    tokens_filtrados = [t for t in tokens if t not in stop_words]
    doc = nlp(' '.join(tokens_filtrados))
    return [token.lemma_ for token in doc]

perguntas = [
        ' '.join(preprocessar_texto(item['pergunta']))
        for item in dados
    ]

vectorizer = TfidfVectorizer()
vetores = vectorizer.fit_transform(perguntas)

# Processa envio
if enviar and pergunta:
    # Armazena pergunta do usuário
    st.session_state.history.append({
        'type': 'user',
        'text': pergunta
    })

    # Obtém resposta do bot
    lemas = preprocessar_texto(pergunta)
    texto_formatado = ' '.join(lemas)
    vetor_usuario = vectorizer.transform([texto_formatado])
    simil = cosine_similarity(vetor_usuario, vetores)
    idx = simil.argmax()
    resposta = dados[idx]['resposta']

    # Armazena resposta do bot
    st.session_state.history.append({
        'type': 'bot',
        'text': resposta
    })

# Renderiza todo o histórico de chat
for entry in st.session_state.history:
    cls = "user-message" if entry["type"] == "user" else "bot-message"
    st.markdown(f'<div class="{cls}">{entry["text"]}</div>', unsafe_allow_html=True)