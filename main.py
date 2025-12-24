# GenAI - VERSÃO INTERATIVA COM CHATGPT

"""
Pipeline didático: Classic ML + Optimization + Deep Learning + RAG/LLM

- Classic ML: Regressão Logística com GridSearchCV
- Optimization: tuning de hiperparâmetros (C, penalty, solver)
- Deep Learning: rede neural densa simples (Keras/TensorFlow)
- LLMs/RAG: recuperação de contexto com sentence-transformers
- DS code: código modular, funções, logging, main()
"""

# =========================
# 1. IMPORTS
# =========================

from typing import List, Tuple
from google.colab import drive
from openai import OpenAI

drive.mount('/content/drive', force_remount=True)

import logging
import numpy as np
import pandas as pd
import keras

import os
os.environ["OPENAI_API_KEY"] = "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"



from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from keras import layers

from sentence_transformers import SentenceTransformer, util  # para RAG simples

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =========================
# 2. LOGGING BÁSICO           # DS CODE
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# =========================
# 3. CARGA E PREPARAÇÃO DE DADOS
#    (Classic DS code: função bem definida, recebe e retorna)
# =========================

PATH = "/content/drive/MyDrive/Colab Notebooks/GenAI/students.csv"

def load_student_data(path: str = PATH) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Carregamento dados de estudantes.

    Espera um CSV com:
      - colunas numéricas (ex.: idade, faltas, notas, etc.)
      - coluna 'target' binária (0 = não evadeu, 1 = evadeu)
    """
    df = pd.read_csv(path)
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


# =========================
# 4. CLASSIC ML + OPTIMIZATION
# =========================

def fit_classic_ml_model(X: pd.DataFrame, y: pd.Series):
    """
    Treina um modelo clássico (Regressão Logística) com otimização de hiperparâmetros.

    - Usa Pipeline (StandardScaler + LogisticRegression)
    - Usa GridSearchCV para encontrar melhor C e penalty
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),                #PADRONIZA OS DADOS
        ("clf", LogisticRegression(max_iter=500))    # MODELO CLASSIFICATÓRIO DOS DADOS
    ])

    param_grid = {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"]
    }

    grid = GridSearchCV(                    # MELHORES VALORES
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="f1",     # SCORE F1
        n_jobs=-1
    )

    logging.info("Treinando modelo clássico com GridSearchCV...")
    grid.fit(X_train, y_train)

    logging.info("Melhores parâmetros (Classic ML): %s", grid.best_params_)

    y_pred = grid.predict(X_test)
    logging.info("Relatório de classificação - Classic ML:\n%s",
                 classification_report(y_test, y_pred))

    return grid, (X_test, y_test)


# =========================
# 5. DEEP LEARNING COM KERAS/TENSORFLOW     ## REDE NEURAL
# =========================

def build_deep_learning_model(input_dim: int) -> keras.Model:    #REDE SEQUENCIAL COM 3 CAMADAS
    """
    Cria uma rede neural densa simples para classificação binária.
    """
    model = keras.Sequential([
        layers.Dense(32, activation="relu", input_shape=(input_dim,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")  # saída binária
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def fit_deep_learning_model(X: pd.DataFrame, y: pd.Series): # TREINO E TESTE DA REDE NEURAL
    """
    Treina a rede neural em cima dos mesmos dados.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.2, random_state=42, stratify=y
    )

    model = build_deep_learning_model(input_dim=X_train.shape[1])

    logging.info("Treinando modelo de Deep Learning...")
    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=32,
        verbose=0
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    logging.info("Desempenho Deep Learning - loss: %.4f - acc: %.4f", loss, acc)

    # retorna modelo e um exemplo de previsão sobre o teste
    sample_probs = model.predict(X_test[:5])
    sample_preds = (sample_probs > 0.5).astype(int).flatten()
    return model, (X_test[:5], sample_preds)


# =========================
# 6. BASE DE CONHECIMENTO PARA RAG
#    (documentos institucionais / explicações de features)   #LLM
# =========================

def build_rag_index() -> Tuple[SentenceTransformer, np.ndarray, List[str]]: # GERA EMBEDDINGS
    """
    Cria um índice vetorial simples com textos de contexto.
    Aqui seria onde você colocaria:
      - Manual do aluno
      - Normas de evasão
      - Documentação do modelo
      - Descrição das variáveis (features)
    """
    # Exemplos fictícios de documentos
    docs = [
        "A evasão escolar está frequentemente associada a alto número de faltas e baixo desempenho acadêmico.",
        "Estudantes com bom engajamento em atividades complementares tendem a apresentar menor risco de evasão.",
        "A instituição oferece programas de apoio financeiro e pedagógico para reduzir a evasão.",
        "Faltas acima de 25% da carga horária são consideradas fator crítico de risco para evasão.",
        "Participação em projetos de pesquisa e extensão é um indicador positivo de permanência."
    ]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(docs, convert_to_tensor=True)

    return model, embeddings, docs


def retrieve_context(    # RETRIEVAL AUGMENTED CONTEXT
    question: str,
    emb_model: SentenceTransformer,
    doc_embeddings,
    docs: List[str],
    top_k: int = 2
) -> List[str]:
    """
    Recupera os documentos mais relevantes para a pergunta (RAG).
    """
    q_emb = emb_model.encode(question, convert_to_tensor=True)
    hits = util.semantic_search(q_emb, doc_embeddings, top_k=top_k)[0]
    return [docs[h["corpus_id"]] for h in hits]


def build_rag_prompt(  # PROMPT DO CONTEXTO RECUPERADO
    question: str,
    classic_pred: int,
    dl_pred: int,
    retrieved_docs: List[str]
) -> str:
    """
    Monta um prompt para um LLM usando as saídas dos modelos + contexto recuperado.
    (Aqui você chamaria uma API de LLM de verdade)
    """
    context_text = "\n\n".join(retrieved_docs)
    prompt = f"""
Você é um assistente que explica risco de evasão escolar.

Contexto recuperado:
{context_text}

Previsões dos modelos:
- Modelo Clássico (Regressão Logística): risco_de_evasao = {classic_pred}
- Modelo Deep Learning (Rede Neural): risco_de_evasao = {dl_pred}

Pergunta do usuário:
{question}

Explique de forma clara, em linguagem acessível, por que o estudante pode estar em risco
e quais ações a instituição pode tomar para reduzir esse risco.
"""
    return prompt


def call_llm(prompt: str) -> str:
    """
    Chamada real a um LLM (OpenAI GPT-4o, por exemplo).
    Recebe um prompt em texto e devolve a resposta do modelo.
    """
    response = client.chat.completions.create(
        model="gpt-4o",  # ou outro modelo disponível na sua conta
        messages=[
            {"role": "system", "content": "You are an educational advisor that explains student dropout risk in clear language."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


# =========================
# 7. FUNÇÃO PRINCIPAL (ORQUESTRA TUDO)
# =========================

def main():
    # 1) Carregar dados
    X, y = load_student_data()

    # 2) Classic ML + Optimization
    classic_model, (X_test_classic, y_test_classic) = fit_classic_ml_model(X, y)

    # Pegar um exemplo do teste para explicar com o LLM
    x_ex = X_test_classic.iloc[[0]]
    classic_pred = int(classic_model.predict(x_ex)[0])

    # 3) Deep Learning
    dl_model, (X_dl_sample, dl_sample_preds) = fit_deep_learning_model(X, y)
    dl_pred = int(dl_sample_preds[0])

    # 4) RAG: índice e contexto
    emb_model, doc_embeddings, docs = build_rag_index()
    question = "Por que este estudante foi classificado como alto risco de evasão?"
    retrieved_docs = retrieve_context(question, emb_model, doc_embeddings, docs)

    # 5) Montar prompt para LLM
    prompt = build_rag_prompt(question, classic_pred, dl_pred, retrieved_docs)

    print("\n================ PROMPT FINAL GERADO ================")
    print(prompt)
    print("=====================================================\n")

    llm_answer = call_llm(prompt)
    print("RESPOSTA DO LLM:\n", llm_answer)



if __name__ == "__main__":
   main()
