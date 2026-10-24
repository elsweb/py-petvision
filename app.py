from pathlib import Path
import re
import time
import math
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image

import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import classification_report

from config import (
    MIN_SPECIES_SAMPLES,
    MIN_BREED_SAMPLES,
    SPECIES_DIR,
    BREED_DIR,
)
from database import (
    init_db,
    add_image,
    add_prediction,
    add_feedback,
    stats,
    feedback_rows,
)
from inference import detect, DEVICE
from storage import save_image
from training import trainer


st.set_page_config(
    page_title="PetVision Local",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# Tailwind + CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <script src="https://cdn.tailwindcss.com"></script>

    <script>
      tailwind.config = {
        darkMode: 'class',
        theme: {
          extend: {
            boxShadow: {
              glow: '0 0 50px rgba(99,102,241,.14)'
            }
          }
        }
      }
    </script>

    <style>
      html, body, [data-testid="stAppViewContainer"] {
        background:
          radial-gradient(
            circle at top left,
            rgba(99,102,241,.12),
            transparent 32rem
          ),
          #080b14 !important;
      }

      [data-testid="stHeader"] {
        background: transparent !important;
      }

      .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
      }

      .pet-card {
        background: rgba(15,23,42,.68);
        border: 1px solid rgba(148,163,184,.13);
        border-radius: 24px;
        padding: 22px;
        box-shadow: 0 0 50px rgba(99,102,241,.07);
      }

      .pet-result {
        background: rgba(2,6,23,.72);
        border: 1px solid rgba(148,163,184,.12);
        border-radius: 18px;
        padding: 18px;
      }

      .pet-muted {
        color: #64748b;
        font-size: 13px;
      }

      .pet-title {
        color: #f8fafc;
        font-weight: 800;
      }

      .pet-accent {
        color: #a5b4fc;
      }
      
      .pet-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
      }

      /* Cards de Métricas Customizados */
      .metric-card {
          background: rgba(15, 23, 42, 0.5);
          border: 1px solid rgba(148, 163, 184, 0.1);
          border-radius: 12px;
          padding: 12px 16px;
          text-align: center;
      }
      .metric-title {
          font-size: 12px;
          color: #94a3b8;
          text-transform: uppercase;
          letter-spacing: 0.05em;
          margin-bottom: 4px;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 4px;
      }
      .metric-value {
          font-size: 24px;
          font-weight: 900;
          color: #f8fafc;
      }

      div[data-testid="stFileUploader"] {
        border-radius: 20px;
      }

      div[data-testid="stButton"] > button {
        border-radius: 14px;
        min-height: 46px;
        font-weight: 700;
      }

      .stProgress > div > div > div > div {
        border-radius: 999px;
      }

      section[data-testid="stSidebar"] {
        background: #080b14;
      }
      
      /* Ajuste para remover espaçamentos fantasmas do Streamlit */
      div[data-testid="stMarkdownContainer"] > div {
         margin-bottom: 0px;
      }

      /* Estilo para o ícone de interrogação */
      .help-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background-color: rgba(148,163,184,0.2);
        color: #94a3b8;
        font-size: 10px;
        font-weight: bold;
        cursor: help;
        margin-left: 4px;
        text-decoration: none;
      }
      .help-icon:hover {
        background-color: rgba(148,163,184,0.4);
        color: #f8fafc;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Inicialização
# ---------------------------------------------------------------------------

init_db()

if "result" not in st.session_state:
    st.session_state.result = None
if "inference_time" not in st.session_state:
    st.session_state.inference_time = 0.0

if "image_id" not in st.session_state:
    st.session_state.image_id = None

if "image_path" not in st.session_state:
    st.session_state.image_path = None

if "species_edit" not in st.session_state:
    st.session_state.species_edit = "dog"

if "breed_edit" not in st.session_state:
    st.session_state.breed_edit = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pretty(value):
    if not value:
        return "—"
    return str(value).replace("_", " ").title()


def confidence(value):
    return f"{float(value) * 100:.1f}%"


def normalize_label(value):
    value = (value or "").strip().lower()
    value = re.sub(r"[^\w]+", "_", value, flags=re.UNICODE)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def progress_card(label, value):
    st.caption(f"{label} · {int(value)}%")
    st.progress(max(0.0, min(1.0, value / 100)))


def calc_entropy(probs):
    """Calcula a entropia de uma lista de probabilidades."""
    return -sum(p * math.log2(p) for p in probs if p > 0)


# ---------------------------------------------------------------------------
# POPUP DE MÉTRICAS DETALHADAS
# ---------------------------------------------------------------------------

@st.dialog("Relatório de Classificação Detalhado", width="large")
def mostrar_popup_metricas():
    st.markdown("### Relatório de Classificação por Classe")
    st.caption("Mostra o desempenho do modelo nos dados de validação (dados que ele não viu no treino).")
    
    current_stats = stats()
    # Pega apenas o último treino de cada camada que tenha relatório salvo
    last_species_run = next((r for r in current_stats["runs"] if r["layer"] == "species" and r.get("report")), None)
    last_breed_run = next((r for r in current_stats["runs"] if r["layer"] == "breed" and r.get("report")), None)

    if not last_species_run and not last_breed_run:
        st.warning("Nenhum modelo treinado com relatório válido encontrado.")
        return

    opcoes = []
    if last_species_run: opcoes.append("Espécie (Gato/Cachorro)")
    if last_breed_run: opcoes.append("Raça")
        
    camada = st.radio("Selecione a camada:", opcoes, horizontal=True)
    
    # Pega o dicionário do relatório salvo no banco
    run_selecionado = last_species_run if camada.startswith("Espécie") else last_breed_run
    metrics_data = json.loads(run_selecionado["report"]) 

    # --- EXIBIR TABELA DE RELATÓRIO ---
    df_report = pd.DataFrame(metrics_data).transpose()
    
    df_report = df_report.rename(columns={
        "precision": "Precisão",
        "recall": "Recall",
        "f1-score": "F1-Score",
        "support": "Suporte"
    })
    
    st.dataframe(
        df_report.style.format({
            "Precisão": "{:.2f}",
            "Recall": "{:.2f}",
            "F1-Score": "{:.2f}",
            "Suporte": "{:.0f}"
        }, na_rep="-"), 
        use_container_width=True
    )

    if st.button("Fechar", use_container_width=True):
        st.rerun()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="flex items-center justify-between gap-4 mb-8">
      <div>
        <div class="pet-accent text-xs font-bold uppercase tracking-[.22em]">
          Vision Transformer
        </div>
        <h1 class="text-3xl sm:text-4xl font-black text-white mt-1">
          PetVision <span class="text-indigo-400">Local</span>
        </h1>
        <p class="text-slate-400 mt-1">
          Detecção em duas camadas + aprendizado com feedback humano.
        </p>
      </div>

      <div class="hidden sm:block rounded-2xl border border-emerald-500/20
                  bg-emerald-500/5 px-4 py-3">
        <div class="text-xs text-slate-500">Execução</div>
        <div class="text-sm font-bold text-emerald-300">
          ● 100% local
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Upload + detecção
# ---------------------------------------------------------------------------

left, right = st.columns([1.05, .95], gap="large")

with left:
    st.markdown("#### Imagem")
    st.caption("Envie JPG, PNG ou WEBP")

    uploaded = st.file_uploader(
        "Arraste uma imagem ou selecione um arquivo",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        st.image(uploaded, use_container_width=True)

        if st.button("🐾 Detectar animal e raça", use_container_width=True, type="primary"):
            try:
                progress_card("Salvando upload", 20)
                filename, path = save_image(uploaded)
                image_id = add_image(filename, path)

                st.session_state.image_id = image_id
                st.session_state.image_path = path

                progress_card("Executando Camada 1 + Camada 2", 55)
                
                start_time = time.time()
                result = detect(path)
                st.session_state.inference_time = time.time() - start_time

                add_prediction(image_id, result)
                st.session_state.result = result

                progress_card("Detecção concluída", 100)
                st.success("Imagem analisada com sucesso.")

            except Exception as exc:
                st.error(str(exc))


with right:
    st.markdown(
        """
        <div class="pet-card">
          <div class="flex items-center gap-3">
            <div class="text-2xl">🧠</div>
            <div>
              <div class="pet-title text-lg">
                Resultado em 2 camadas
              </div>
              <div class="pet-muted">
                Inferência local com Transformers
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    result = st.session_state.result

    if not result:
        st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)
        st.info("Envie uma imagem para iniciar a detecção.")
    else:
        species_top = result["species_top"]
        is_pet = species_top["label"].lower() in ["cat", "dog"]

        st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)

        # CARD CAMADA 1 ENRIQUECIDO
        top_species = result.get("species", [])
        second_species = top_species[1] if len(top_species) > 1 else None
        
        species_html = f"""
            <div class="pet-result">
              <div class="flex justify-between items-start">
                  <div>
                      <div class="pet-muted uppercase tracking-wider mb-1 flex items-center">
                        Camada 1 · Espécie
                        <span class="help-icon" title="Probabilidade do modelo de que o animal pertence a esta espécie.">?</span>
                      </div>
                      <div class="text-2xl font-black text-white">
                        {pretty(species_top["label"])}
                      </div>
                  </div>
                  <div class="text-right">
                      <div class="text-indigo-300 font-bold text-xl">
                        {confidence(species_top["probability"])}
                      </div>
                      <div class="pet-muted text-xs mt-1">Confiança</div>
                  </div>
              </div>
        """
        
        if second_species:
            species_html += f"""
              <div class="mt-3 pt-3 border-t border-slate-700/50 flex justify-between items-center">
                  <span class="pet-muted text-xs flex items-center">
                    2ª opção: {pretty(second_species['label'])}
                    <span class="help-icon" title="Segunda espécie mais provável segundo o modelo.">?</span>
                  </span>
                  <span class="text-slate-400 text-xs font-bold">{confidence(second_species['probability'])}</span>
              </div>
            """
            
        species_html += "</div>"
        st.markdown(species_html, unsafe_allow_html=True)

        # CARD CAMADA 2 ENRIQUECIDO (Se for pet)
        if is_pet:
            breed_top = result["breed_top"]
            
            st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)
            
            breed_html = f"""
                <div class="pet-result">
                  <div class="flex justify-between items-start">
                      <div>
                          <div class="pet-muted uppercase tracking-wider mb-1 flex items-center">
                            Camada 2 · Raça
                            <span class="help-icon" title="Probabilidade do modelo de que o animal pertence a esta raça específica.">?</span>
                          </div>
                          <div class="text-2xl font-black text-white">
                            {pretty(breed_top["label"])}
                          </div>
                      </div>
                      <div class="text-right">
                          <div class="text-indigo-300 font-bold text-xl">
                            {confidence(breed_top["probability"])}
                          </div>
                          <div class="pet-muted text-xs mt-1">Confiança</div>
                      </div>
                  </div>
            """
            
            top_breeds = result.get("breed", [])[:3]
            if len(top_breeds) > 1:
                breed_html += '<div class="mt-3 pt-3 border-t border-slate-700/50 space-y-1">'
                breed_html += '<div class="pet-muted text-xs mb-2 uppercase tracking-wider">Top 3 Raças</div>'
                
                for idx, item in enumerate(top_breeds, start=1):
                    breed_html += f'<div class="flex justify-between items-center text-sm"><span class="text-slate-300">{idx}. {pretty(item["label"])}</span><span class="text-slate-400 font-medium">{confidence(item["probability"])}</span></div>'
                
                breed_html += '</div>'
                
            breed_html += "</div>"
            st.markdown(breed_html, unsafe_allow_html=True)
        else:
            st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)
            st.info("A camada de raça não é aplicada para esta espécie.")

        # ---------------------------------------------------------
        # MÉTRICAS POR IMAGEM (Gap e Entropia)
        # ---------------------------------------------------------
        st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)
        
        gap_species = 0.0
        if len(top_species) > 1:
            gap_species = top_species[0]["probability"] - top_species[1]["probability"]
            
        probs_species = [item["probability"] for item in top_species]
        entropy_val = calc_entropy(probs_species)
        
        if gap_species > 0.5:
            gap_status = "Alta Certeza"
            gap_color = "text-emerald-400"
        elif gap_species > 0.15:
            gap_status = "Moderada"
            gap_color = "text-amber-400"
        else:
            gap_status = "Baixa (Dúvida)"
            gap_color = "text-rose-400"

        metrics_html = f"""
            <div class="grid grid-cols-2 gap-3">
                <div class="bg-slate-800/30 p-3 rounded-xl border border-slate-700/50">
                    <div class="text-xs text-slate-400 uppercase tracking-wider mb-1 flex items-center">
                        Gap (Top1 - Top2)
                        <span class="help-icon" title="A diferença entre a probabilidade da 1ª e da 2ª opção. Valores altos indicam que o modelo está confiante na escolha.">?</span>
                    </div>
                    <div class="text-lg font-bold {gap_color}">{gap_species*100:.1f}%</div>
                    <div class="text-[10px] text-slate-500">{gap_status}</div>
                </div>
                <div class="bg-slate-800/30 p-3 rounded-xl border border-slate-700/50">
                    <div class="text-xs text-slate-400 uppercase tracking-wider mb-1 flex items-center">
                        Entropia
                        <span class="help-icon" title="Medida matemática de incerteza. Quanto menor o valor, mais confiante o modelo está.">?</span>
                    </div>
                    <div class="text-lg font-bold text-white">{entropy_val:.3f}</div>
                    <div class="text-[10px] text-slate-500">Incerteza da predição</div>
                </div>
            </div>
        """
        st.markdown(metrics_html, unsafe_allow_html=True)

        # METADADOS DA INFERÊNCIA (Tempo, Dispositivo)
        st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)
        inf_time = st.session_state.get("inference_time", 0.0)
        
        st.markdown(f"""
            <div class="flex justify-between items-center bg-slate-800/30 p-3 rounded-xl border border-slate-700/50">
                <div class="flex items-center gap-2">
                    <span class="text-slate-400 text-xs uppercase tracking-wider">Dispositivo</span>
                    <span class="pet-badge bg-indigo-500/20 text-indigo-300">{result['device'].upper()}</span>
                </div>
                <div class="flex items-center gap-2">
                    <span class="text-slate-400 text-xs uppercase tracking-wider flex items-center">
                        Tempo
                        <span class="help-icon" title="Tempo total que o modelo levou para processar a imagem.">?</span>
                    </span>
                    <span class="pet-badge bg-emerald-500/20 text-emerald-300">{inf_time:.3f}s</span>
                </div>
            </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Correção humana
# ---------------------------------------------------------------------------

if st.session_state.result:
    st.divider()

    st.markdown(
        """
        <div class="pet-card">
          <div class="pet-title text-lg">
            ✏️ Corrigir / enriquecer
          </div>
          <div class="pet-muted mt-1">
            A correção será armazenada e poderá ser usada no retreinamento.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    result = st.session_state.result
    species_label = result["species_top"]["label"].lower()
    species_options = ["cat", "dog"]
    
    species_index = species_options.index(species_label) if species_label in species_options else None 

    species_edit = st.selectbox(
        "Espécie correta",
        species_options,
        index=species_index,
        format_func=lambda x: "Gato" if x == "cat" else "Cachorro",
        placeholder="Selecione a espécie correta...",
    )

    if species_edit in ["cat", "dog"]:
        breed_default = result["breed_top"]["label"]
        known_breeds = sorted({item["label"] for item in result["breed"]} | {breed_default})
        NEW_BREED_OPTION = "➕ Outra raça (digitar)"
        breed_options = known_breeds + [NEW_BREED_OPTION]

        breed_choice = st.selectbox(
            "Raça correta",
            breed_options,
            index=0,
            format_func=lambda x: x if x == NEW_BREED_OPTION else pretty(x),
        )

        extra_samples = []
        include_current = True

        if breed_choice == NEW_BREED_OPTION:
            breed_custom = st.text_input(
                "Nome da raça",
                placeholder="Ex.: Pastor Alemão, SRD, Maine Coon...",
                help="Digite o nome da raça.",
            )
            breed_edit = normalize_label(breed_custom)
            if breed_custom.strip():
                st.caption(f"Label que será salvo: `{breed_edit}`")
            include_current = st.checkbox("Incluir também a imagem detectada acima", value=True)
            extra_samples = st.file_uploader(
                "Imagens de amostra para o novo rótulo (opcional)",
                type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=True,
                key="new_breed_samples",
            )
            if extra_samples:
                st.caption(f"{len(extra_samples)} imagem(ns) adicional(is) serão rotuladas como `{breed_edit or '—'}`.")
        else:
            breed_edit = breed_choice

        if st.button("💾 Salvar correção para treinamento", use_container_width=True):
            if not breed_edit:
                st.warning("Informe o nome da raça antes de salvar.")
            else:
                saved = 0
                failed = 0
                if include_current and st.session_state.image_id:
                    try:
                        add_feedback(st.session_state.image_id, species_edit, breed_edit)
                        saved += 1
                    except Exception as exc:
                        failed += 1
                        st.error(f"Falha na imagem detectada: {exc}")

                if extra_samples:
                    bar = st.progress(0)
                    status = st.empty()
                    total = len(extra_samples)

                    for index, sample in enumerate(extra_samples, start=1):
                        status.caption(f"Salvando {index}/{total} · {sample.name}")
                        try:
                            filename, path = save_image(sample)
                            new_id = add_image(filename, path)
                            add_feedback(new_id, species_edit, breed_edit)
                            saved += 1
                        except Exception as exc:
                            failed += 1
                            st.error(f"Falha em {sample.name}: {exc}")
                        bar.progress(index / total)

                    status.empty()
                    bar.empty()

                if saved:
                    st.success(f"{saved} imagem(ns) salva(s) no dataset local com o rótulo `{breed_edit}`.")
                if failed:
                    st.warning(f"{failed} imagem(ns) falharam ao salvar.")
    else:
        st.info("A correção de raça só está disponível para cães e gatos. Selecione a espécie correta acima.")


# ---------------------------------------------------------------------------
# Dataset e Métricas Globais
# ---------------------------------------------------------------------------

st.divider()
current_stats = stats()

col_dataset, col_metrics = st.columns([1, 1], gap="large")

with col_dataset:
    st.markdown(
        """
        <div class="pet-card h-full">
          <div class="pet-title text-lg">
            📚 Dataset local
          </div>
          <div class="pet-muted mt-1">
            Imagens confirmadas manualmente e disponíveis para treinamento.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Rotulados", current_stats["total"], help="Total de imagens já corrigidas e salvas por você.")
    with c2:
        st.metric("Min. Espécie", MIN_SPECIES_SAMPLES, help="Número mínimo de imagens necessárias para treinar a Camada 1 (Gato/Cachorro).")
    with c3:
        st.metric("Min. Raça", MIN_BREED_SAMPLES, help="Número mínimo de imagens por raça necessárias para treinar a Camada 2.")

with col_metrics:
    c_header, c_btn = st.columns([4, 1])
    
    with c_header:
        st.markdown(
            """
            <div class="pet-title text-lg">
              📈 Métricas Globais do Modelo
            </div>
            <div class="pet-muted mt-1">
              Desempenho no último treinamento (Dados de Validação).
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with c_btn:
        # Verifica se há algum relatório salvo nos últimos runs
        tem_relatorio = any(r.get("report") for r in current_stats["runs"])
        if tem_relatorio:
            if st.button("➕", help="Ver Relatório Detalhado", use_container_width=True):
                mostrar_popup_metricas()

    # Pega o último treino que tem accuracy e report
    last_run = next((r for r in current_stats["runs"] if r.get("report")), None)

    if last_run:
        tipo_modelo = "Raça" if last_run["layer"] == "breed" else "Espécie"
        st.caption(f"Exibindo métricas do modelo treinado: **{tipo_modelo}**")
        
        # Puxa os dados do JSON salvo no banco
        report_dict = json.loads(last_run["report"])
        
        # No classification_report, a acurácia fica na chave 'accuracy'
        acc = report_dict.get("accuracy", 0.0)
        # As médias ponderadas ficam em 'weighted avg'
        weighted = report_dict.get("weighted avg", {})
        prec = weighted.get("precision", 0.0)
        rec = weighted.get("recall", 0.0)
        f1 = weighted.get("f1-score", 0.0)

        acc_str = f"{acc*100:.1f}%"
        prec_str = f"{prec*100:.1f}%"
        rec_str = f"{rec*100:.1f}%"
        f1_str = f"{f1*100:.1f}%"

        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Acurácia <span class="help-icon" title="Porcentagem geral de acertos.">?</span></div>
                    <div class="metric-value">{acc_str}</div>
                </div>
            """, unsafe_allow_html=True)
            
        with m2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Precisão <span class="help-icon" title="Média ponderada de precisão.">?</span></div>
                    <div class="metric-value">{prec_str}</div>
                </div>
            """, unsafe_allow_html=True)
            
        with m3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Recall <span class="help-icon" title="Média ponderada de recall.">?</span></div>
                    <div class="metric-value">{rec_str}</div>
                </div>
            """, unsafe_allow_html=True)
            
        with m4:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">F1-Score <span class="help-icon" title="Média harmônica entre precisão e recall.">?</span></div>
                    <div class="metric-value">{f1_str}</div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Nenhum treinamento com métricas registradas foi encontrado. Treine um modelo para visualizar.")


# ---------------------------------------------------------------------------
# Treinamento
# ---------------------------------------------------------------------------

st.divider()
st.markdown(
    """
    <div class="pet-card">
      <div class="pet-title text-lg">
        🔄 Retreinamento local
      </div>
      <div class="pet-muted mt-1">
        Fine-tuning do Vision Transformer com os dados corrigidos.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

train_col1, train_col2 = st.columns(2)

with train_col1:
    layer = st.selectbox(
        "Camada",
        ["species", "breed"],
        format_func=lambda x: "Camada 1 · Gato/Cachorro" if x == "species" else "Camada 2 · Raça",
    )

with train_col2:
    epochs = st.selectbox("Épocas", [1, 2, 3, 5], index=1)

state = trainer.state()

if state["running"]:
    progress_card(f"Treinando {state['layer']}", state["progress"])
    st.info(state["message"])
    time.sleep(.25)
    st.rerun()

if st.button("🚀 Retreinar modelo", use_container_width=True, disabled=state["running"]):
    try:
        trainer.start(layer, epochs)
        st.rerun()
    except Exception as exc:
        st.error(str(exc))

state = trainer.state()

if state["result"] and not state["running"]:
    result_train = state["result"]
    st.success(state["message"])
    a, b, c = st.columns(3)
    with a:
        st.metric("Amostras", result_train["samples"])
    with b:
        st.metric("Loss", f"{result_train['loss']:.4f}", help="Erro do modelo durante o treinamento. Quanto menor, melhor.")
    with c:
        st.metric("Validação", f"{result_train['accuracy'] * 100:.1f}%", help="Acurácia nos dados de validação.")


# ---------------------------------------------------------------------------
# Histórico
# ---------------------------------------------------------------------------

with st.expander("📊 Histórico de treinamentos"):
    if current_stats["runs"]:
        st.dataframe(current_stats["runs"], use_container_width=True, hide_index=True)
    else:
        st.caption("Nenhum treinamento executado ainda.")

st.markdown(
    """
    <div class="text-center text-xs text-slate-600 mt-10">
      PetVision Local · Streamlit · PyTorch · Transformers
    </div>
    """,
    unsafe_allow_html=True,
)