from pathlib import Path
import re
import time

import streamlit as st

from config import (
    MIN_SPECIES_SAMPLES,
    MIN_BREED_SAMPLES,
)
from database import (
    init_db,
    add_image,
    add_prediction,
    add_feedback,
    stats,
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

        # Espaçador FIXO
        st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)

        # CARD CAMADA 1 ENRIQUECIDO
        top_species = result.get("species", [])
        second_species = top_species[1] if len(top_species) > 1 else None
        
        species_html = f"""
            <div class="pet-result">
              <div class="flex justify-between items-start">
                  <div>
                      <div class="pet-muted uppercase tracking-wider mb-1">
                        Camada 1 · Espécie
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
                  <span class="pet-muted text-xs">2ª opção: {pretty(second_species['label'])}</span>
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
                          <div class="pet-muted uppercase tracking-wider mb-1">
                            Camada 2 · Raça
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
            
            # Adiciona o Top 3 de raças
            top_breeds = result.get("breed", [])[:3]
            if len(top_breeds) > 1:
                breed_html += '<div class="mt-3 pt-3 border-t border-slate-700/50 space-y-1">'
                breed_html += '<div class="pet-muted text-xs mb-2 uppercase tracking-wider">Top 3 Raças</div>'
                
                for idx, item in enumerate(top_breeds, start=1):
                    # AJUSTE: Removida a indentação (espaços à esquerda) para o Streamlit não ler como código
                    breed_html += f'<div class="flex justify-between items-center text-sm"><span class="text-slate-300">{idx}. {pretty(item["label"])}</span><span class="text-slate-400 font-medium">{confidence(item["probability"])}</span></div>'
                
                breed_html += '</div>'
                
            breed_html += "</div>"
            st.markdown(breed_html, unsafe_allow_html=True)
        else:
            st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)
            st.info("A camada de raça não é aplicada para esta espécie.")

        # METADADOS DA INFERÊNCIA (Tempo, Dispositivo)
        st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)
        
        inf_time = st.session_state.get("inference_time", 0.0)
        
        st.markdown(f"""
            <div class="flex justify-between items-center bg-slate-800/30 p-3 rounded-xl border border-slate-700/50">
                <div class="flex items-center gap-2">
                    <span class="text-slate-400 text-xs uppercase tracking-wider">Dispositivo</span>
                    <span class="pet-badge bg-indigo-500/20 text-indigo-300">{result['device'].upper()}</span>
                </div>
                <div class="flex items-center gap-2">
                    <span class="text-slate-400 text-xs uppercase tracking-wider">Tempo</span>
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
# Dataset
# ---------------------------------------------------------------------------

st.divider()
current_stats = stats()

st.markdown(
    """
    <div class="pet-card">
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
    st.metric("Exemplos rotulados", current_stats["total"])
with c2:
    st.metric("Mínimo espécie", MIN_SPECIES_SAMPLES)
with c3:
    st.metric("Mínimo raça", MIN_BREED_SAMPLES)


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
        st.metric("Loss", f"{result_train['loss']:.4f}")
    with c:
        st.metric("Validação", f"{result_train['accuracy'] * 100:.1f}%")


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