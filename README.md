# PetVision Local

Aplicação **100% local**, construída somente com:

- Python
- Streamlit
- HTML/CSS/Tailwind via CDN para a interface
- PyTorch
- Hugging Face Transformers
- SQLite
- Pillow

Não utiliza React, Vite, FastAPI ou qualquer backend web separado.

## Fluxo

```text
Upload
  ↓
Camada 1 — Gato ou Cachorro
  ↓
Probabilidade
  ↓
Camada 2 — Raça
  ↓
Probabilidade
  ↓
Correção manual
  ↓
SQLite + imagem
  ↓
Retreinamento local
```

O feedback humano é utilizado como dado supervisionado adicional.

## Modelos

Camada 1:

`akahana/vit-base-cats-vs-dogs`

Camada 2:

`MaxPowerUnlimited/vit-base-oxford-iiit-pets`

Os modelos são baixados para:

```text
models/
├── species/
└── breed/
```

Os pesos são ignorados pelo `.gitignore`.

## Instalação

### 1. Ambiente virtual

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Baixar os modelos

```bash
python scripts/download_models.py
```

Isso baixa os checkpoints para dentro do próprio projeto.

Depois do download, a aplicação não precisa consultar um servidor de inferência externo.

### 4. Executar

```bash
streamlit run app.py
```

Abra:

```text
http://localhost:8501
```

## Interface

A interface é mobile-first e possui:

- dark mode;
- upload por arquivo;
- preview;
- progresso de upload;
- progresso de inferência;
- resultado da camada 1;
- resultado da camada 2;
- top classificações;
- edição manual;
- salvamento do feedback;
- estatísticas do dataset;
- treinamento;
- progresso do treinamento;
- histórico de treinamentos.

## Tailwind

A interface utiliza classes Tailwind diretamente no HTML renderizado pelo Streamlit.

O CDN é usado somente para carregar o CSS da interface no navegador.

Os modelos e a inferência continuam locais.

Se quiser uma instalação totalmente offline, o arquivo `app.py` possui também CSS fallback próprio para os componentes principais.

## Armazenamento

```text
data/
├── images/
└── petvision.db
```

SQLite registra:

- imagem;
- previsão;
- confiança;
- correção;
- data;
- treinamento;
- loss;
- accuracy.

## Retreinamento

O botão de retreinamento utiliza as imagens que receberam confirmação/correção humana.

Exemplo:

```text
Modelo:
cachorro 97%
Labrador 81%

Usuário:
cachorro
Golden Retriever
```

A imagem entra no dataset com:

```text
species = dog
breed = golden_retriever
```

e poderá ser usada pelo treinamento posterior.

### Por que não treinar automaticamente a previsão do próprio modelo?

Porque isso pode criar um ciclo:

```text
erro
 ↓
modelo considera o erro correto
 ↓
treina com o erro
 ↓
erro aumenta
```

Por isso o projeto usa confirmação humana.

## Requisitos mínimos

Python 3.10+.

GPU NVIDIA/CUDA é opcional.

O código usa:

```python
torch.cuda.is_available()
```

e automaticamente seleciona:

```text
CUDA
ou
CPU
```

## Estrutura

```text
petvision-local/
├── app.py
├── config.py
├── database.py
├── inference.py
├── training.py
├── storage.py
├── requirements.txt
├── .gitignore
├── scripts/
│   └── download_models.py
├── models/
│   ├── species/
│   └── breed/
└── data/
    └── images/
```

## Observação

O arquivo fornecido como base era uma aplicação Streamlit com tema escuro, cards, tooltips e gráficos. Esta versão mantém Streamlit como plataforma e reaproveita a ideia visual de cards/dark mode, mas substitui o domínio de análise de correlação pela classificação de pets.
