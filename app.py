"""
🚖 TaxiLM Agent — Interface Streamlit
Chauffeur de taxi Hassaniya avec outils vivants (meteo, geocodage, change, priere)
"""

import streamlit as st
import torch
import uuid
import requests
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
import importlib.util, sys

# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(page_title="🚖 TaxiLM Agent", page_icon="🚖", layout="wide")

REPO_ID = "AlihIng/TaxiLM"

# ============================================================
# CHARGEMENT DU MODELE (cache)
# ============================================================
@st.cache_resource
def load_taxilm():
    config_path = hf_hub_download(repo_id=REPO_ID, filename="config.py")
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config_module
    spec.loader.exec_module(config_module)
    
    model_path = hf_hub_download(repo_id=REPO_ID, filename="model.py")
    spec = importlib.util.spec_from_file_location("model", model_path)
    model_module = importlib.util.module_from_spec(spec)
    sys.modules["model"] = model_module
    spec.loader.exec_module(model_module)
    
    pt_path = hf_hub_download(repo_id=REPO_ID, filename="model.pt")
    checkpoint = torch.load(pt_path, map_location="cpu")
    config = config_module.TaxiConfig(**checkpoint["config"])
    model = model_module.TaxiLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    tokenizer = Tokenizer.from_file(hf_hub_download(repo_id=REPO_ID, filename="tokenizer.json"))
    
    return model, tokenizer

model, tokenizer = load_taxilm()

# ============================================================
# OUTILS
# ============================================================
from langchain_core.tools import tool

@tool
def get_meteo_nouakchott() -> dict:
    """Retourne la meteo actuelle a Nouakchott : temperature, vent, pluie."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": 18.0735, "longitude": -15.9582, "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation"}
    r = requests.get(url, params=params, timeout=10)
    data = r.json()["current"]
    return {"temperature_c": data["temperature_2m"], "vent_kmh": data["wind_speed_10m"], "pluie_mm": data["precipitation"]}

@tool
def trouver_quartier(nom_quartier: str) -> dict:
    """Cherche les coordonnees d'un quartier de Nouakchott."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": f"{nom_quartier}, Nouakchott, Mauritanie", "format": "json", "limit": 1}
    headers = {"User-Agent": "TaxiLM/1.0"}
    r = requests.get(url, params=params, headers=headers, timeout=10)
    results = r.json()
    if results:
        return {"quartier": nom_quartier, "latitude": results[0]["lat"], "longitude": results[0]["lon"], "trouve": True}
    return {"quartier": nom_quartier, "trouve": False}

@tool
def convertir_en_ouguiya(montant: float, devise: str = "EUR") -> dict:
    """Convertit un montant en EUR ou USD vers l'Ouguiya (MRU)."""
    url = f"https://api.exchangerate-api.com/v4/latest/{devise.upper()}"
    r = requests.get(url, timeout=10)
    taux = r.json()["rates"]["MRU"]
    return {"devise": devise.upper(), "montant": montant, "taux": taux, "montant_mru": round(montant * taux, 2)}

@tool
def get_horaires_priere(city: str = "Nouakchott", country: str = "Mauritania") -> dict:
    """Retourne les horaires de priere du jour."""
    url = "https://api.aladhan.com/v1/timingsByCity"
    params = {"city": city, "country": country, "method": 3}
    r = requests.get(url, params=params, timeout=10)
    timings = r.json()["data"]["timings"]
    return {"Fajr": timings["Fajr"], "Dhuhr": timings["Dhuhr"], "Asr": timings["Asr"], "Maghrib": timings["Maghrib"], "Isha": timings["Isha"]}

tools = [get_meteo_nouakchott, trouver_quartier, convertir_en_ouguiya, get_horaires_priere]

# ============================================================
# LLM WRAPPER
# ============================================================
class TaxiLLM:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def invoke(self, prompt):
        if isinstance(prompt, list):
            prompt = prompt[-1].content if hasattr(prompt[-1], 'content') else str(prompt)
        input_ids = self.tokenizer.encode(prompt).ids
        input_t = torch.tensor([input_ids], dtype=torch.long)
        output_t, _ = self.model.generate(input_t, max_new_tokens=128)
        return self.tokenizer.decode(output_t[0].tolist()[len(input_ids):])
    
    def bind_tools(self, tools, **kwargs):
        return self

llm = TaxiLLM(model, tokenizer)

# ============================================================
# AGENT
# ============================================================
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

character_profile = """
Nom : Mohamed Vall (محمد فال).
Langue : Hassaniya avec quelques mots francais (clim, frein, carrefour, essence, match).
Role : Chauffeur de taxi a Nouakchott depuis 20 ans.
Ton : Bavard, raleur mais bon coeur, philosophe de la route.
Expressions : ماشي امنين, ماني ماشي لهيه, يا راجل, لا إله إلا الله, اشواعد, إن شاء الله.
Univers : Les rues de Nouakchott, les embouteillages, le prix de l'essence, les courses,
les matchs des Mourabitounes, le the, la chaleur, les dos d'ane.
Sagesse : "ذا كامل يتخطا" (tout passe, meme les embouteillages).
Limites strictes : Ne JAMAIS inventer un itineraire, un prix officiel, un nom de rue
administratif ou une information factuelle inconnue.
Si hors de son univers, il repond : "والله مانعرف, سول حد يعرف".
"""

@st.cache_resource
def create_taxi_agent():
    memory_saver = MemorySaver()
    system_prompt = f"""{character_profile}

SOURCES DISPONIBLES :
- API meteo (Open-Meteo)
- API geocodage (Nominatim)
- API taux de change (Exchange Rate API)
- API horaires de priere (AlAdhan)

STRUCTURE TA REPONSE FINALE AVEC :
Action : [outil utilise]
Source : [API utilisee]
Confiance : [forte / moyenne / faible]

Puis reponds dans ton style de chauffeur Hassaniya.
"""
    return create_agent(model=llm, tools=tools, system_prompt=system_prompt, checkpointer=memory_saver)

agent = create_taxi_agent()

# ============================================================
# INTERFACE STREAMLIT
# ============================================================
st.title("🚖 TaxiLM Agent — Chauffeur Nouakchott")
st.markdown("*Agent Hassaniya avec outils vivants : meteo, geocodage, change, priere*")

with st.sidebar:
    st.header("📋 Exemples")
    exemples = [
        "شماسي الجواليوم؟",
        "شنو حالة الطقس اليوم؟",
        "بكم التوصيل لكرفور؟",
        "شنو اوقات الصلاة اليوم؟",
        "شنو نصيحتك للحياة؟",
    ]
    for ex in exemples:
        if st.button(ex):
            st.session_state.prompt = ex
    
    st.divider()
    st.header("📊 Metriques de l'agent")
    st.markdown(f"**Outils :** {len(tools)}")
    st.markdown("""
    - Meteo (Open-Meteo)
    - Geocodage (Nominatim)
    - Taux de change
    - Horaires priere
    """)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Pose ta question en Hassaniya ou en francais...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.spinner("🚖 Le chauffeur reflechit..."):
        run_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": run_id}}
        
        response = agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config=config
        )
        
        reply = response["messages"][-1].content
    
    with st.chat_message("assistant"):
        st.markdown(f"**🚖 Chauffeur :** {reply}")
        
        with st.expander("📋 Details de l'agent"):
            st.markdown(f"**Action :** appel a l'outil approprie")
            st.markdown(f"**Sources :** API + modele TaxiLM")
            st.markdown(f"**Confiance :** moyenne (modele 9M)")
            st.markdown(f"**Trace :** `{run_id}`")
    
    st.session_state.messages.append({"role": "assistant", "content": reply})

st.divider()
st.caption("TaxiLM Agent — TP3 LangChain — Modele 8.7M entraine sur corpus Hassaniya")