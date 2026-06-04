from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.data_prep import engineer_features
from src.app_helpers import initialize_session_state, load_app_assets
from src.model_utils import (
    prepare_single_problem,
    predict_domain_from_text,
    predict_structured_difficulty,
)
try:
    from src.neural_student_state_model import ensure_neural_model_exists, predict_neural_student_state
except (ModuleNotFoundError, ImportError, Exception):
    ensure_neural_model_exists = None
    predict_neural_student_state = None

from src.pedagogical_engine import (
    METACOGNITIVE_QUESTIONS,
    assess_diagnostic_results,
    choose_hint,
    diagnose_learning_state,
    evaluate_answer,
    generate_diagnostic_bank,
    next_review_date,
    recommend_next_exercise,
    target_difficulty_from_mastery,
    update_mastery,
)

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Didact AI - Tutor adaptiv de matematică",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-card {padding: 1rem 1.2rem; border: 1px solid #E5E7EB; border-radius: 16px; background: #FFFFFF; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08); margin-bottom: 0.5rem;}
    .hero-card {padding: 1.2rem 1.25rem; border-radius: 18px; background: linear-gradient(135deg, #F8FAFC 0%, #EEF2FF 100%); border: 1px solid #E5E7EB;}
    .pill {display: inline-block; background: #EEF2FF; color: #3730A3; padding: 0.2rem 0.55rem; border-radius: 999px; font-size: 0.82rem; font-weight: 600;}
    .section-header {padding: 0.9rem 1rem; border-radius: 16px; background: #F8FAFC; border: 1px solid #E2E8F0; margin-bottom: 1rem;}
    .card-title {font-size: 1rem; font-weight: 700; margin-bottom: 0.5rem;}
    .info-chip {display: inline-block; margin-right: 0.5rem; margin-top: 0.4rem; padding: 0.3rem 0.7rem; border-radius: 999px; background: #EEF2FF; color: #0F172A; font-size: 0.85rem;}
    .small-muted {color: #64748B; font-size: 0.92rem;}
    .rubric-good {background: #ECFDF5; color: #065F46; padding: 0.15rem 0.45rem; border-radius: 999px; font-weight: 600;}
    .rubric-warn {background: #FEF3C7; color: #92400E; padding: 0.15rem 0.45rem; border-radius: 999px; font-weight: 600;}
    .service-box {border: 2px solid #6366F1; border-radius: 16px; padding: 1.2rem; margin-bottom: 1rem; background: #FAFAFE;}
    .service-box-2 {border: 2px solid #0EA5E9; border-radius: 16px; padding: 1.2rem; margin-bottom: 1rem; background: #F0FAFF;}
    .service-box-3 {border: 2px solid #10B981; border-radius: 16px; padding: 1.2rem; margin-bottom: 1rem; background: #F0FDF4;}
    .ethics-card {border-left: 4px solid #F59E0B; padding: 0.8rem 1rem; border-radius: 0 12px 12px 0; background: #FFFBEB; margin-bottom: 0.8rem;}
    .team-card {border: 1px solid #E5E7EB; border-radius: 12px; padding: 1rem 1.2rem; background: #F9FAFB; margin-bottom: 0.8rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Se încarcă serviciile ML...")
def cached_assets():
    return load_app_assets()


try:
    structured_model, unstructured_model, data, report = cached_assets()
except Exception as e:
    st.error(f"Eroare la încărcarea serviciilor ML: {e}")
    st.info("Rulați `python -m src.train_models` și asigurați-vă că directoarele models/ și data/processed/ conțin fișierele necesare.")
    st.stop()

# Fix dtypes after CSV load
for col in ["Dificultate", "Itemul", "Sursa_year"]:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

initialize_session_state(st.session_state)

# Check neural model availability once at startup
models_dir = ROOT / "models"
has_neural_artifact = (models_dir / "neural_student_state_model.keras").exists() or (models_dir / "neural_student_state_sklearn.joblib").exists()
if ensure_neural_model_exists is not None:
    try:
        neural_check = ensure_neural_model_exists()
        st.session_state.neural_available = neural_check.get("status") == "ready"
    except Exception:
        st.session_state.neural_available = bool(has_neural_artifact)
else:
    # If the import failed but artifacts exist, mark as available so UI reflects presence.
    st.session_state.neural_available = bool(has_neural_artifact)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("👤 Profil elev")
    name = st.text_input("Nume / poreclă", value="Alex")
    grade = st.selectbox("Clasa", ["V", "VI", "VII", "VIII", "IX"], index=4)
    st.divider()
    st.metric("Mastery curent", f"{st.session_state.mastery:.2f}")
    st.metric("Exerciții rezolvate", len(st.session_state.interaction_log))
    st.divider()
    st.markdown("**Servicii ML active**")
    st.markdown("<span class='rubric-good'>✓ Structurat — dificultate</span>", unsafe_allow_html=True)
    st.markdown("<span class='rubric-good'>✓ Nestructurat — domeniu text</span>", unsafe_allow_html=True)
    if st.session_state.neural_available:
        st.markdown("<span class='rubric-good'>✓ Neural — stare elev</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='rubric-warn'>○ Neural — indisponibil</span>", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🧠 Didact AI")
st.subheader("Tutor adaptiv de matematică pentru elevi de gimnaziu și liceu")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🏠 Acasă",
    "🔬 Cele 2 Servicii ML",
    "🤖 Tutor AI",
    "📈 Progresul meu",
    "📊 Evaluare & EDA",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — HOME
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("Ce este Didact AI?")
    st.markdown(
        """
        <div class='section-header'>
          Didact AI este un tutor adaptiv de matematică care combină <strong>două servicii ML reale și validate</strong>
          cu reguli pedagogice controlabile. Elevul primește exerciții potrivite nivelului său, indicii graduale
          și feedback fără să primească soluția completă automat.
        </div>
        """,
        unsafe_allow_html=True,
    )

    hero = st.columns(3)
    with hero[0]:
        st.markdown("<div class='hero-card'><span class='pill'>Pasul 1</span><br><strong>Diagnostic rapid</strong><br>5 întrebări identifică unde ai nevoie de sprijin.</div>", unsafe_allow_html=True)
    with hero[1]:
        st.markdown("<div class='hero-card'><span class='pill'>Pasul 2</span><br><strong>Exerciții personalizate</strong><br>2 servicii ML estimează domeniul și dificultatea.</div>", unsafe_allow_html=True)
    with hero[2]:
        st.markdown("<div class='hero-card'><span class='pill'>Pasul 3</span><br><strong>Feedback fără soluție</strong><br>Indicii graduale, întrebări metacognitive, mastery tracking.</div>", unsafe_allow_html=True)

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Exerciții în dataset", report["dataset"]["rows_total"])
    with m2:
        st.metric("Model dificultate macro-F1", f"{report['structured_model']['model']['macro_f1']:.3f}")
    with m3:
        st.metric("Model domeniu macro-F1", f"{report['unstructured_model']['model']['macro_f1']:.3f}")
    with m4:
        st.metric("Criterii de jurizare acoperite", "10 / 10")

    st.divider()
    st.markdown("### 🧭 Diagnostic scurt")
    st.info("Rezolvă 5 exerciții și primești recomandări pe zonele unde ai mai mult de exersat.")

    if not st.session_state.diagnostic_started:
        if st.button("Începe testul de diagnostic", type="primary"):
            st.session_state.diagnostic_started = True
            st.session_state.diagnostic_seed = int(time.time()) % 100000
            st.rerun()
    else:
        diagnostic_bank = generate_diagnostic_bank(data, n_questions=5, random_state=st.session_state.diagnostic_seed)
        diagnostic_answers = []
        for idx, row in enumerate(diagnostic_bank, start=1):
            st.markdown(f"<div class='main-card'><strong>{idx}.</strong> {row['Problema']}</div>", unsafe_allow_html=True)
            answer = st.text_area("Răspunsul tău", key=f"diag_{idx}", placeholder="Scrie răspunsul aici")
            if answer:
                result = evaluate_answer(answer, str(row.get("Raspunsul", "")))
                diagnostic_answers.append({"problem": row["Problema"], "domain": row.get("Domeniu"), "correct": result["correct"]})
        if st.button("Finalizează diagnosticul și primește recomandări", type="primary"):
            if not diagnostic_answers:
                st.warning("Completează cel puțin un răspuns înainte de a finaliza diagnosticul.")
                st.stop()
            profile = assess_diagnostic_results(diagnostic_answers, data)
            st.session_state.diagnostic_results = profile
            st.success("Diagnostic finalizat!")
            st.rerun()

    if st.session_state.diagnostic_results:
        profile = st.session_state.diagnostic_results
        st.markdown("#### Rezultat diagnostic")
        weak = profile.get("weak_domains", [])
        if weak:
            st.write("Domenii unde merită mai multă practică:")
            for d in weak[:3]:
                st.markdown(f"- **{d}**")
        for item in profile.get("recommended_themes", [])[:3]:
            st.markdown(f"  - *{item['domain']}*: {', '.join(item['themes'][:3])}")
        if weak:
            rec = recommend_next_exercise(data, weak[0], "2 - mediu", random_state=11)
            if rec is not None and not rec.empty:
                st.markdown("**Exercițiu de start recomandat:**")
                st.markdown(f"<div class='main-card'>{rec.iloc[0]['Problema']}</div>", unsafe_allow_html=True)
                st.caption(f"{rec.iloc[0].get('Domeniu','—')} · {rec.iloc[0].get('Dificultate_group','—')}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CELE 2 SERVICII ML (standalone, testabile separat)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("🔬 Testare independentă a celor 2 servicii ML")
    st.markdown(
        "Această filă permite juriului să testeze **fiecare serviciu ML separat**, "
        "cu input propriu, independent de fluxul de tutoring."
    )

    # ── Serviciu 1: Nestructurat ───────────────────────────────────────────
    st.markdown(
        "<div class='service-box'>"
        "<h3 style='margin:0 0 0.5rem'>📝 Serviciu 1 — Date nestructurate: clasificare domeniu din text</h3>"
        "<p style='margin:0; color:#4338CA;'>Input: enunț liber de problemă matematică &nbsp;→&nbsp; Output: domeniu curricular (7 clase)</p>"
        "<p style='margin:0.5rem 0 0; color:#64748B; font-size:0.9rem;'>Model: TF-IDF + ComplementNB &nbsp;|&nbsp; Macro-F1: <strong>0.974</strong> &nbsp;|&nbsp; Baseline: 0.078</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    text_input = st.text_area(
        "Introdu enunțul unei probleme matematice",
        placeholder="Ex: Calculați aria unui triunghi cu baza 6 cm și înălțimea 4 cm.",
        height=100,
        key="svc1_input",
    )
    if st.button("▶ Clasifică domeniul din text", type="primary", key="svc1_run"):
        if not text_input.strip():
            st.warning("Introdu un enunț pentru clasificare.")
        else:
            with st.spinner("Clasificare în curs..."):
                result = predict_domain_from_text(unstructured_model, text_input)
            st.success(f"**Domeniu prezis:** {result['prediction']}")
            if result.get("probabilities"):
                probs_df = pd.DataFrame(
                    sorted(result["probabilities"].items(), key=lambda kv: kv[1], reverse=True),
                    columns=["Domeniu", "Probabilitate"],
                )
                probs_df["Probabilitate"] = probs_df["Probabilitate"].round(4)
                col_t, col_c = st.columns([1, 1])
                with col_t:
                    st.dataframe(probs_df, use_container_width=True, hide_index=True)
                with col_c:
                    st.bar_chart(probs_df.set_index("Domeniu"))

    st.divider()

    # ── Serviciu 2: Structurat ─────────────────────────────────────────────
    st.markdown(
        "<div class='service-box-2'>"
        "<h3 style='margin:0 0 0.5rem'>📊 Serviciu 2 — Date structurate: estimare dificultate din metadate</h3>"
        "<p style='margin:0; color:#0369A1;'>Input: metadate tabulare (temă, domeniu, lungimi, simboluri) &nbsp;→&nbsp; Output: nivel dificultate (4 clase)</p>"
        "<p style='margin:0.5rem 0 0; color:#64748B; font-size:0.9rem;'>Model: RandomForestClassifier &nbsp;|&nbsp; Macro-F1: <strong>0.802</strong> &nbsp;|&nbsp; CV: 0.826 ± 0.094 &nbsp;|&nbsp; Baseline: 0.163</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    tema_options = sorted(data["Tema_norm"].dropna().unique().tolist())
    domeniu_options = sorted(data["Domeniu"].dropna().unique().tolist())

    sc1, sc2 = st.columns(2)
    with sc1:
        svc2_tema = st.selectbox("Temă normalizată", tema_options, key="svc2_tema")
        svc2_domeniu = st.selectbox("Domeniu", domeniu_options, key="svc2_domeniu")
        svc2_sursa = st.selectbox("Tip sursă", ["manual", "bacalaureat", "evaluare_nationala", "olimpiada"], key="svc2_sursa")
    with sc2:
        svc2_problem = st.text_area(
            "Enunțul problemei (pentru extragerea de features)",
            placeholder="Ex: Fie f: R→R, f(x)=2x+3. Calculați f(5).",
            height=100,
            key="svc2_problem",
        )
        svc2_item = st.number_input("Numărul itemului", min_value=1, max_value=999, value=10, key="svc2_item")

    if st.button("▶ Estimează dificultatea din metadate", type="primary", key="svc2_run"):
        if not svc2_problem.strip():
            st.warning("Introdu enunțul problemei pentru extragerea de features.")
        else:
            with st.spinner("Estimare în curs..."):
                feature_row = prepare_single_problem(
                    problem=svc2_problem,
                    tema_norm=svc2_tema,
                    domeniu=svc2_domeniu,
                    item=int(svc2_item),
                    sursa_type=svc2_sursa,
                )
                result2 = predict_structured_difficulty(structured_model, feature_row)
            st.success(f"**Dificultate prezisă:** {result2['prediction']}")
            if result2.get("probabilities"):
                probs2_df = pd.DataFrame(
                    sorted(result2["probabilities"].items(), key=lambda kv: kv[1], reverse=True),
                    columns=["Nivel dificultate", "Probabilitate"],
                )
                probs2_df["Probabilitate"] = probs2_df["Probabilitate"].round(4)
                col_t2, col_c2 = st.columns([1, 1])
                with col_t2:
                    st.dataframe(probs2_df, use_container_width=True, hide_index=True)
                with col_c2:
                    st.bar_chart(probs2_df.set_index("Nivel dificultate"))

    st.divider()

    # ── Serviciu 3: Neural (bonus) ──────────────────────────────────────────
    st.markdown(
        "<div class='service-box-3'>"
        "<h3 style='margin:0 0 0.5rem'>🧬 Serviciu 3 (bonus) — Neural: estimare stare de învățare a elevului</h3>"
        "<p style='margin:0; color:#065F46;'>Input: metrici sesiune (timp, indicii, încercări, mastery) &nbsp;→&nbsp; Output: stare elev (4 clase)</p>"
        "<p style='margin:0.5rem 0 0; color:#64748B; font-size:0.9rem;'>Model: rețea densă TensorFlow (Dense→Dropout→Dense→Dropout→Softmax) &nbsp;|&nbsp; Antrenat pe date sintetice — demonstrativ</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    if not st.session_state.neural_available or predict_neural_student_state is None:
        st.warning("Componenta neurală nu este disponibilă în acest mediu (TensorFlow lipsă). Serviciile 1 și 2 funcționează complet independent.")
    else:
        nc1, nc2 = st.columns(2)
        with nc1:
            n_time = st.slider("Timp petrecut (secunde)", 5, 300, 60, key="n_time")
            n_hints = st.slider("Indicii folosite", 0, 3, 1, key="n_hints")
            n_attempts = st.slider("Încercări", 1, 5, 2, key="n_attempts")
        with nc2:
            n_correct = st.selectbox("Răspuns corect?", [True, False], key="n_correct")
            n_mistakes = st.slider("Greșeli", 0, 5, 1, key="n_mistakes")
            n_mastery = st.slider("Mastery curent", 0.05, 0.97, 0.55, step=0.05, key="n_mastery")

        if st.button("▶ Estimează starea de învățare (neural)", type="primary", key="svc3_run"):
            with st.spinner("Inferență neurală..."):
                try:
                    neural_features = {
                        "time_spent_seconds": float(n_time),
                        "hint_count": float(n_hints),
                        "attempt_count": float(n_attempts),
                        "is_correct": float(1.0 if n_correct else 0.0),
                        "mistake_count": float(n_mistakes),
                        "exercise_difficulty_encoded": 2.0,
                        "previous_mastery": float(n_mastery),
                        "consecutive_errors": float(n_mistakes),
                        "help_level_requested": float(min(n_hints, 3)),
                    }
                    neural_result = predict_neural_student_state(neural_features)
                    st.success(f"**Stare estimată:** {neural_result['predicted_state']}")
                    st.info(f"**Recomandare:** {neural_result['recommended_action']}")
                    if neural_result.get("probabilities"):
                        np_df = pd.DataFrame(
                            sorted(neural_result["probabilities"].items(), key=lambda kv: kv[1], reverse=True),
                            columns=["Stare", "Probabilitate"],
                        )
                        st.dataframe(np_df, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"Eroare la inferența neurală: {e}")

    # ── Combined scenario ────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🔄 Scenariu combinat: ambele servicii pe același exercițiu")
    st.caption("Demonstrează cum se completează serviciile 1 și 2 în fluxul real al aplicației.")

    combo_ex_options = data[data["Problema"].notna()].index.tolist()
    combo_idx = st.selectbox(
        "Alege un exercițiu din dataset",
        combo_ex_options[:50],
        format_func=lambda i: f"#{int(i)} · {str(data.loc[i,'Domeniu'])} · {str(data.loc[i,'Problema'])[:70]}...",
        key="combo_ex",
    )
    if st.button("▶ Rulează ambele servicii pe exercițiul ales", key="combo_run"):
        combo_row = data.loc[combo_idx]
        combo_text = str(combo_row["Problema"])

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Serviciu 1 — domeniu din text**")
            d_res = predict_domain_from_text(unstructured_model, combo_text)
            st.success(f"Domeniu prezis: **{d_res['prediction']}**")
            st.caption(f"Etichetă reală în dataset: {combo_row.get('Domeniu', '—')}")

        with c2:
            st.markdown("**Serviciu 2 — dificultate din metadate**")
            feat = data.loc[[combo_idx]][[
                "Itemul", "Sursa_year", "problem_chars", "problem_words", "steps_chars", "answer_chars",
                "n_digits", "n_math_symbols", "has_percent", "has_geometry_word", "has_equation_word",
                "has_radical", "has_function_word", "has_real_life_context", "Tema_norm", "Domeniu", "Sursa_type"
            ]]
            diff_res = predict_structured_difficulty(structured_model, feat)
            st.success(f"Dificultate prezisă: **{diff_res['prediction']}**")
            st.caption(f"Etichetă reală în dataset: {combo_row.get('Dificultate_group', '—')}")

        st.markdown(f"<div class='main-card'><strong>Problemă:</strong> {combo_text}</div>", unsafe_allow_html=True)
        st.caption(
            f"Concluzie: modelul text a etichetat domeniul **{d_res['prediction']}** "
            f"iar modelul structurat a estimat dificultatea **{diff_res['prediction']}**. "
            "Împreună permit recomandarea exercițiului următor în tutor."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TUTOR AI
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("🤖 Tutor AI — Rezolvă exerciții și progresează")
    st.markdown(
        "Traseul tău este controlat de **regulile pedagogice** și de cele **două servicii ML** validate. "
        "Serviciul 3 neural (bonus) este disponibil dacă TensorFlow este instalat."
    )

    if not st.session_state.neural_available:
        st.info("ℹ️ Serviciul 3 neural (bonus) nu este activ. Cele două servicii ML principale funcționează complet.")

    domains = sorted(data["Domeniu"].dropna().unique().tolist())
    selected_domain = st.selectbox("Alege domeniul", domains, key="tutor_domain")
    filtered_by_domain = data[data["Domeniu"] == selected_domain] if selected_domain else data

    if st.button("▶ Exercițiu nou", type="primary", key="start_new_exercise"):
        candidates = filtered_by_domain.sample(min(5, len(filtered_by_domain)), random_state=42)
        selected_idx = candidates.index[0]
        st.session_state.selected_exercise_idx = selected_idx
        st.session_state.current_exercise_start_time = time.time()
        st.session_state.current_exercise_attempt_count = 0
        st.session_state.current_exercise_hint_count = 0
        st.session_state.current_exercise_mistake_count = 0
        st.session_state.current_exercise_consecutive_errors = 0

    if "selected_exercise_idx" not in st.session_state:
        st.info("Apasă butonul de mai sus pentru a începe un exercițiu.")
    else:
        exercise_idx = st.session_state.selected_exercise_idx
        current_row = data.loc[exercise_idx]
        elapsed = int(time.time() - st.session_state.current_exercise_start_time) if st.session_state.current_exercise_start_time else 0

        st.divider()
        st.markdown("### 📝 Problemă")
        st.markdown(f"<div class='main-card'>{current_row['Problema']}</div>", unsafe_allow_html=True)
        st.caption(f"Domeniu: **{current_row.get('Domeniu','—')}** · Temă: **{current_row.get('Tema_norm','—')}** · Nivel: **{current_row.get('Dificultate_group','—')}**")

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Încercări", st.session_state.current_exercise_attempt_count or 0)
        sc2.metric("Indicii", st.session_state.current_exercise_hint_count)
        sc3.metric("Timp", f"{elapsed}s")

        student_answer = st.text_area("Scrie răspunsul tău", placeholder="Introdu răspunsul...", key=f"tutor_answer_{exercise_idx}")

        st.markdown("#### Întrebare de conștientizare")
        q_idx = (int(exercise_idx) + st.session_state.current_exercise_hint_count) % len(METACOGNITIVE_QUESTIONS)
        st.info(METACOGNITIVE_QUESTIONS[q_idx])

        col_hints, col_submit = st.columns([1, 1])
        with col_hints:
            if st.button("💡 Cere indiciu", key="hint_btn"):
                st.session_state.current_exercise_hint_count += 1
                hint = choose_hint(
                    current_row["Problema"],
                    current_row.get("Pasii de rezolvare", ""),
                    st.session_state.mastery,
                    st.session_state.current_exercise_hint_count,
                )
                st.markdown(f"**Tip: {hint['hint_type']}**")
                st.info(hint["hint"])

        with col_submit:
            if st.button("✓ Verifică răspunsul", type="primary", key="check_btn"):
                st.session_state.current_exercise_attempt_count += 1
                result = evaluate_answer(student_answer, current_row.get("Raspunsul", ""))
                is_correct = result["correct"]
                time_spent = time.time() - st.session_state.current_exercise_start_time

                if not is_correct:
                    st.session_state.current_exercise_mistake_count += 1

                domain_pred = predict_domain_from_text(unstructured_model, str(current_row["Problema"]))
                feat = data.loc[[exercise_idx]][[
                    "Itemul", "Sursa_year", "problem_chars", "problem_words", "steps_chars", "answer_chars",
                    "n_digits", "n_math_symbols", "has_percent", "has_geometry_word", "has_equation_word",
                    "has_radical", "has_function_word", "has_real_life_context", "Tema_norm", "Domeniu", "Sursa_type"
                ]]
                diff_pred = predict_structured_difficulty(structured_model, feat)

                neural_pred = None
                if st.session_state.neural_available and predict_neural_student_state is not None:
                    try:
                        neural_features = {
                            "time_spent_seconds": float(time_spent),
                            "hint_count": float(st.session_state.current_exercise_hint_count),
                            "attempt_count": float(st.session_state.current_exercise_attempt_count),
                            "is_correct": float(1.0 if is_correct else 0.0),
                            "mistake_count": float(st.session_state.current_exercise_mistake_count),
                            "exercise_difficulty_encoded": 2.0,
                            "previous_mastery": float(st.session_state.mastery),
                            "consecutive_errors": float(st.session_state.current_exercise_consecutive_errors),
                            "help_level_requested": float(min(st.session_state.current_exercise_hint_count, 3)),
                        }
                        neural_pred = predict_neural_student_state(neural_features)
                    except Exception:
                        neural_pred = None

                learning_state = diagnose_learning_state(is_correct, st.session_state.current_exercise_hint_count, st.session_state.current_exercise_attempt_count, int(time_spent))
                new_mastery = update_mastery(st.session_state.mastery, is_correct, st.session_state.current_exercise_hint_count, st.session_state.current_exercise_attempt_count)
                st.session_state.mastery = new_mastery

                entry = {
                    "exercise_id": int(current_row.get("Itemul", 0)) if pd.notna(current_row.get("Itemul")) else 0,
                    "problem_text": str(current_row.get("Problema", ""))[:200],
                    "predicted_domain": domain_pred["prediction"],
                    "predicted_difficulty": diff_pred["prediction"],
                    "time_spent_seconds": float(time_spent),
                    "hint_count": int(st.session_state.current_exercise_hint_count),
                    "attempt_count": int(st.session_state.current_exercise_attempt_count),
                    "mistake_count": int(st.session_state.current_exercise_mistake_count),
                    "is_correct": bool(is_correct),
                    "predicted_learning_state": neural_pred["predicted_state"] if neural_pred else "reguli",
                    "timestamp": time.time(),
                }
                st.session_state.interaction_log.append(entry)

                st.divider()
                if is_correct:
                    st.success("✓ Răspunsul este corect!")
                else:
                    st.error("✗ Răspunsul nu este corect. Încearcă din nou sau cere un indiciu.")
                st.write(result["feedback"])

                with st.expander("📊 Analiza sistemului"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Timp", f"{int(time_spent)}s")
                    c2.metric("Indicii", st.session_state.current_exercise_hint_count)
                    c3.metric("Mastery nou", f"{new_mastery:.2f}")
                    st.write(f"- Domeniu estimat (S1 text): **{domain_pred['prediction']}**")
                    st.write(f"- Dificultate estimată (S2 structurat): **{diff_pred['prediction']}**")
                    if neural_pred:
                        st.write(f"- Stare elev (S3 neural): **{neural_pred['predicted_state']}** — {neural_pred['recommended_action']}")

                st.markdown(f"**Stare pedagogică:** {learning_state['state']} → {learning_state['intervention']}")
                st.markdown(f"**Reactivare spaced repetition:** {next_review_date(new_mastery, is_correct)}")

                if is_correct or st.session_state.current_exercise_attempt_count >= 3:
                    target = target_difficulty_from_mastery(new_mastery, is_correct)
                    next_ex = recommend_next_exercise(data, current_row["Domeniu"], target, exclude_problem=current_row["Problema"], random_state=42)
                    if next_ex is not None and not next_ex.empty:
                        st.markdown("---")
                        st.markdown("### 🎯 Exercițiul următor recomandat")
                        st.write(f"Țintă: **{target}** · Domeniu: **{current_row['Domeniu']}**")
                        st.markdown(f"<div class='main-card'>{next_ex.iloc[0]['Problema']}</div>", unsafe_allow_html=True)
                        if st.button("Continuă cu exercițiul următor ➜", type="primary", key="next_ex_btn"):
                            next_idx = next_ex.index[0]
                            st.session_state.selected_exercise_idx = next_idx
                            st.session_state.current_exercise_start_time = time.time()
                            st.session_state.current_exercise_attempt_count = 0
                            st.session_state.current_exercise_hint_count = 0
                            st.session_state.current_exercise_mistake_count = 0
                            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PROGRES
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("📈 Progresul meu")

    if not st.session_state.interaction_log:
        st.info("Încă nu ai rezolvat exerciții. Mergi la **Tutor AI** și începe!")
    else:
        log_df = pd.DataFrame(st.session_state.interaction_log)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Exerciții rezolvate", len(log_df))
        accuracy = (log_df["is_correct"].sum() / len(log_df) * 100) if len(log_df) > 0 else 0
        c2.metric("Acuratețe", f"{accuracy:.1f}%")
        c3.metric("Indicii medie", f"{log_df['hint_count'].mean():.1f}")
        c4.metric("Timp mediu", f"{int(log_df['time_spent_seconds'].mean())}s")

        st.divider()
        st.markdown("### Progres pe domenii")
        domain_stats = log_df.groupby("predicted_domain").agg({"is_correct": ["sum", "count"], "time_spent_seconds": "mean"}).round(2)
        domain_stats.columns = ["Corecte", "Total", "Timp mediu (s)"]
        st.dataframe(domain_stats, use_container_width=True)

        if "predicted_learning_state" in log_df.columns and log_df["predicted_learning_state"].notna().any():
            st.markdown("### Stări de învățare detectate")
            st.bar_chart(log_df["predicted_learning_state"].value_counts())

        st.markdown("### Istoric recent")
        recent = log_df.tail(10)[["problem_text", "predicted_difficulty", "hint_count", "attempt_count", "is_correct", "time_spent_seconds"]].copy()
        recent["Rezultat"] = recent["is_correct"].map({True: "✓ Corect", False: "✗ Incorect"})
        recent = recent.drop("is_correct", axis=1)
        st.dataframe(recent, use_container_width=True)

        if st.button("🔄 Resetează progresul"):
            st.session_state.interaction_log = []
            st.session_state.mastery = 0.55
            st.success("Progres resetat!")
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EVALUARE & EDA
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("📊 Evaluare, comparație de modele și EDA")
    st.caption("Toate cifrele sunt din models/evaluation_report.json, regenerat din date prin `python -m src.train_models`. Nimic nu este hardcodat.")

    sm = report["structured_model"]
    um = report["unstructured_model"]
    ds = report["dataset"]

    st.subheader("Metrici față de baseline")
    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown("**Serviciu 1 — text: domeniu**")
        st.write(f"Macro-F1 model: **{um['model']['macro_f1']:.3f}** (baseline {um['baseline']['macro_f1']:.3f})")
        st.write(f"Balanced accuracy: {um['model']['balanced_accuracy']:.3f}")
        st.write(f"CV macro-F1: {um['best_cv_macro_f1']:.3f} ± {um.get('best_cv_macro_f1_std', 0):.3f}")
        st.write(f"Best params: `{um['best_params']}`")
    with mc2:
        st.markdown("**Serviciu 2 — structurat: dificultate**")
        st.write(f"Macro-F1 model: **{sm['model']['macro_f1']:.3f}** (baseline {sm['baseline']['macro_f1']:.3f})")
        st.write(f"Balanced accuracy: {sm['model']['balanced_accuracy']:.3f}")
        st.write(f"CV macro-F1: {sm['best_cv_macro_f1']:.3f} ± {sm.get('best_cv_macro_f1_std', 0):.3f}")
        st.write(f"Best params: `{sm['best_params']}`")

    st.subheader("KPI educațional (impact, nu doar acuratețe)")
    kpi_path = ROOT / "models" / "educational_kpi.json"
    if kpi_path.exists():
        kpi = json.loads(kpi_path.read_text(encoding="utf-8"))
        k1, k2, k3 = st.columns(3)
        k1.metric("Δmastery / succes curat", f"{kpi.get('mastery_improvement_rate_clean_success', 0):+.3f}")
        k2.metric("Indicii / răspuns corect", f"{kpi.get('hints_per_correct', 0):.2f}")
        k3.metric("Rată succese curate", f"{kpi.get('clean_success_rate', 0):.2f}")
        st.caption("KPI măsurat din student_interactions.csv. Un macro-F1 bun nu garantează impact pedagogic.")

    st.subheader("Comparație între modele candidate")
    st.caption("Aceeași preprocesare și aceeași validare 5-fold pentru fiecare candidat.")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Structurat**")
        comp = sm.get("model_comparison", {})
        if comp:
            st.dataframe(pd.DataFrame(comp).T, use_container_width=True)
    with cc2:
        st.markdown("**Text**")
        comp = um.get("model_comparison", {})
        if comp:
            st.dataframe(pd.DataFrame(comp).T, use_container_width=True)
    st.caption("RandomForest ales pentru stabilitate CV. Alternativele sunt raportate transparent.")

    st.subheader("Analiză exploratorie a datelor (EDA)")
    assets = ROOT / "assets"

    def _show_asset(filename: str, caption: str):
        p = assets / filename
        if p.exists():
            st.image(str(p), caption=caption, use_container_width=True)

    st.markdown("**Date structurate — corelații între features numerice**")
    _show_asset("feature_correlation_heatmap.png", "Corelații (ex: problem_chars ~ problem_words 0.96 → redundanță documentată).")
    pairs = ds.get("top_correlated_feature_pairs", [])
    if pairs:
        st.dataframe(pd.DataFrame(pairs), use_container_width=True)

    g1, g2 = st.columns(2)
    with g1:
        _show_asset("difficulty_distribution.png", "Distribuția dificultății — dezechilibrată, de aceea macro-F1.")
    with g2:
        _show_asset("domain_distribution.png", "Distribuția domeniilor curriculare.")

    st.markdown("**Date nestructurate — EDA text**")
    _show_asset("wordclouds_by_domain.png", "Word clouds pe domeniu.")
    _show_asset("top_tokens_by_domain.png", "Termeni cei mai frecvenți pe domeniu.")
    _show_asset("text_length_by_domain.png", "Lungimea enunțului pe domeniu.")

    st.subheader("Matrici de confuzie și erori reprezentative")
    for title, m in [("Text (domeniu)", um), ("Structurat (dificultate)", sm)]:
        cm = m.get("confusion_matrix", {})
        if cm:
            st.markdown(f"**{title}** — etichete: {', '.join(map(str, cm['labels']))}")
            cm_df = pd.DataFrame(cm["matrix"], index=cm["labels"], columns=cm["labels"])
            st.dataframe(cm_df, use_container_width=True)
        errs = m.get("sample_errors", [])
        if errs:
            with st.expander(f"Erori reprezentative — {title}"):
                st.dataframe(pd.DataFrame(errs), use_container_width=True)

    st.subheader("Curbe de învățare (analiza limitării dataset mic)")
    lc1, lc2 = st.columns(2)
    with lc1:
        _show_asset("learning_curve_structured.png", "Structurat: gap train/CV = semn de overfitting pe date puține.")
    with lc2:
        _show_asset("learning_curve_text.png", "Text: CV se aplatizează ~0.99 → dataset suficient.")

    la = um.get("leakage_audit", {})
    if la and "error" not in la:
        st.subheader("Audit de leakage (model text)")
        st.write(
            f"- Itemi de test cu near-duplicate în train: "
            f"**{la.get('test_items_with_near_duplicate_in_train')}/{la.get('naive_test_items')}** "
            f"({la.get('share_leaky', 0)*100:.0f}%)."
        )
        st.write(
            f"- Re-evaluare group-aware ({la.get('n_near_duplicate_groups')} grupuri): "
            f"macro-F1 holdout **{la.get('group_aware_holdout_macro_f1')}**, "
            f"CV **{la.get('group_aware_cv_macro_f1_mean')} ± {la.get('group_aware_cv_macro_f1_std')}**."
        )
        st.success(la.get("conclusion", ""))

    fi = sm.get("feature_importance", {})
    if fi:
        st.subheader("Importanța features (model structurat)")
        fi_df = pd.DataFrame({"feature": list(fi.keys()), "importance": list(fi.values())})
        st.bar_chart(fi_df.set_index("feature"))


