import altair as alt
import numpy as np
import pandas as pd
import rax
import scipy.stats
import streamlit as st
from scipy.stats import kendalltau
from typing import Optional

from src.model import CombinatorialUCBBandit, PBMUCBBandit, CascadeUCBBandit, \
    ImpressionUCBBandit
from src.simulation import PBMSimulator, CascadeSimulator, GeometricSimulator
from src.util import negative_log_likelihood, min_max_scale


def plot_pbm_bias(simulator: PBMSimulator, n_actions: int):
    bias_df = pd.DataFrame(
        {
            "rank": np.arange(1, n_actions + 1),
            "examination": simulator.get_position_bias(n_actions),
        }
    )

    return (
        alt.Chart(bias_df, width=250, height=200)
        .mark_line(point=n_actions < 25)
        .encode(
            x="rank:Q",
            y=alt.Y("examination:Q"),
        )
    )


def plot_geometric_bias(simulator: PBMSimulator, n_actions: int):
    bias_df = pd.DataFrame(
        {
            "rank": np.arange(1, n_actions + 1),
            "examination": simulator.get_position_bias(n_actions),
        }
    )

    cumulative = st.checkbox("Plot cumulative")

    if cumulative:
        bias_df.examination = bias_df.examination.cumsum()

    return (
        alt.Chart(bias_df, width=250, height=200)
        .mark_line(point=n_actions < 25)
        .encode(
            x=alt.X("rank:Q", title="rank k"),
            y=alt.Y("examination:Q", title="prob. to stop at k"),
        )
    )


def plot_beta(relevance, alpha, beta):
    x = np.linspace(0.0, 1.0, 100)
    pdf_df = pd.DataFrame({"x": x, "pdf": scipy.stats.beta.pdf(x, alpha, beta)})
    relevance_df = pd.DataFrame({"x": relevance, "pdf": np.zeros_like(relevance)})

    return alt.Chart(pdf_df, width=250, height=200).mark_line().encode(
        x="x",
        y="pdf",
    ) + alt.Chart(relevance_df).mark_circle().encode(
        x="x",
        y="pdf",
    )


def settings_menu():
    with st.sidebar.expander("**General settings**", expanded=True):
        n_rounds = st.slider(
            "Rounds to simulate:",
            min_value=0,
            max_value=100_000,
            value=20_000,
            step=10_000,
        )

        n_actions = st.slider(
            "Total items:",
            min_value=2,
            max_value=50,
            value=20,
        )

        top_k = st.slider(
            "Top-k items to recommend each round:",
            min_value=1,
            max_value=n_actions,
            value=10,
        )

        random_seed = st.number_input(
            "Random seed:",
            min_value=0,
            max_value=2 ** 32,
            value=42,
        )

    return n_rounds, n_actions, top_k, random_seed


def item_simulation_menu(n_actions):
    with st.sidebar.expander("**Sample item relevance**"):
        st.markdown(
            f"""
        #### Configure Beta distribution
        *Points are the sampled relevance for all {n_actions} items under the current seed.*
        """
        )
        c1, c2 = st.columns(2)
        a = c1.number_input("Alpha:", min_value=1, value=3)
        b = c2.number_input("Beta:", min_value=1, value=10)
        relevance = np.random.beta(a, b, n_actions)
        st.altair_chart(plot_beta(relevance, a, b), use_container_width=True)

        return relevance


def click_simulation_menu(top_k):
    with st.sidebar.expander("**Sample user clicks**"):
        user_model_name = st.selectbox("User model:", ["PBM", "Cascade", "Geometric"])
        examination = None

        if user_model_name == "PBM":
            st.divider()
            st.markdown("#### Configure Position-based Model")

            position_bias = st.slider(
                "Bias strength $\eta$ in $(\\frac{1}{rank})^\eta$:",
                min_value=0.0,
                max_value=2.0,
                step=0.1,
                value=0.0,
            )
            simulator = PBMSimulator(position_bias=position_bias)
            examination = simulator.get_position_bias(top_k)
            st.altair_chart(plot_pbm_bias(simulator, top_k), use_container_width=True)

            if position_bias == 0.0:
                st.warning("$\eta = 0$, no position bias simulated")

        elif user_model_name == "Cascade":
            simulator = CascadeSimulator()
        elif user_model_name == "Geometric":
            st.divider()
            st.markdown("#### Configure Geometric User Model")

            position_bias = st.slider(
                "Bias strength:",
                min_value=0.05,
                max_value=1.0,
                step=0.05,
                value=0.25,
            )
            simulator = GeometricSimulator(position_bias=position_bias)
            st.altair_chart(plot_geometric_bias(simulator, top_k),
                            use_container_width=True)

            st.markdown("""
            A rank is drawn from the geometric distribution
            configured above. The user examines all items until that drawn rank and
            clicks on examined items with their probability of relevance.
            The rank until which all items are examined is available to the 
            Impression-UCB bandit model below.
            """)

        else:
            raise ValueError(f"Unknown user model: {user_model_name}")

        return simulator, examination


def bandit_menu(examination: Optional[np.ndarray] = None):
    with st.sidebar.expander("**Select Bandit algorithm**"):
        name = st.selectbox("Method:",
                            ["CUCB", "PBM-UCB", "Cascade-UCB", "Impression-UCB"])

        if name == "CUCB":
            st.write(
                """
            **[Combinatorial Multi-Armed Bandit: General Framework, Results and Applications](https://proceedings.mlr.press/v28/chen13a.pdf)**\\
            *Wei Chen, Yajun Wang, Yang Yuan (ICML 2013).*
            
            UCB for action i with $N_i$ impressions:
            $$
            \\mu_i + \\epsilon * \\sqrt{\\frac{1.5 \ln N}{N_i}}
            $$
            
            """
            )

            exploration = st.slider("Exploration $\epsilon$", min_value=0.0,
                                 max_value=5.0,
                                 value=1.0,
                                 step=0.25,
                                 )
            bandit = CombinatorialUCBBandit(actions=n_actions, exploration=exploration)
        elif name == "PBM-UCB":
            st.markdown(
                """
            **[Multiple-Play Bandits in the Position-Based Model](https://proceedings.neurips.cc/paper_files/paper/2016/file/51ef186e18dc00c2d31982567235c559-Paper.pdf)**\\
            *Paul Lagrée, Claire Vernade, Olivier Cappé (NIPS 2016).*
            """
            )

            if examination is None:
                st.error(
                    "PBM-UCB currently requires a PBM click simulation with known "
                    "examination probabilities, as position bias estimation is not yet "
                    "implemented."
                )
                st.stop()

            examination = simulator.get_position_bias(top_k)
            bandit = PBMUCBBandit(
                actions=n_actions,
                examination=examination,
                delta=0.1,
            )
        elif name == "Cascade-UCB":
            bandit = CascadeUCBBandit(actions=n_actions)
            st.markdown(
                """
            **[Cascading bandits: Learning to rank in the cascade model](https://proceedings.mlr.press/v37/kveton15.pdf)**\\
            *Branislav Kveton, Csaba Szepesvari, Zheng Wen, Azin Ashkan (ICML 2015).*
            """
            )
        elif name == "Impression-UCB":
            bandit = ImpressionUCBBandit(actions=n_actions)
            st.markdown(
                """
            A cascading bandit with impression tracking. I.e., only arms for items
            scrolled on screen will be updated. Interactions with user models:
            
            * PBM: Has no impression tracking in this simulation, however, we consider
            all items until the last clicked item as observed.
            
            * Cascade: All items until the clicked items are considered observed.
            
            * Geometric: Has impression tracking built-in.  

            **[Cascading bandits: Learning to rank in the cascade model](https://proceedings.mlr.press/v37/kveton15.pdf)**\\
            *Branislav Kveton, Csaba Szepesvari, Zheng Wen, Azin Ashkan (ICML 2015).*
            """
            )
        else:
            raise ValueError(f"Unknown bandit model: {name}")

        return bandit


n_rounds, n_actions, top_k, random_seed = settings_menu()
np.random.seed(random_seed)

relevance = item_simulation_menu(n_actions)
simulator, examination = click_simulation_menu(top_k)
bandit = bandit_menu(examination)

results = []

for i in range(n_rounds):
    actions = bandit.get_actions(top_k)
    clicks, impressions = simulator(relevance[actions])
    bandit.update(actions=actions, reward=clicks, impressions=impressions)

# Prob. of relevance is scaled between 0 and 10:
normalized_relevance = min_max_scale(relevance) * 10
predicted_relevance = bandit.get_relevance()

result_df = pd.DataFrame(
    {
        "action": np.arange(n_actions),
        "relevance": relevance,
        "predicted_relevance": predicted_relevance,
    }
)

st.title("🤖 Simulating Online Bandits for Ranking under Position Bias")

st.success(f"""
    Assessing predicted ranking order (higher is better): nDCG: {rax.ndcg_metric(predicted_relevance, normalized_relevance):.5f},
    Kendall Tau: {kendalltau(predicted_relevance, relevance).correlation:.5f}\\
    Assessing predicted relevance probabilities (lower is better):
    MSE: {rax.pointwise_mse_loss(predicted_relevance, relevance):.5f},
    NLL: {negative_log_likelihood(predicted_relevance, relevance):.5f}
"""
)

sort_by_relevance = st.toggle("Sort actions by relevance", value=True)

if sort_by_relevance:
    result_df = result_df.sort_values("relevance", ascending=False)

df = result_df.melt("action", var_name="relevance_type", value_name="relevance")

chart = (
    alt.Chart(df, width=500, height=200)
    .mark_bar()
    .encode(
        row=alt.Row("relevance_type:N", sort=["relevance", "predicted_relevance"], title=""),
        x=alt.X("action:O", sort=result_df.action.values)
        if sort_by_relevance
        else "action:O",
        y=alt.Y("relevance:Q", title=""),
        color=alt.Color("relevance_type:N", title=""),
    )
)

st.altair_chart(chart, use_container_width=True)
