from pathlib import Path

from semantic_sql.agent import SemanticSQLService
from semantic_sql.providers import DemoProvider, OllamaProvider


EXAMPLES = (
    "Show total sales from Czech Republic in 2024",
    "How many orders were placed in 2024?",
    "Which customer spent the most?",
)


def build_service(provider_name: str, model: str) -> SemanticSQLService:
    root = Path(__file__).parent
    provider = DemoProvider() if provider_name == "Demo (offline)" else OllamaProvider(model=model)
    return SemanticSQLService(root / "demo" / "demo.db", provider=provider)


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Semantic Text-to-SQL", page_icon="⌘", layout="wide")
    st.title("Semantic Text-to-SQL")
    st.caption("Local Text-to-SQL with deterministic verification, bounded repair, and read-only execution.")
    st.markdown("**Generate** → **Verify** → **Repair if needed** → **Execute safely**")

    with st.sidebar:
        st.header("Runtime")
        provider_name = st.selectbox("Provider", ["Demo (offline)", "Ollama"])
        model = st.text_input("Ollama model", "qwen3.5:4b")
        st.caption("The offline provider is deterministic and only demonstrates the product flow. Benchmark results use the frozen local-model run.")

    example = st.selectbox("Try an example", EXAMPLES)
    question = st.text_input("Ask the demo database", example)
    if not st.button("Generate SQL", type="primary", use_container_width=True):
        st.info("Choose a question and run it to see SQL generation, verification, repair activity, and the final result.")
        return

    with st.spinner("Generating and checking SQL..."):
        result = build_service(provider_name, model).ask(question)

    status_col, attempts_col, rows_col = st.columns(3)
    status_col.metric("Status", result.status.upper())
    attempts_col.metric("SQL attempts", (result.candidate.attempt + 1) if result.candidate else 0)
    rows_col.metric("Rows returned", len(result.rows))

    sql_col, verify_col = st.columns([3, 2])
    with sql_col:
        st.subheader("Generated SQL")
        st.code(result.candidate.sql if result.candidate else "No SQL generated", language="sql")
    with verify_col:
        st.subheader("Verification")
        if result.verification and result.verification.ok:
            st.success("PASS — query is read-only and passed deterministic checks")
        elif result.verification:
            st.error("\n".join(issue.message for issue in result.verification.issues))
        else:
            st.warning(result.message or "No verification result")

    st.subheader("Result")
    if result.status == "ok":
        st.dataframe([dict(zip(result.columns, row)) for row in result.rows], use_container_width=True, hide_index=True)
    else:
        st.warning(f"{result.status}: {result.message}")

    st.subheader("Pipeline activity")
    if result.stages:
        for index, stage in enumerate(result.stages, start=1):
            st.markdown(f"**{index}. {stage.name}** — {stage.summary}")
    else:
        st.caption("No stage activity recorded.")


if __name__ == "__main__":
    main()
