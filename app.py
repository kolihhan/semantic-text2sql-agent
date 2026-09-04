from pathlib import Path

from semantic_sql.agent import SemanticSQLService
from semantic_sql.providers import DemoProvider, OllamaProvider


def build_service(provider_name: str, model: str) -> SemanticSQLService:
    root = Path(__file__).parent
    provider = DemoProvider() if provider_name == "Demo (offline)" else OllamaProvider(model=model)
    return SemanticSQLService(root / "demo" / "demo.db", provider=provider)


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Semantic Text-to-SQL Agent", layout="wide")
    st.title("Semantic Text-to-SQL Agent")
    st.caption("Generate SQL from the database schema → verify safely → repair if needed.")

    provider_name = st.sidebar.selectbox("Provider", ["Demo (offline)", "Ollama"])
    model = st.sidebar.text_input("Ollama model", "qwen3.5:4b")
    question = st.text_input("Ask the demo database", "Show total sales from Czech Republic in 2024")
    if not st.button("Run", type="primary"):
        return

    result = build_service(provider_name, model).ask(question)
    cols = st.columns(2)
    with cols[0]:
        st.subheader("Generated SQL")
        st.code(result.candidate.sql if result.candidate else "No SQL generated", language="sql")
    with cols[1]:
        st.subheader("Verification")
        if result.verification and result.verification.ok:
            st.success("PASS")
        elif result.verification:
            st.error("; ".join(issue.message for issue in result.verification.issues))
        else:
            st.warning(result.message)

    st.subheader("Result")
    if result.status == "ok":
        st.dataframe([dict(zip(result.columns, row)) for row in result.rows], use_container_width=True)
    else:
        st.warning(f"{result.status}: {result.message}")

    with st.expander("Agent activity"):
        for stage in result.stages:
            st.write(f"**{stage.name}** — {stage.summary}")


if __name__ == "__main__":
    main()
