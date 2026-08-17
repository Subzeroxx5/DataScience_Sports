"""Sure DataScience — Sports Betting Agent Architecture Research dashboard
(Milestone 13).

    streamlit run dashboard/app.py

Purely a UI layer over the existing agents/experiments/evaluation
modules (see docs/ARCHITECTURE.md, "Dashboard / UI"). This file only
wires the sidebar mode switch to the two view modules — it contains no
sportsbook or quant business logic itself.
"""

from __future__ import annotations

import streamlit as st

from dashboard import demo_view, research_view

st.set_page_config(page_title="Sports Agent Architecture Research", layout="wide")


def main() -> None:
    st.title("Sure DataScience")
    st.caption("Sports Betting Agent Architecture Research — a controlled research prototype, not a betting tool.")

    with st.sidebar:
        st.header("Mode")
        mode = st.radio("Mode", ["Demo", "Research Comparison"], key="app_mode")

    if mode == "Demo":
        render = demo_view.render_demo_mode
    else:
        render = research_view.render_research_mode

    render()


if __name__ == "__main__":
    main()
