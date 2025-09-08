import os
import streamlit as st
import pandas as pd
from directory_scraper import scrape_expert_directory
from authorship_check import check_authorship

# For Streamlit Cloud: install Chromium for Playwright
os.system("playwright install chromium")

# Streamlit app setup
st.set_page_config(page_title="Authorship Verification Tool", layout="wide")
st.title("🕵️ Authorship Verification Tool")

tab1, tab2 = st.tabs([
    "📂 Scrape + Check Directory",
    "📝 Check Author-Submitted Content"
])

# --------------------------------
# 🔹 Shared function: highlight mismatches
# --------------------------------
def highlight_mismatch(row):
    if row["Match"] is False:
        return ['background-color: #ffe6e6'] * len(row)  # Light red
    return [''] * len(row)

# --------------------------------
# 📂 TAB 1: Scrape Directory
# --------------------------------
with tab1:
    st.subheader("📂 Scrape Expert Directory and Verify Authorship")

    directory_url = st.text_input(
        "📍 Paste the expert directory URL:",
        value="https://www.mlforseo.com/experts/"
    )

    if st.button("Scrape & Verify Authorship", key="scrape_button"):
        # Only use spinner for scraping step
        with st.spinner("Scraping expert cards from the directory..."):
            try:
                expert_data = scrape_expert_directory(directory_url)
            except Exception as e:
                st.error(f"❌ Error scraping directory: {e}")
                expert_data = None

        if not expert_data:
            st.warning("⚠️ No experts or links found. Check the directory URL or scraping logic.")
        else:
            all_results = []
            total_experts = len(expert_data)
            progress = st.progress(0)
            progress_text = st.empty()

            for i, (expert_name, urls) in enumerate(expert_data.items()):
                if not urls:
                    continue

                progress_text.text(f"🔍 Verifying: {expert_name} ({i + 1}/{total_experts})")
                results = check_authorship(expert_name, urls)

                for r in results:
                    all_results.append({
                        "Expert": expert_name,
                        "Submitted URL": r['url'],
                        "Detected Author": r.get("author"),
                        "Match": r['match'],
                        "Reason": r['reason']
                    })

                progress.progress((i + 1) / total_experts)

            progress_text.text("✅ All authorship checks completed.")

            if all_results:
                st.success("✅ Scraping and authorship verification complete.")
                df = pd.DataFrame(all_results)

                # Highlight mismatches
                styled_df = df.style.apply(highlight_mismatch, axis=1)

                st.subheader("📊 Authorship Evaluation Results")
                st.dataframe(styled_df, use_container_width=True)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download results as CSV",
                    data=csv,
                    file_name="authorship_verification_results.csv",
                    mime="text/csv"
                )
            else:
                st.warning("⚠️ No author data could be extracted from the submitted URLs.")

# --------------------------------
# 📝 TAB 2: Manual Submission
# --------------------------------
with tab2:
    st.subheader("📝 Check Author-Submitted Content")

    with st.form("manual_check_form"):
        author_name = st.text_input("👤 Author Name")
        url_input = st.text_area(
            "🔗 Content URLs (comma-separated)",
            placeholder="https://example.com/post1, https://example.com/post2"
        )
        submitted = st.form_submit_button("Check Authorship")

    if submitted:
        if not author_name or not url_input:
            st.warning("⚠️ Please enter both the author name and at least one URL.")
        else:
            urls = [u.strip() for u in url_input.split(",") if u.strip()]
            with st.spinner("Verifying submitted URLs..."):
                try:
                    results = check_authorship(author_name, urls)
                    if results:
                        st.success(f"✅ Checked {len(results)} content source(s) for '{author_name}'.")

                        df = pd.DataFrame([{
                            "Expert": author_name,
                            "Submitted URL": r['url'],
                            "Detected Author": r.get("author"),
                            "Match": r['match'],
                            "Reason": r['reason']
                        } for r in results])

                        styled_df = df.style.apply(highlight_mismatch, axis=1)
                        st.dataframe(styled_df, use_container_width=True)

                        csv = df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Download results as CSV",
                            data=csv,
                            file_name="manual_authorship_check.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("⚠️ No author data could be extracted.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
