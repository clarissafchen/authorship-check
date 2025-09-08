import os
import streamlit as st
import pandas as pd
from directory_scraper import scrape_expert_directory
from authorship_check import check_authorship

# Install Playwright Chromium on first run (for Streamlit Cloud)
os.system("playwright install chromium")

# Streamlit config
st.set_page_config(page_title="Authorship Verification Tool", layout="wide")
st.title("🕵️ Authorship Verification Tool")
st.markdown("Submit an expert directory URL. We'll scrape expert names and URLs, and verify content authorship.")

# URL input
directory_url = st.text_input(
    "📍 Paste the expert directory URL:",
    value="https://www.mlforseo.com/experts/"
)

if st.button("Scrape & Verify Authorship"):
    with st.spinner("Scraping expert cards and verifying authorship..."):
        try:
            expert_data = scrape_expert_directory(directory_url)

            if not expert_data:
                st.warning("⚠️ No experts or links found. Check the directory URL or scraping logic.")
            else:
                all_results = []

                for expert_name, urls in expert_data.items():
                    if not urls:
                        continue

                    results = check_authorship(expert_name, urls)
                    for r in results:
                        all_results.append({
                            "Expert": expert_name,
                            "Submitted URL": r['url'],
                            "Detected Author": r.get("author"),
                            "Match": r['match'],
                            "Reason": r['reason']
                        })

                if all_results:
                    st.success("✅ Scraping and authorship verification complete.")

                    df = pd.DataFrame(all_results)
                    st.subheader("📊 Authorship Evaluation Results")
                    st.dataframe(df, use_container_width=True)

                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download results as CSV",
                        data=csv,
                        file_name="authorship_verification_results.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("⚠️ No author data could be extracted from the submitted URLs.")

        except Exception as e:
            st.error(f"❌ Error during scraping or verification: {e}")                    
