import streamlit as st
import os
os.system("playwright install chromium")
from directory_scraper import scrape_expert_directory
from authorship_check import check_authorship
import pandas as pd

st.set_page_config(page_title="Authorship Verifier", layout="wide")
st.title("🔎 Authorship Verification Tool")

directory_url = st.text_input("📍 Paste the expert directory URL:", value="https://www.mlforseo.com/experts/")

if st.button("Scrape & Verify Authorship"):
    with st.spinner("Scraping expert cards and verifying URLs..."):
        try:
            expert_data = scrape_expert_directory(directory_url)

            if not expert_data:
                st.warning("No experts or links found. Check the directory URL or scraping logic.")
            else:
                all_results = []

                for expert_name, urls in expert_data.items():
                    if not urls:
                        continue  # skip experts with no submitted content

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
                    df = pd.DataFrame(all_results)
                    st.dataframe(df, use_container_width=True)

                    st.download_button(
                        label="📥 Download results as CSV",
                        data=df.to_csv(index=False),
                        file_name="authorship_verification_results.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No author data could be extracted from the submitted URLs.")
        except Exception as e:
            st.error(f"❌ Error: {e}")
