import streamlit as st
from directory_scraper import scrape_expert_directory
from authorship_check import check_authorship
import pandas as pd

st.title("🔎 Authorship Verifier for Expert/Contributor Applications")
st.markdown("Submit a directory URL to check whether submitted content is actually authored by each expert.")

# Input: Directory URL
directory_url = st.text_input("📍 Enter the directory URL:", value="https://www.mlforseo.com/experts/")

if st.button("Scrape & Verify"):
    with st.spinner("Scraping directory and verifying authorship..."):
        try:
            expert_data = scrape_expert_directory(directory_url)
            all_results = []

            for expert_name, urls in expert_data.items():
                st.markdown(f"### 👤 {expert_name}")
                results = check_authorship(expert_name, urls)
                for r in results:
                    st.markdown(f"**🔗 URL:** [{r['url']}]({r['url']})")
                    st.markdown(f"• 📝 Author found: `{r.get('author')}`")
                    st.markdown(f"• ✅ Match: `{r['match']}`")
                    st.markdown(f"• 💬 Reason: {r['reason']}")
                    st.markdown("---")
                    all_results.append({
                        "Expert": expert_name,
                        "Submitted URL": r['url'],
                        "Detected Author": r.get("author"),
                        "Match": r['match'],
                        "Reason": r['reason']
                    })

            # DataFrame + download
            df = pd.DataFrame(all_results)
            st.download_button(
                label="📥 Download results as CSV",
                data=df.to_csv(index=False),
                file_name="authorship_verification_results.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")
