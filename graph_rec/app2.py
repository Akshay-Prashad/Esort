import streamlit as st
from graph_rec import EnhancedPCPartGraph
from Scraper import PCMarketScraper
import pandas as pd
import time
import matplotlib.pyplot as plt 

def main():
    st.set_page_config(page_title="PC Part Advisor", layout="wide")
    
    # Initialize session state
    if 'purchased_parts' not in st.session_state:
        st.session_state.purchased_parts = []
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []
    
    # Initialize components
    pc_graph = EnhancedPCPartGraph()
    scraper = PCMarketScraper(headless=True)
    
    st.title("🖥️ PC Part Compatibility Advisor")
    st.markdown("""
    Get intelligent recommendations for PC parts based on compatibility and market trends.
    """)
    
    # Sidebar for user input
    with st.sidebar:
        st.header("Your Configuration")
        
        # Part input
        new_part = st.text_input("Add a part you own or plan to buy:")
        if st.button("Add Part") and new_part:
            if new_part not in st.session_state.purchased_parts:
                st.session_state.purchased_parts.append(new_part)
                st.success(f"Added {new_part} to your parts list!")
                time.sleep(1)
                st.rerun()
        
        # Display current parts
        if st.session_state.purchased_parts:
            st.subheader("Your Current Parts")
            for part in st.session_state.purchased_parts:
                st.write(f"- {part}")
            
            if st.button("Clear All Parts"):
                st.session_state.purchased_parts = []
                st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Recommendations")
        
        if st.session_state.purchased_parts:
            # Get recommendations
            st.session_state.recommendations = pc_graph.get_recommendations(
                st.session_state.purchased_parts, 
                top_n=5
            )
            
            if st.session_state.recommendations:
                  st.success("Here are our top recommendations for your build:")
                  for i, rec in enumerate(st.session_state.recommendations, 1):
                        with st.expander(f"#{i}: {rec}"):
                              # Get compatibility status
                              compatible = all(pc_graph.check_compatibility(rec, p) 
                                          for p in st.session_state.purchased_parts)
                              
                              st.markdown(f"""
                              - **Category**: {pc_graph.get_part_category(rec)}
                              - **Compatibility**: {"✅ Compatible" if compatible else "❌ Incompatible"}
                              - **Reason**: {pc_graph.get_compatibility_reason(rec, st.session_state.purchased_parts)}
                              """)
                              
                              if compatible and st.button(f"Add {rec} to your build", key=f"add_{rec}"):
                                    st.session_state.purchased_parts.append(rec)
                                    pc_graph.add_user_transaction(st.session_state.purchased_parts)
                                    st.success(f"Added {rec} to your build!")
                                    time.sleep(1)
                                    st.rerun()
                              else:
                                    st.warning("No recommendations found. Try adding more parts.")
                        

    
    with col2:
        st.header("Market Trends")
        
        if st.button("Refresh Market Data"):
            with st.spinner("Fetching latest market trends..."):
                if scraper.update_market_data():
                    st.success("Market data updated!")
                else:
                    st.error("Failed to update market data")
        
        try:
            market_data = pd.read_csv("pc_market_data.csv")
            st.dataframe(market_data.head(10), 
                         height=300,
                         use_container_width=True)
        except FileNotFoundError:
            st.warning("No market data available. Click 'Refresh' to fetch.")
    
    # Graph visualization section
    st.header("Compatibility Graph")
    if st.session_state.purchased_parts:
        pc_graph.visualize_graph(st.session_state.purchased_parts)
        st.pyplot(plt.gcf())
    else:
        st.info("Add parts to visualize compatibility relationships")

if __name__ == "__main__":
    main()