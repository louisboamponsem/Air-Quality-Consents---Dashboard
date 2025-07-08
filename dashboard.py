if 'df' in locals():
    if 'filename' in df.columns:
        df['Consent Number'] = df['filename'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    elif '__file_name__' in df.columns:
        df['Consent Number'] = df['__file_name__'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    else:
        st.warning("No filename column found; cannot extract Consent Number.")

# Now render the consent table only if DataFrame is ready
if 'df' in locals() and not df.empty:
    with st.expander("Consent Table", expanded=True):
        # 1) Status filter
        status_filter = st.selectbox(
            "Filter by Status",
            ["All"] + df["Consent Status Enhanced"].unique().tolist(),
            key="consent_status_filter"
        )
        # 2) Progress indicator
        if 'my_bar' in locals():
            my_bar.progress(95, text="Step 4/4: Filtering and displaying consent table...")
        # 3) Apply filter
        filtered_df = df.copy() if status_filter == "All" else df[df["Consent Status Enhanced"] == status_filter]
        # 4) Define exactly the columns to show
        columns_to_display = [
            "Consent Number",  # from file name
            "Company Name",
            "Address",
            "Issue Date",
            "Expiry Date",
            "Consent Status Enhanced",
            "AUP(OP) Triggers",
        ]
        if "Reason for Consent" in filtered_df.columns:
            columns_to_display.append("Reason for Consent")
        # 5) Slice & rename
        display_df = (
            filtered_df[columns_to_display]
            .rename(columns={"Consent Status Enhanced": "Consent Status"})
        )
        # 6) Render & download
        st.dataframe(display_df)
        csv_output = display_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv_output, "filtered_consents.csv", "text/csv")
else:
    st.warning("No consent data available. Please upload PDF files or check processing logs.")
