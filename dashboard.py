try:
    consents
except NameError:
    consents = []

# Safely ensure both df and consents exist
if 'df' in locals() and 'consents' in locals() and consents:
    # If df rows align one-to-one with uploaded files, insert consent numbers directly
    if len(df) == len(consents):
        df.insert(0, 'Consent Number', consents)
    else:
        # Otherwise, derive consent number from an existing filename column
        if 'filename' in df.columns:
            df['Consent Number'] = df['filename'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
        else:
            st.warning("Could not map consent numbers to DataFrame rows. Ensure each row has a corresponding filename or that row count matches uploads.")

# Now you can safely reference 'Consent Number' in your expander table
with st.expander("Consent Table", expanded=True):
    # 1) Status filter
    status_filter = st.selectbox(
        "Filter by Status",
        ["All"] + df["Consent Status Enhanced"].unique().tolist(),
        key="consent_status_filter"
    )
    # 2) Progress indicator
    my_bar.progress(95, text="Step 4/4: Filtering and displaying consent table...")
    # 3) Apply filter
    filtered_df = df.copy() if status_filter == "All" else df[df["Consent Status Enhanced"] == status_filter]
    # 4) Define exactly the columns to show
    columns_to_display = [
        "Consent Number",  # <-- now the file name
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
