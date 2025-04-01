import streamlit as st
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import altair as alt
import time  #Import the time module

# Function to load Excel filex
file_path = r"C:\Users\101652\Downloads\combined_data2.xlsx"
def load_excel(file_path):
    return pd.read_excel(file_path)

# Function for business MIS section (renamed from data viewer)
def business_mis_section(df):
    # Create a layout with two columns: main content on left, filters on right
    main_col, filter_col = st.columns([0.7, 0.3])
    
    # Initialize session state variables if they don't exist
    if 'filtered_df' not in st.session_state:
        st.session_state.filtered_df = df.copy()
    if 'filters' not in st.session_state:
        st.session_state.filters = {}
    if 'filter_applied' not in st.session_state:
        st.session_state.filter_applied = False
    
    # Load custom CSS
    with open("styles.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    # Filter column (right side)
    with filter_col:
        st.write("### Filter Options")
        
        # Only filter specific columns
        filter_columns = ['Branch_name', 'SM_Name', 'Program_type', 'Applied_loan', 'CM_name', 'Current_progress']
        
        # Create multiselect filters for specific columns
        temp_filters = {}
        for column in filter_columns:
            if column in df.columns:
                unique_values = df[column].unique()
                temp_filters[column] = st.multiselect(
                    f"{column}",
                    options=unique_values,
                    default=st.session_state.filters.get(column, [])
                )
        
        # Add Filter and Reset buttons
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="filter-container">', unsafe_allow_html=True)
            if st.button("Filter"):
                st.session_state.filters = temp_filters
                st.session_state.filter_applied = True
                
                # Apply filters
                filtered_df = df.copy()
                for column, selected in st.session_state.filters.items():
                    if selected:
                        filtered_df = filtered_df[filtered_df[column].isin(selected)]
                
                st.session_state.filtered_df = filtered_df
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="filter-container">', unsafe_allow_html=True)
            if st.button("Reset"):
                st.session_state.filters = {}
                st.session_state.filter_applied = False
                st.session_state.filtered_df = df.copy()
                st.rerun()  # Updated from st.experimental_rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Main column (left side)
    with main_col:
        # Only show one dataframe (the filtered one if filters applied, otherwise original)
        current_df = st.session_state.filtered_df
        
        if st.session_state.filter_applied:
            st.write("Filtered Dataframe:")
        else:
            st.write("Original Dataframe:")
        
        # Display the dataframe with scrolling
        st.dataframe(current_df, height=500)  # Increased height for better visibility
        
        # Display row count
        st.write(f"Showing {len(current_df)} rows out of {len(df)} total rows")
    
    # Create tabs for business MIS section
    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "City", "Credit", "Sales"])
    
    # Activity Over Time section using Seaborn
    with tab1:
            st.markdown("<h3 style='text-align: center;'>Activity Over Time</h3>", unsafe_allow_html=True)

            # Date columns for analysis
            date_columns = ['Login_sales_date', 'First_Credit_Decision', 'Disbursal_date']
            date_columns_present = [col for col in date_columns if col in df.columns]

            # Initialize session state for chart filters
            if 'chart_filters' not in st.session_state:
                st.session_state.chart_filters = {}
            if 'chart_filtered_df' not in st.session_state:
                st.session_state.chart_filtered_df = df.copy()

            # Create filters for the chart with 3 columns for better layout
            chart_filter_col1, chart_filter_col2, chart_filter_col3 = st.columns(3)

            with chart_filter_col1:
                st.markdown('<div class="filter-container">', unsafe_allow_html=True)
                if 'Program_type' in df.columns:
                    program_types = ['All'] + list(df['Program_type'].dropna().unique())
                    selected_program = st.selectbox(
                        "Filter by Program Type",
                        options=program_types,
                        key="program_type_filter"
                    )
                else:
                    selected_program = 'All'
                    st.info("'Program_type' column not found")
                st.markdown('</div>', unsafe_allow_html=True)

            with chart_filter_col2:
                st.markdown('<div class="filter-container">', unsafe_allow_html=True)
                if 'Branch_name' in df.columns:
                    # Map branches to zones
                    zone_mapping = {
                        "Coimbatore": "South 2", "Delhi": "North 1", "Hyderabad": "South 1",
                        "Vadodara": "West", "Bangalore": "South 2", "Ahmedabad": "West",
                        "Jaipur": "North 2", "Mumbai": "West", "Rajkot": "West",
                        "Pune": "West", "Chennai": "South 2", "Kolkata": "North 1",
                        "Mysore": "South 2", "Vijayawada": "South 1", "Jodhpur": "North 2",
                        "Panipat": "North 1", "Lucknow": "North 2", "Madurai": "South 2",
                        "Raipur": "West", "Ludhiana": "North 1", "Salem": "South 2",
                        "Hubli": "South 2", "Surat": "West", "Chandigarh": "North 1",
                        "Davangere": "South 2", "Kanpur": "North 2", "Indore": "West",
                        "Vapi": "West", "Nashik": "West", "Visakhapatnam": "South 1",
                        "Aurangabad": "West", "Varanasi": "North 2", "Kolhapur": "West",
                        "Warangal": "South 1", "Agra": "North 2"
                    }
                    
                    # Add a Zone column if it doesn't exist
                    if 'Zone' not in df.columns:
                        df['Zone'] = df['Branch_name'].map(lambda x: zone_mapping.get(x, 'Other'))
                    
                    zones = ['All'] + list(df['Zone'].dropna().unique())
                    selected_zone = st.selectbox(
                        "Filter by Zone",
                        options=zones,
                        key="zone_filter"
                    )
                else:
                    selected_zone = 'All'
                    st.info("'Branch_name' column not found for Zone filtering")
                st.markdown('</div>', unsafe_allow_html=True)

            with chart_filter_col3:
                st.markdown('<div class="filter-container">', unsafe_allow_html=True)
                # Add date range filter
                min_date = pd.to_datetime('2020-01-01')  # Default minimum date
                max_date = pd.to_datetime('today')       # Default maximum date
                
                # Find actual min/max dates across all date columns
                for col in date_columns_present:
                    temp_dates = pd.to_datetime(df[col], errors='coerce').dropna()
                    if not temp_dates.empty:
                        if temp_dates.min() < min_date:
                            min_date = temp_dates.min()
                        if temp_dates.max() > max_date:
                            max_date = temp_dates.max()
                
                # Convert to date for the date picker
                min_date = min_date.date()
                max_date = max_date.date()
                
                # Date range selector
                date_range = st.date_input(
                    "Select Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key="date_range_filter"
                )
                
                # Handle single date selection case
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    start_date, end_date = date_range
                else:
                    start_date = end_date = date_range
                st.markdown('</div>', unsafe_allow_html=True)

            # Add Filter and Reset buttons for chart
            chart_btn_col1, chart_btn_col2 = st.columns(2)

            @st.cache_data
            def filter_data(df, selected_program, selected_zone, start_date, end_date, date_columns_present):
                filtered_chart_df = df.copy()
                if selected_program != 'All':
                    filtered_chart_df = filtered_chart_df[filtered_chart_df['Program_type'] == selected_program]
                if selected_zone != 'All':
                    filtered_chart_df = filtered_chart_df[filtered_chart_df['Zone'] == selected_zone]
                
                # Apply date filters to all date columns
                date_filtered_df = filtered_chart_df.copy()
                for col in date_columns_present:
                    date_filtered_df[col] = pd.to_datetime(date_filtered_df[col], errors='coerce')
                
                # Create a combined mask for all date columns
                date_mask = pd.Series(False, index=date_filtered_df.index)
                for col in date_columns_present:
                    col_mask = (
                        (date_filtered_df[col].dt.date >= start_date) & 
                        (date_filtered_df[col].dt.date <= end_date)
                    )
                    date_mask = date_mask | col_mask
                
                filtered_chart_df = filtered_chart_df[date_mask]
                return filtered_chart_df

            with chart_btn_col1:
                st.markdown('<div class="filter-container">', unsafe_allow_html=True)
                if st.button("Apply Chart Filters"):
                    with st.spinner('Updating chart...'):
                        st.session_state.chart_filters = {
                            'Program_type': selected_program,
                            'Zone': selected_zone,
                            'start_date': start_date,
                            'end_date': end_date
                        }
                        
                        st.session_state.chart_filtered_df = filter_data(df, selected_program, selected_zone, start_date, end_date, date_columns_present)
                st.markdown('</div>', unsafe_allow_html=True)

            with chart_btn_col2:
                st.markdown('<div class="filter-container">', unsafe_allow_html=True)
                if st.button("Reset Chart Filters"):
                    with st.spinner('Resetting chart...'):
                        st.session_state.chart_filters = {}
                        st.session_state.chart_filtered_df = df.copy()
                        st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            # Use the filtered dataframe for the chart
            chart_df = st.session_state.chart_filtered_df

            if date_columns_present:
                activity_data = {}
                total_counts = {}  # For mini metrics
                
                # Process each date column
                for i, column in enumerate(date_columns_present):
                    # Convert to datetime
                    chart_df[column] = pd.to_datetime(chart_df[column], errors='coerce')
                    
                    # Filter out rows with missing dates
                    valid_data = chart_df[~chart_df[column].isna()]
                    
                    if not valid_data.empty:
                        # Count by date
                        date_counts = valid_data.groupby(valid_data[column].dt.date).size()
                        
                        # Use readable names
                        column_name = {
                            'Login_sales_date': 'Logins',
                            'First_Credit_Decision': 'Credit Decisions',
                            'Disbursal_date': 'Disbursals'
                        }.get(column, column)
                        
                        activity_data[column_name] = date_counts
                        total_counts[column_name] = len(valid_data)
                
                if activity_data:
                  
                    
                    # Prepare data for Plotly
                    chart_data = []
                    for metric, counts in activity_data.items():
                        for date, count in counts.items():
                            chart_data.append({
                                'Date': date,
                                'Count': count,
                                'Metric': metric
                            })
                    
                    # Convert to DataFrame
                    chart_df_plotly = pd.DataFrame(chart_data)
                    
                    # Create Plotly line plot
                    import plotly.express as px
                    fig = px.line(
                        chart_df_plotly, 
                        x='Date', 
                        y='Count',
                        color='Metric',
                        color_discrete_sequence=px.colors.qualitative.Set2,  # A more visually appealing color sequence
                        title='Login-Credit-Disbursal'
                    )
                    # Center align the title
                    fig.update_layout(
                        title={
                            'text': 'Login-Credit-Disbursal',
                            'x': 0.5,
                            'xanchor': 'center'
                        }
                    )
                    
                    # Improve the plot appearance
                    fig.update_layout(
                        xaxis_title='Date',
                        yaxis_title='Count',
                        hovermode="x unified"
                    )
                    
                    # Display the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add a sunburst chart for Zone and Program distribution
                    st.markdown("<h3 style='text-align: center;'>Zone and Program Distribution</h3>", unsafe_allow_html=True)
                    
                    # Add some spacing
                    st.markdown("<div style='margin: 1em;'></div>", unsafe_allow_html=True)
                    
                    # Filter out rejected cases
                    non_rejected_df = chart_df[chart_df['Charts_dot'] != 'Rejected'].copy()
                    
                    if not non_rejected_df.empty and 'Zone' in non_rejected_df.columns and 'Program_type' in non_rejected_df.columns:
                        # Group by Zone and Program_type
                        zone_program_counts = non_rejected_df.groupby(['Zone', 'Program_type']).size().reset_index(name='Count')
                        
                        # Calculate percentages within each zone
                        zone_totals = zone_program_counts.groupby('Zone')['Count'].sum().reset_index()
                        zone_totals.rename(columns={'Count': 'Zone_Total'}, inplace=True)
                        zone_program_counts = zone_program_counts.merge(zone_totals, on='Zone')
                        zone_program_counts['Percentage'] = (zone_program_counts['Count'] / zone_program_counts['Zone_Total'] * 100).round(1)
                        
                        # Add percentage to customdata for hover info
                        zone_program_counts['Label'] = zone_program_counts['Program_type'] + ' (' + zone_program_counts['Percentage'].astype(str) + '%)'
                        
                        # Create sunburst chart
                        fig_sunburst = px.sunburst(
                            zone_program_counts,
                            path=['Zone', 'Label'],
                            values='Count',
                            color_discrete_sequence=px.colors.qualitative.Set2,  # Match the color theme from line chart
                            title='*Excludes Rejected Cases*'
                        )
                        
                        fig_sunburst.update_layout(
                            margin=dict(t=30, b=0, l=0, r=0),
                            height=500
                        )
                        
                        # Display the sunburst chart
                        st.plotly_chart(fig_sunburst, use_container_width=True)
                        
                        # Add some spacing
                        st.markdown("<div style='margin: 1em;'></div>", unsafe_allow_html=True)
                        
                        # Add explanation
                        st.markdown("<div style='text-align: center; font-size: 0.8em; color: gray;'>This chart displays the distribution of non-rejected cases across different geographical zones. The inner ring shows the proportion of each zone, while the outer ring shows the program type distribution within each zone (with percentages relative to that zone's total).</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='text-align: center;'>Insufficient data for Zone-Program distribution chart</div>", unsafe_allow_html=True)
                                    
                else:
                    st.markdown("<div style='text-align: center;'>No activity data available for charting</div>", unsafe_allow_html=True)
                    
                # Display current filters
                if st.session_state.chart_filters:
                    filter_text = []
                    if st.session_state.chart_filters.get('Program_type') != 'All':
                        filter_text.append(f"Program Type: {st.session_state.chart_filters.get('Program_type')}")
                    if st.session_state.chart_filters.get('Zone') != 'All':
                        filter_text.append(f"Zone: {st.session_state.chart_filters.get('Zone')}")
                    if st.session_state.chart_filters.get('start_date'):
                        start = st.session_state.chart_filters.get('start_date')
                        end = st.session_state.chart_filters.get('end_date')
                        filter_text.append(f"Date Range: {start} to {end}")
                    
                    if filter_text:
                        st.markdown(f"<div style='text-align: center; font-size: 0.8em;'>Applied filters: {', '.join(filter_text)}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='text-align: center;'>No valid date columns found for charting</div>", unsafe_allow_html=True)

    with tab2:
        st.write("### City-wise Data")

        # Load a different dataframe
        df2 = df.groupby(["Branch_name", "Charts_dot"]).size().reset_index(name='Count')

        # Create an interactive scatter plot using plotly express
        import plotly.express as px
        fig = px.scatter(
            df2,
            x='Charts_dot',
            y='Branch_name',
            color='Charts_dot',
            size='Count',
            color_continuous_scale='temps',
            hover_name='Branch_name',
            hover_data=['Charts_dot', 'Count']
        )

        # Update layout for better scaling and font adjustments
        fig.update_layout(
            title={
                'text': 'Interactive Scatter Plot with Multiple Variables',
                'x': 0.5,  # Center-align the title
                'xanchor': 'center',
                'font': {'size': 16}  # Adjust title font size
            },
            xaxis_title={
                'text': 'Current Stage',
                'font': {'size': 12}  # Adjust x-axis title font size
            },
            yaxis_title={
                'text': 'Branch',
                'font': {'size': 12}  # Adjust y-axis title font size
            },
            height=600,  # Increase the height for better visualization
            yaxis=dict(
                autorange=False,
                range=[-0.5, 9.5],
                tickfont={'size': 10}  # Reduce tick font size for better fit
            ),
            xaxis=dict(
                tickfont={'size': 10}  # Reduce tick font size for better fit
            )
        )

        # Update marker to scale horizontally and adjust sizeref for better sizing
        fig.update_traces(
            marker=dict(
                symbol='square',
                sizemode='area',
                sizeref=2. * max(df2['Count']) / (50. ** 2)  # Adjust marker sizing logic
            )
        )

        # Display the Plotly chart with Streamlit
        st.plotly_chart(fig, use_container_width=True)  # Autofit to container width
    
    with tab3:
        st.write("### Credit Manager Progress")

        # Filter for unique CM_Name values
        cm_df = df[df['CM_Name'].notna() & (df['CM_Name'] != '')]
        if not cm_df.empty:
            unique_cms = cm_df['CM_Name'].unique()
            selected_cm = st.selectbox("Select Credit Manager", unique_cms)

            # Filter DataFrame based on selected CM_Name
            filtered_df = cm_df[cm_df['CM_Name'] == selected_cm]

            # Calculate total count and unique counts
            total_count = filtered_df.shape[0]
            
            if total_count > 0:
                # Get counts by Current_progress
                progress_counts = filtered_df['Current_progress'].value_counts().reset_index()
                progress_counts.columns = ['Current_progress', 'Count']
                
                # Add percentage column
                progress_counts['Percentage'] = (progress_counts['Count'] / total_count * 100).round(1)
                
                # Sort by count descending
                progress_counts = progress_counts.sort_values('Count', ascending=False)
                
                # Define waffle chart dimensions
                cols = 20  # Width of the rectangle
                rows = 10  # Height of the rectangle
                total_blocks = rows * cols
                
                # Create data for the optimized waffle chart
                waffle_data = []
                remaining_blocks = total_blocks
                
                # Assign blocks to each status proportionally
                for _, row in progress_counts.iterrows():
                    status = row['Current_progress']
                    percentage = row['Percentage']
                    blocks_count = min(int(total_blocks * percentage / 100), remaining_blocks)
                    remaining_blocks -= blocks_count
                    
                    # If there's a rounding issue and we're at the last status, allocate remaining blocks
                    if percentage > 0 and blocks_count == 0:
                        blocks_count = 1
                        remaining_blocks -= 1
                    
                    # Add the blocks to the waffle data
                    waffle_data.extend([status] * blocks_count)
                
                # If we have any remaining blocks due to rounding, fill with the first status
                if remaining_blocks > 0:
                    waffle_data.extend([progress_counts['Current_progress'].iloc[0]] * remaining_blocks)
                    
                # Create a DataFrame for the waffle chart with grid coordinates
                waffle_df = pd.DataFrame({
                    'Current_progress': waffle_data,
                    'row': [i // cols for i in range(total_blocks)],
                    'col': [i % cols for i in range(total_blocks)]
                })
                
                # Create the rectangular waffle chart using Plotly
                fig = px.scatter(
                    waffle_df,
                    x='col',
                    y='row',
                    color='Current_progress',
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    labels={'Current_progress': 'Status'},
                    title=f'Portfolio Status for {selected_cm} ({total_count} applications)'
                )
                
                # Update layout for a clean rectangular grid
                fig.update_traces(marker=dict(
                    size=25,
                    symbol='square',
                    line=dict(width=1, color='white')
                ))
                
                fig.update_layout(
                    height=400,
                    width=800,
                    xaxis=dict(
                        showgrid=False,
                        zeroline=False,
                        showticklabels=False,
                        range=[-1, cols]
                    ),
                    yaxis=dict(
                        showgrid=False,
                        zeroline=False,
                        showticklabels=False,
                        range=[rows, -1],
                        scaleanchor="x",
                        scaleratio=1  # Ensures squares are square
                    ),
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                
                # Display the waffle chart
                st.plotly_chart(fig, use_container_width=True)
                
                # Display legend as a summary table
                st.write("### Application Status Summary")
                summary_df = progress_counts[['Current_progress', 'Count', 'Percentage']]
                summary_df = summary_df.rename(columns={
                    'Current_progress': 'Status',
                    'Count': 'Applications',
                    'Percentage': '% of Portfolio'
                })
                summary_df['% of Portfolio'] = summary_df['% of Portfolio'].apply(lambda x: f"{x}%")
                
                # Display the summary table
                st.dataframe(summary_df, use_container_width=True)
            else:
                st.info(f"No applications found for {selected_cm}")
        else:
            st.warning("No Credit Manager data available")

    with tab4:
        st.write("### Sales Manager Progress")

        # Filter for unique SM_Name values
        sm_df = df[df['SM_Name'].notna() & (df['SM_Name'] != '')]
        if not sm_df.empty:
            unique_sms = sm_df['SM_Name'].unique()
            selected_sm = st.selectbox("Select Sales Manager", unique_sms)

            # Filter DataFrame based on selected SM_Name
            filtered_df = sm_df[sm_df['SM_Name'] == selected_sm]

            # Calculate total count and unique counts
            total_count = filtered_df.shape[0]
            
            if total_count > 0:
                # Get counts by Current_progress
                progress_counts = filtered_df['Current_progress'].value_counts().reset_index()
                progress_counts.columns = ['Current_progress', 'Count']
                
                # Add percentage column
                progress_counts['Percentage'] = (progress_counts['Count'] / total_count * 100).round(1)
                
                # Sort by count descending
                progress_counts = progress_counts.sort_values('Count', ascending=False)
                
                # Define waffle chart dimensions
                cols = 20  # Width of the rectangle
                rows = 10  # Height of the rectangle
                total_blocks = rows * cols
                
                # Create data for the optimized waffle chart
                waffle_data = []
                remaining_blocks = total_blocks
                
                # Assign blocks to each status proportionally
                for _, row in progress_counts.iterrows():
                    status = row['Current_progress']
                    percentage = row['Percentage']
                    blocks_count = min(int(total_blocks * percentage / 100), remaining_blocks)
                    remaining_blocks -= blocks_count
                    
                    # If there's a rounding issue and we're at the last status, allocate remaining blocks
                    if percentage > 0 and blocks_count == 0:
                        blocks_count = 1
                        remaining_blocks -= 1
                    
                    # Add the blocks to the waffle data
                    waffle_data.extend([status] * blocks_count)
                
                # If we have any remaining blocks due to rounding, fill with the first status
                if remaining_blocks > 0:
                    waffle_data.extend([progress_counts['Current_progress'].iloc[0]] * remaining_blocks)
                    
                # Create a DataFrame for the waffle chart with grid coordinates
                waffle_df = pd.DataFrame({
                    'Current_progress': waffle_data,
                    'row': [i // cols for i in range(total_blocks)],
                    'col': [i % cols for i in range(total_blocks)]
                })
                
                # Create the rectangular waffle chart using Plotly
                fig = px.scatter(
                    waffle_df,
                    x='col',
                    y='row',
                    color='Current_progress',
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    labels={'Current_progress': 'Status'},
                    title=f'Portfolio Status for {selected_sm} ({total_count} applications)'
                )
                
                # Update layout for a clean rectangular grid
                fig.update_traces(marker=dict(
                    size=25,
                    symbol='square',
                    line=dict(width=1, color='white')
                ))
                
                fig.update_layout(
                    height=400,
                    width=800,
                    xaxis=dict(
                        showgrid=False,
                        zeroline=False,
                        showticklabels=False,
                        range=[-1, cols]
                    ),
                    yaxis=dict(
                        showgrid=False,
                        zeroline=False,
                        showticklabels=False,
                        range=[rows, -1],
                        scaleanchor="x",
                        scaleratio=1  # Ensures squares are square
                    ),
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                
                # Display the waffle chart
                st.plotly_chart(fig, use_container_width=True)
                
                # Display legend as a summary table
                st.write("### Application Status Summary")
                summary_df = progress_counts[['Current_progress', 'Count', 'Percentage']]
                summary_df = summary_df.rename(columns={
                    'Current_progress': 'Status',
                    'Count': 'Applications',
                    'Percentage': '% of Portfolio'
                })
                summary_df['% of Portfolio'] = summary_df['% of Portfolio'].apply(lambda x: f"{x}%")
                
                # Display the summary table
                st.dataframe(summary_df, use_container_width=True)
                
                # Add additional sales-specific metrics if available
                if 'Applied_loan' in filtered_df.columns:
                    st.write("### Sales Performance Metrics")
                    
                    # Clean and convert the Applied_loan column
                    filtered_df['Applied_loan_numeric'] = filtered_df['Applied_loan'].apply(clean_loan_amount)
                    
                    # Calculate sales metrics using the cleaned column
                    total_loan_amount = filtered_df['Applied_loan_numeric'].sum()
                    average_loan = filtered_df['Applied_loan_numeric'].mean()
                    max_loan = filtered_df['Applied_loan_numeric'].max()
                    
                    # Display metrics in columns
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    
                    with metrics_col1:
                        st.metric("Total Loan Amount", f"₹{total_loan_amount:,.2f}")
                    
                    with metrics_col2:
                        st.metric("Average Loan Size", f"₹{average_loan:,.2f}")
                    
                    with metrics_col3:
                        st.metric("Largest Application", f"₹{max_loan:,.2f}")
                    
            else:
                st.info(f"No applications found for {selected_sm}")
        else:
            st.warning("No Sales Manager data available")

# Add this function to clean and convert loan amounts properly
def clean_loan_amount(value):
    if pd.isna(value):
        return 0
    
    try:
        # If it's already a number, return it
        if isinstance(value, (int, float)):
            return value
        
        # If it's a string, clean it and convert
        if isinstance(value, str):
            # Remove all spaces
            value = value.strip().replace(" ", "")
            
            # Check if the string appears to be multiple values concatenated together
            if len(value) > 15:  # Arbitrary threshold for detecting concatenated values
                # Extract just the first number before any concatenation
                import re
                match = re.search(r'^\d{1,2}(?:,\d{2})*(?:,\d{3})*', value)
                if match:
                    value = match.group(0)
                else:
                    return 0
            
            # Remove commas
            value = value.replace(",", "")
            
            # Convert to float
            return float(value)
        
        return 0
    except:
        return 0

# Streamlit app
def main():
    # Initialize session state for loading state
    if 'loading' not in st.session_state:
        st.session_state.loading = True

    # Display spinner while loading
    if st.session_state.loading:
        col1, col2, col3 = st.columns([1, 2, 1])  # Adjust column widths as needed
        with col2:
            st.markdown(
                """
                <div style="display: flex; justify-content: center; align-items: center;">
                    <div class="spinner" style="width: 35.2px; height: 35.2px;">
                        <div></div>
                        <div></div>
                        <div></div>
                        <div></div>
                        <div></div>
                        <div></div>
                    </div>
                    <h1 style="margin-left: 10px;">Business MIS</h1>
                </div>
                """,
                unsafe_allow_html=True
            )
            # Simulate loading time (replace with your actual data loading)
            time.sleep(2)  # Adjust the sleep time as needed
            st.session_state.loading = False  # Set loading to False after loading

    # Once loading is complete, display the rest of the app
    if not st.session_state.loading:
        
   
        if os.path.exists(file_path):
            # Load the Excel file directly from the file path
            df = load_excel(file_path)
            # Add metrics in a 4x2 grid layout
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric(label="Login Count", value="TBD")
            with metric_col2:
                st.metric(label="Approval Rate", value="TBD")
            with metric_col3:
                st.metric(label="Sanction Count", value="TBD")
            with metric_col4:
                st.metric(label="WROI", value="TBD")
                
            metric_col5, metric_col6, metric_col7, metric_col8 = st.columns(4)
            with metric_col5:
                st.metric(label="Average Ticket Size", value="TBD")
            with metric_col6:
                st.metric(label="Processing TAT", value="TBD")
            with metric_col7:
                st.metric(label="Disbursed Amount", value="TBD")
            with metric_col8:
                st.metric(label="Portfolio Yield", value="TBD")
            
            # Add some space
            st.write("")
            
            # Create sidebar navigation
            page = st.sidebar.radio("Navigation", ["Business MIS","Disbursal MIS","Leaderboard"])
            
            # Display the selected page
            if page == "Business MIS":
                business_mis_section(df)
            elif page == "Analysis":
                analysis_section()
            elif page == "Stay Tuned":
                stay_tuned_section()
            
        else:
            st.error(f"File not found: {file_path}")

# Simplified sections for Analysis and Stay Tuned
def analysis_section():
    st.header("Analysis")
    st.write("This section is under development.")
    st.info("Future features will include trend analysis, performance metrics, and forecasting tools.")

def stay_tuned_section():
    st.header("Stay Tuned!")
    st.success("Coming soon: Advanced reporting, export options, and customizable dashboards.")

if __name__ == "__main__":
    main()
