import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_policy_percentages(csv_file, output_path):
    # Check the csv file
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"The file {csv_file} does not exist.")

    # Load the data
    df = pd.read_csv(csv_file)
    df['year_month'] = pd.to_datetime(df['year_month'])

    # Create the figure
    fig = make_subplots(specs=[[{"secondary_y": False}]])

    # Add traces for each policy area
    for column in df.columns:
        if column != 'year_month':
            fig.add_trace(
                go.Scatter(x=df['year_month'], 
                           y=df[column], 
                           name=column, 
                           mode='lines', 
                           visible=True)
            )

    # Update layout with reduced margin to avoid empty space
    fig.update_layout(
        margin=dict(l=50, r=50, t=20, b=50),  # Adjust top margin to reduce empty space
        yaxis_title='Monthly Percentage of Articles per Policy Area',
        legend_title='Policy Area',
        hovermode='x unified',
        xaxis=dict(
            tickformat='%b-%Y',
            tickmode='auto',
            nticks=20,
        ),
        yaxis=dict(
            tickformat='.1f',
            ticksuffix='%'
        ),
    )

    # Add buttons for selecting all or none
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        args=["visible", "legendonly"],
                        label="Unselect All",
                        method="restyle"
                    ),
                    dict(
                        args=["visible", True],
                        label="Select All",
                        method="restyle"
                    )
                ],
                pad={"r": 10, "t": 10},
                showactive=False,
                x=1.25,
                xanchor="right",
                y=1.05,
                yanchor="top"
            ),
        ]
    )


    # Add vertical line at 2022-01 with hover text
    fig.add_shape(
        type="line",
        x0="2022-01-01",
        y0=0,
        x1="2022-01-01",
        y1=1,
        line=dict(color="Gray", width=2, dash="dash"),
        xref='x',
        yref='paper'
    )

    fig.add_annotation(
        x="2022-01-01",
        y=0.90,  
        xref='x',
        yref='paper',
        text="<b>Jan 2022</b><br>Model prediction starts",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#FF4500",  # OrangeRed color
        ax=0,
        ay=-40,
        bordercolor="#FF4500",
        borderwidth=2,
        borderpad=4,
        bgcolor="#FF4500",
        opacity=0.8,
        font=dict(color="white", size=12),
        align="center"
    )

    # Configure the legend to always show and allow single item toggle
    fig.update_layout(
        legend=dict(
        itemsizing='constant',
        itemclick='toggle',
        itemdoubleclick='toggleothers'
    ))

    # Save the plot as an HTML file
    fig.write_html(
        output_path,
        full_html=True,
        include_plotlyjs='cdn'
    )

# Use the funciont
if __name__ == "__main__":
    csv_file = 'pci_nko_fg2_df.csv'  
    output_path = 'plot2.html'
    plot_policy_percentages(csv_file, output_path)