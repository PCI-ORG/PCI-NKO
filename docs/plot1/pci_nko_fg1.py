import plotly.graph_objs as go
import pandas as pd
import os

def create_nko_plotly_figure():
    # Load data
    results_file = 'pci_nko_fg1_df.csv'
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"The file {results_file} does not exist.")

    # Read in the results
    results = pd.read_csv(results_file)

    # Prepare main data
    df = pd.DataFrame({
        "y": results['difference'] if 'difference' in results.columns else results.iloc[:, -1],
        "x": pd.to_datetime(results['Pred End'] if 'Pred End' in results.columns else results.iloc[:, 0]),
    })
    df = df.sort_values(by='x')

    # Create hover text
    df['text'] = df.x.dt.strftime('%Y-%m-%d') + '<br>PCI: ' + df.y.round(4).astype(str)

    # Hard-coded events
    events = pd.DataFrame({
        "date": pd.to_datetime([
            "2022-03-06", "2022-03-24", "2022-04-25", "2022-05-12", "2022-08-10",
            "2022-11-18", "2023-04-13", "2023-09-12", "2023-11-21", "2023-12-31",
            "2024-02-08", "2024-03-15"
        ]),
        "text": [
            "Began to destroy inter-Korean projects in the North",
            "Resumed testing liquid-fuel ICBM",
            "Announced nuclear vision beyond self-defense",
            "Declared severe national emergency in response to Covid outbreak",
            "Declared victory over COVID outbreak",
            "First successful launch of (liquid-fuel) ICBM Hwasong-17, <br>and first public appearance of Kim Jong Un's daughter",
            "First successful launch of (solid-fuel) ICBM Hwasong-18",
            "Kim Jong Un's visit to Russia and the beginning of closer Russo-North Korean ties",
            "First successful launch of spy satellite Malligyong-1",
            "Abandoned reunification goal toward South Korea",
            "Abolished the law on economic cooperation with South Korea",
            "Further signal of Kim Jong Un's daughter as a potential successor"
        ]
    })
    events['formatted_text'] = events['date'].dt.strftime('%Y-%m-%d') + '<br>' + events['text']

    # Calculate the appropriate x-axis range
    x_min = min(df.x.min(), events['date'].min())
    x_max = max(df.x.max(), events['date'].max())

    # Add some padding to the x-axis range (e.g., 1 month before and after)
    x_min -= pd.Timedelta(days=30)
    x_max += pd.Timedelta(days=30)

    # Create traces
    trace_main = go.Scatter(
        x=df.x.tolist(),  # Convert to list of datetime objects
        y=df.y.tolist(),
        text=df.text.tolist(),
        hoverinfo="text",
        name="PCI"
    )

    trace_events = go.Scatter(
        x=events.date.tolist(),  # Convert to list of datetime objects
        y=[df.y.max()] * len(events),
        text=events.formatted_text.tolist(),
        hoverinfo="text",
        mode='markers',
        marker=dict(opacity=0),
        name="Events",
        hovertemplate="<b>%{text}</b><extra></extra>",
    )

    # Create shapes for event lines
    shapes = [dict(
        type="line",
        line=dict(color="gray", width=2.5),
        x0=date, x1=date,
        y0=0, y1=0.15,
        xref="x", yref="y"
    ) for date in events.date.tolist()]

    # Add horizontal line at y=0
    trace_horizontal = go.Scatter(
        x=[x_min, x_max],
        y=[0, 0],
        mode='lines',
        line=dict(color='black', dash='dash'),
        showlegend=False,
        hoverinfo="skip"
    )

    # Layout
    layout = go.Layout(
        # title="Policy Change Index for North Korea",
        margin=dict(l=50, r=50, t=20, b=50), 
        showlegend=False,
        hovermode="x",
        hoverdistance=16,
        spikedistance=-1,
        yaxis=dict(
            title="Monthly PCI for North Korea",
            linecolor='black',
            range=[min(df.y.min(), -0.01), max(df.y.max(), 0.16)],
            zeroline=False
        ),
        shapes=shapes,
        hoverlabel=dict(
            font=dict(size=12, color="white")
        ),
        xaxis=dict(
            type="date",
            range=[x_min, x_max],
            showgrid=False,
            spikethickness=2,
            spikecolor="black",
            showspikes=True,
            spikedash="dot",
            spikemode="toaxis+across+marker",
            spikesnap="cursor",
            dtick="M2",
            tickformat="%b\n%Y",
            hoverformat="%Y-%m-%d",
            showline=True,
            linecolor='black',
            linewidth=2,
            ticks="outside",
            tickfont=dict(family='Arial', size=12, color='black'),
        )
    )

    # Create figure
    fig = go.Figure(data=[trace_main, trace_events, trace_horizontal], layout=layout)

    # Save the figure as an HTML file
    fig.write_html("plot1.html", full_html=True, include_plotlyjs='cdn')
    print("Plot saved as plot.html")

if __name__ == "__main__":
    create_nko_plotly_figure()