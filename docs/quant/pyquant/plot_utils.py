"""
Plotting utilities for financial data visualization.
"""
import numpy as np
import plotly.graph_objects as go
import pandas as pd



def plot_volatility_surface(
    df: pd.DataFrame,
    value_column: str | list[str],
    x_column: str = 'strike',
    y_column: str = 'time_to_maturity',
    title: str = None,
    z_axis_title: str = None,
    colorscale: str | list[str] = 'Viridis',
    opacity: float | list[float] = 1.0,
    width: int = 1000,
    height: int = 800
) -> go.Figure:
    """
    Create a 3D surface plot from a DataFrame with dark mode styling.
    Can plot a single surface or multiple surfaces from different columns.
    
    Arguments:
        df: DataFrame containing the data to plot
        value_column: Name(s) of the column(s) containing the z-axis values.
                     Can be a single string or a list of strings for multiple surfaces.
        x_column: Name of the column for x-axis (default: 'strike')
        y_column: Name of the column for y-axis (default: 'time_to_maturity')
        title: Plot title (default: '{value_column} Surface' or 'Volatility Surfaces')
        z_axis_title: Z-axis title (default: value_column or 'Value')
        colorscale: Plotly colorscale name(s). Can be a single string or a list.
                   Default: 'Viridis'. For multiple surfaces, uses different colorscales.
        opacity: Opacity of the surface(s). Can be a single float or a list.
                 Default: 1.0. Useful for overlapping surfaces.
        width: Figure width in pixels (default: 1000)
        height: Figure height in pixels (default: 800)
    
    Returns:
        Plotly Figure object
    """
    # Normalize inputs to lists
    if isinstance(value_column, str):
        value_columns = [value_column]
    else:
        value_columns = value_column
    
    if isinstance(colorscale, str):
        colorscales = [colorscale] * len(value_columns)
    else:
        if len(colorscale) != len(value_columns):
            # Use default colorscales if not enough provided
            default_colorscales = ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Turbo']
            colorscales = [colorscale[i] if i < len(colorscale) else default_colorscales[i % len(default_colorscales)]
                          for i in range(len(value_columns))]
        else:
            colorscales = colorscale
    
    if isinstance(opacity, (int, float)):
        opacities = [opacity] * len(value_columns)
    else:
        if len(opacity) != len(value_columns):
            opacities = [opacity[i] if i < len(opacity) else 0.7 for i in range(len(value_columns))]
        else:
            opacities = opacity
    
    # Get common axes from first column (assuming all columns share same x/y structure)
    first_surface_data = df.pivot_table(
        values=value_columns[0],
        index=x_column,
        columns=y_column,
        aggfunc='first'
    ).sort_index()
    
    x_values = first_surface_data.index.values
    y_values = first_surface_data.columns.values
    X, Y = np.meshgrid(x_values, y_values)
    
    # Create surfaces for each column
    surfaces = []
    for i, col in enumerate(value_columns):
        surface_data = df.pivot_table(
            values=col,
            index=x_column,
            columns=y_column,
            aggfunc='first'
        ).sort_index()
        
        Z = surface_data.values.T
        
        # Create surface with name for legend
        surface = go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale=colorscales[i],
            opacity=opacities[i],
            name=col.replace("_", " ").title(),
            showscale=(i == 0),  # Only show colorbar for first surface
            colorbar=dict(
                title=dict(
                    text=z_axis_title if z_axis_title else col.replace("_", " ").title(),
                    font=dict(color='white')
                ),
                tickfont=dict(color='white')
            ) if i == 0 else None
        )
        surfaces.append(surface)
    
    # Create figure with all surfaces
    fig = go.Figure(data=surfaces)
    
    # Determine axis titles based on column names
    x_axis_title = x_column.replace("_", " ").title()
    y_axis_title = y_column.replace("_", " ").title()
    
    # Default titles
    if title is None:
        if len(value_columns) == 1:
            title = f'{value_columns[0].replace("_", " ").title()} Surface'
        else:
            title = 'Volatility Surfaces'
    
    if z_axis_title is None:
        if len(value_columns) == 1:
            z_axis_title = value_columns[0].replace("_", " ").title()
        else:
            z_axis_title = 'Value'
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(color='white')
        ),
        scene=dict(
            xaxis=dict(
                title=dict(text=x_axis_title, font=dict(color='white')),
                tickfont=dict(color='white'),
                gridcolor='rgba(255, 255, 255, 0.1)',
                backgroundcolor='rgb(20, 20, 20)'
            ),
            yaxis=dict(
                title=dict(text=y_axis_title, font=dict(color='white')),
                tickfont=dict(color='white'),
                gridcolor='rgba(255, 255, 255, 0.1)',
                backgroundcolor='rgb(20, 20, 20)'
            ),
            zaxis=dict(
                title=dict(text=z_axis_title, font=dict(color='white')),
                tickfont=dict(color='white'),
                gridcolor='rgba(255, 255, 255, 0.1)',
                backgroundcolor='rgb(20, 20, 20)'
            ),
            bgcolor='rgb(20, 20, 20)'
        ),
        paper_bgcolor='rgb(20, 20, 20)',
        plot_bgcolor='rgb(20, 20, 20)',
        width=width,
        height=height
    )
    
    return fig

