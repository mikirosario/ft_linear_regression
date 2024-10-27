import pandas as pd
import plotly.express as px

def load_csv(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

def generate_plot(df: pd.DataFrame, x_col: str, y_col: str, plot_title: str, file_name: str):
    fig = px.scatter(df, x=x_col, y=y_col, title=plot_title)
    fig.write_image(f"{file_name}.png")

# Load the dataset
data = load_csv('../data/data.csv')

# Generate the plot
generate_plot(data, 'km', 'price', 'Price vs. Mileage', 'price_mileage_plot')