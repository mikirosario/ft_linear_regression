import pandas as pd
import plotly.express as px

def load_csv(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

def calculate_squared_errors(df: pd.DataFrame, dependent_var: str) -> pd.DataFrame:
    mean = df[dependent_var].mean()
    deviations = df[dependent_var] - mean
    squared_deviations = deviations ** 2
    squared_deviations.name = f'sq_{dependent_var}'
    return squared_deviations.to_frame()

def print_sum_of_squared_errors(squared_df: pd.DataFrame):
    total_sum_of_squares = squared_df.sum().item()
    print(f"Total Sum of Squares: {total_sum_of_squares}")

def generate_plot(df: pd.DataFrame, dependent_var: str, plot_title: str, file_name: str):
    fig = px.scatter(df, x=df.index, y=dependent_var, title=f'{plot_title}')
    fig.add_hline(y=df[dependent_var].mean(), line_dash="dash", line_color="red", 
                  annotation_text="Mean", annotation_position="top right")
    fig.write_image(f"{file_name}.png")

df = load_csv('../data/data.csv')
squared_errors = calculate_squared_errors(df, 'price')
print_sum_of_squared_errors(squared_errors) # Model is good if it reduces this value by a significant amount
generate_plot(df, 'price', 'Deviations of Prices From Mean', 'deviations')
generate_plot(squared_errors, 'sq_price', 'Deviations of Squared Prices From Mean', 'squared_deviations')
