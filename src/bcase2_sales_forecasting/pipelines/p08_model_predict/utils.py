import matplotlib.pyplot as plt
import seaborn as sns

def create_actual_vs_predicted_plot(df_results, df_full, parameters):
    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot actual sales
    ax.plot(df_results['Date'], df_results['Actual_Sales_EUR'], label='Actual Sales EUR', color='green')

    # Determine the end index of the training data
    train_end_idx = len(df_full) - 1

    # Plot train predictions in red
    ax.plot(df_results['Date'][:train_end_idx], df_results['Predicted_Sales_EUR'][:train_end_idx], label='Train Predictions', color='red')

    # Plot future predictions in blue
    ax.plot(df_results['Date'][train_end_idx:], df_results['Predicted_Sales_EUR'][train_end_idx:], label='Test Predictions', color='blue')

    # Details
    ax.set_title(f"Actual vs Predicted Sales for Product {parameters['target_product']}")
    ax.set_xlabel('Year-Month')
    ax.set_ylabel('Sales €')
    # ax.set_xticks(rotation=45)

    # Decorations
    ax.legend()
    ax.grid(alpha=0.3) 
    plt.tight_layout()
    sns.despine()

    # Return the Figure object
    return fig