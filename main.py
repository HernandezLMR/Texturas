import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from momentos import Instance

def plot_features_boxplot(textured_instances, flat_instances):
    # Collect data
    data = []
    for instance in textured_instances:
        data.append({
            'Dataset': 'Textured',
            'Contrast': instance.contrast,
            'Homogeneity': instance.homogeneity,
            'Energy': instance.energy,
            'Entropy': instance.entropy,
            'Median': instance.median,
            'Variance': instance.variance,
            'Asymmetry': instance.asymmetry,
            'Kurtosis': instance.kurtosis
        })
    for instance in flat_instances:
        data.append({
            'Dataset': 'Flat',
            'Contrast': instance.contrast,
            'Homogeneity': instance.homogeneity,
            'Energy': instance.energy,
            'Entropy': instance.entropy,
            'Median': instance.median,
            'Variance': instance.variance,
            'Asymmetry': instance.asymmetry,
            'Kurtosis': instance.kurtosis
        })

    # Convert data to DataFrame
    df = pd.DataFrame(data)

    # Print median and standard deviation for each feature and dataset
    print("Median and Standard Deviation for each feature by dataset:\n")
    for feature in df.columns[1:]:  # Skip 'Dataset' column
        for dataset in ['Textured', 'Flat']:
            feature_data = df[df['Dataset'] == dataset][feature]
            median_value = feature_data.median()
            std_dev = feature_data.std()
            print(f"{feature} - {dataset}: Median = {median_value:.2f}, Std Dev = {std_dev:.2f}")
        print()  # Blank line between features for readability

    # Set up the matplotlib figure
    plt.figure(figsize=(14, 10))

    # Create a box plot for each feature
    for i, feature in enumerate(df.columns[1:], 1):  # Skip 'Dataset' column
        plt.subplot(2, 4, i)
        sns.boxplot(x='Dataset', y=feature, data=df)
        plt.title(f'{feature} Comparison')
        plt.xlabel('')
    
    plt.tight_layout()
    plt.show()

def main():
    # Load all images
    texture_database = "./DS1"
    flats_database = "./DS2"
    textured_instances = Instance.load_images_from_directory(texture_database)
    flat_instances = Instance.load_images_from_directory(flats_database)

    # Plot box plots
    plot_features_boxplot(textured_instances, flat_instances)

if __name__ == "__main__":
    main()