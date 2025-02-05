from Dependecies import *
#List out the data set available
class_dataset = list_datasets(full=True, filter="text-classification")
classification_dataset_names = [dataset.id for dataset in class_dataset]

#choose one of the text classification data
ag_news = load_dataset("fancyzhx/ag_news")

# Choose one of the text classification datasets
ag_news = load_dataset("fancyzhx/ag_news")


# Function to process a column in the dataset
def process_column(df, column_name, features, custom_map=None, is_label_name=False):
    # Check if the column exists in the dataframe
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the dataframe")
    
    if is_label_name:
        # Convert integer labels to string labels using the dataset's features
        df[f"{column_name}_name"] = df[column_name].apply(lambda row: features['label'].int2str(row))
        return df, None
    else:
        # Check for missing values
        if df[column_name].isnull().any():
            raise ValueError(f"Column '{column_name}' contains missing values")
        
        if custom_map:
            # Use a custom mapping if provided
            unique_values = df[column_name].unique()
            for value in unique_values:
                if value not in custom_map:
                    raise ValueError(f"Value '{value}' in column '{column_name}' is not found in custom map")
            df[f"{column_name}_int"] = df[column_name].map(custom_map)
            mapping = custom_map
        else:
            # Use LabelEncoder to map strings to integers
            Label_encoder = LabelEncoder()
            df[f"{column_name}_int"] = Label_encoder.fit_transform(df[column_name])
            mapping = dict(zip(Label_encoder.classes_, Label_encoder.transform(Label_encoder.classes_)))
        return df, mapping

# Process all splits (train, test, validation)
for split in ["train", "test"]:
    # Convert the split to a DataFrame
    df = ag_news[split].to_pandas()
    
    # Process the 'label' column to add string labels
    df, _ = process_column(df, column_name='label', features=ag_news[split].features, is_label_name=True)
    
    # Save the processed DataFrame back to the dataset (optional)
    ag_news[split] = df

    # Display the processed data for the current split
    print(f"Processed {split} split:")
    print(df.head())
    print("\n")