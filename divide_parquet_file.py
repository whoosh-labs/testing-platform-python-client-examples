import pandas as pd

# Assuming 'train_with_embedding.parquet' is the input file
input_file_path = "/raga/examples/assets/train_with_embedding.parquet"

# Read the entire Parquet file
all_data = pd.read_parquet(input_file_path)
print(len(all_data))
print(all_data.shape[1])
# Calculate the number of datapoints in each chunk
chunk_size = 50000
num_chunks = all_data.size // chunk_size
print(num_chunks)
# Split the data into chunks
data_chunks = [all_data.iloc[i * chunk_size: (i + 1) * chunk_size] for i in range(num_chunks + 1)]

 # Save each chunk to a separate Parquet file
for i, chunk in enumerate(data_chunks):
    output_file_path = f"/Users/rupalitripathi/IdeaProjects/testing-platform-python-client/raga/examples/assets/chunk50k_{i + 1}_train_with_embedding.parquet"
    chunk.to_parquet(output_file_path, index=False)
    print(f"Saved {len(chunk)} datapoints to {output_file_path}")
