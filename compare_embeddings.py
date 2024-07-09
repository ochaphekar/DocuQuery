#from langchain.embeddings import OpenAIEmbeddings
from langchain.evaluation import load_evaluator
from langchain_community.document_loaders import DirectoryLoader

# For embeddings
from langchain_openai import OpenAIEmbeddings


def main():
    # Get embedding for a word.
    if os.path.exists(embeddings_file):
        os.remove(embeddings_file)
        print(f"Deleted existing embeddings file: {embeddings_file}")
    embedding_function = OpenAIEmbeddings()
    vector = embedding_function.embed_query("apple")
    print(f"Vector for 'apple': {vector}")
    print(f"Vector length: {len(vector)}")

    # Compare vector of two words
    evaluator = load_evaluator("pairwise_embedding_distance")
    words = ("apple", "iphone")
    x = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1])
    print(f"Comparing ({words[0]}, {words[1]}): {x}")


if __name__ == "__main__":
    main()
